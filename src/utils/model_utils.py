import sys
import os
import gc
import time
import random
import string
import json
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Literal
import torch.multiprocessing as mp
from multiprocessing import Manager
import logging
import transformers
from math import ceil
import multiprocessing.util
import subprocess
import psutil
from torch.optim import AdamW
from deepspeed.pipe import PipelineModule
multiprocessing.util.log_to_stderr().setLevel("ERROR")

def benchmark_batch_sizes(
    model_name: str,
    seq_len: int,
    batch_sizes,
    dtype=torch.bfloat16,
    sharding: bool = False,
    world_size: int = 1,
    rank: int = 0,
    ds_config: dict = {}
):
    """
    Run benchmarks across different batch sizes and return results.

    Initializes the process group for distributed strategies if needed.
    """
    # Initialize distributed process group once
    if not dist.is_available():
        raise RuntimeError("Distributed package is not available")
    if not dist.is_initialized() and world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12355")
        deepspeed.init_distributed(
            world_size=world_size,
            rank=rank
        )

    results = []
    try: 
        model = load_sharded_model(model_name, dtype, sharding, world_size, ds_config)
    
        tok = AutoTokenizer.from_pretrained(model_name)

        for batch in batch_sizes:
            if rank == 0:
                print(f"\n🔁 Running batch size = {batch}")

            elapsed_s, tokens_generated, metrics, cost = benchmark_llm(
                model=model,
                tok=tok,
                batch=batch,
                seq_len=seq_len,
                dtype=dtype,
                sharding=sharding,
                world_size=world_size,
                ds_config=ds_config
            )
            # if rank == 0:
            results.append({
                "batch_size": batch,
                "world_size": world_size,
                "avg_time_seconds": elapsed_s,
                **metrics,
                "cost_per_1m_tokens": cost,
            })

            # Cleanup between runs
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
    except Exception as e:
        _cleanup_all()
        print(f"benchmark failed: {e}")
        return None
    # Destroy process group if this function initialized it
    if dist.is_initialized():
        dist.destroy_process_group()

    return results

def _cleanup_all():
    # 1) kill every child (and grandchild, etc.) of this process
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()   # polite SIGTERM
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()        # force SIGKILL if needed
        except psutil.NoSuchProcess:
            pass

    # 2) tear down torch.distributed
    if dist.is_initialized():
        dist.destroy_process_group()

    # 3) free GPU memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def load_sharded_model(
    model_name: str,
    dtype: torch.dtype,
    sharding: bool = False,
    world_size: int = 1,
    ds_config: dict = {}
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    if not sharding:
        return model.to(torch.cuda.current_device()).train()
    
    pipeline_cfg = ds_config.get("pipeline", {})
    if pipeline_cfg.get("enabled", False):
        # 1) Extract top‐level submodules into a list
        layers = [module for _, module in model.named_children()]
        sequential_model = torch.nn.Sequential(*layers)

        # 2) Read pipeline settings
        num_stages = pipeline_cfg.get("stages", 1)
        partition_method = pipeline_cfg.get("partition_method", "parameters")
        checkpoint_interval = pipeline_cfg.get("activation_checkpoint_interval", 0)

        # 3) Build the PipelineModule
        model = PipelineModule(
            layers=list(sequential_model),
            loss_fn=torch.nn.CrossEntropyLoss(),
            num_stages=num_stages,
            partition_method=partition_method,
            activation_checkpoint_interval=checkpoint_interval
        )


    # Training path using DeepSpeed
    model = model.to(torch.cuda.current_device())
    model.train()

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
        dist_init_required=True
    )
    return model_engine



def shard_list(data, rank, world_size):
    total = len(data)
    if total == 0:
        return []
    shard_size = (total + world_size - 1) // world_size  # ceiling division
    start = rank * shard_size
    end = min(start + shard_size, total)
    return data[start:end] if start < total else []


def benchmark_llm(
    model,
    tok,
    batch: int,
    seq_len: int,
    dtype=torch.bfloat16,
    sharding: bool = False,
    world_size: int = 1,
    ds_config: dict = {}
):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    


    # model.gradient_checkpointing_enable()
    tok.pad_token = tok.eos_token
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # Build training prompts and targets
    prompts = [tok.bos_token + ''.join(random.choices(string.ascii_letters, k=seq_len - 1)) for _ in range(batch)]
    
    # Ensure divisibility for world_size
    if batch % world_size != 0:
        pad = world_size - (batch % world_size)
        prompts += [prompts[-1]] * pad

    local_prompts = shard_list(prompts, rank, world_size)
    if not local_prompts:
        print(f"[Rank {rank}] No data after sharding. Skipping.")
        return 0.0, 0, {}, 0.0

    tokenized = tok(local_prompts, return_tensors="pt", padding=True, truncation=True).to(device)


    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = input_ids.clone()
    # print(f"[Rank {rank}] input_ids.shape: {input_ids.shape}")
    # print(f"[Rank {rank}] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # print(f"[Rank {rank}] Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    # print(f"[Rank {rank}] model type: {type(model)}")
    # print(f"[Rank {rank}] has backward: {hasattr(model, 'backward')}")

    # Warm-up step
    for _ in range(2):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if hasattr(model, 'backward'):
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            model.zero_grad()

    torch.cuda.synchronize()
    start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.zero_grad()
    max_accum = 4
    accum_steps = min(max_accum, batch) 
    micro_bs = (input_ids.size(0) + accum_steps - 1) // accum_steps  # ceiling division
    if batch == 0 or accum_steps == 0 or micro_bs == 0:
        print(f"[Rank {rank}] Invalid batch setup. Skipping.")
        return 0.0, 0, {}, 0.0

    optimizer.zero_grad()

    # timed, accumulated step:
    start_evt.record()
    for i in range(accum_steps):
        lo = i * micro_bs
        hi = lo + micro_bs
        ids   = input_ids[lo:hi]
        masks = attention_mask[lo:hi]
        out   = model(input_ids=ids, attention_mask=masks, labels=ids)
        # scale the loss so the gradient = average over micro-batches
        (out.loss / accum_steps).backward()
    # now do one optimizer step for the whole “effective” batch:
    optimizer.step()
    optimizer.zero_grad()
    end_evt.record()

    torch.cuda.synchronize()
    elapsed_s = start_evt.elapsed_time(end_evt) / 1e3

    # total tokens processed = batch * seq_len
    tokens_processed = batch * seq_len

    metrics = theoretical_tflops(
        model=model,
        seq_len=seq_len,
        batch=batch,
        elapsed_s=elapsed_s
    )

    # Gather across ranks
    elapsed_tensor = torch.tensor([elapsed_s], device=device)
    tokens_tensor = torch.tensor([tokens_processed], device=device)

    gathered_elapsed = [torch.zeros_like(elapsed_tensor) for _ in range(world_size)]
    gathered_tokens = [torch.zeros_like(tokens_tensor) for _ in range(world_size)]

    if world_size > 1:
        dist.all_gather(gathered_elapsed, elapsed_tensor)
        dist.all_gather(gathered_tokens, tokens_tensor)
        if rank == 0:
            all_elapsed = [t.item() for t in gathered_elapsed]
            all_tokens = [t.item() for t in gathered_tokens]
            total_tokens = sum(all_tokens)
            avg_time = sum(all_elapsed) / len(all_elapsed)
            cost = estimate_cost_per_1m_tokens(total_tokens, avg_time)
        else:
            cost = 0.0
    else:
        cost = estimate_cost_per_1m_tokens(tokens_processed, elapsed_s)

    return elapsed_s, tokens_processed, metrics, cost


def theoretical_tflops(
    model,
    seq_len: int,
    batch: int,
    elapsed_s: float,
    backward_factor: float = 2.0,
    update_factor: float = 2.0,
    error_factor: float = 2, # These are all estimations so we must account for what isn't being calculated
):
    """
    Compute approximate TFLOPs and arithmetic intensity for one training step.
    seq_len: number of tokens per example
    """
    # 1) total params & dtype size
    params = sum(p.numel() for p in model.parameters())
    dtype_bytes = torch.finfo(next(model.parameters()).dtype).bits // 8

    # 2) base model hidden size
    base = model.module if hasattr(model, "module") else model
    hidden = base.config.hidden_size

    # 3) FLOPs: forward + backward + update
    fwd_flops = 2 * params * batch * seq_len
    total_flops = fwd_flops * (1 + backward_factor + update_factor)

    # 4) Bytes moved:
    #    forward: read weights + read KV activations + write outputs
    w_bytes  = params * dtype_bytes
    kv_bytes = batch * seq_len * hidden * dtype_bytes * 2
    out_bytes= batch * seq_len * hidden * dtype_bytes
    fwd_bytes = w_bytes + kv_bytes + out_bytes

    #    backward: same as forward
    bwd_bytes = fwd_bytes

    #    update: read + write weights
    upd_bytes = params * dtype_bytes * 2

    total_bytes = fwd_bytes + backward_factor * bwd_bytes + update_factor * upd_bytes

    # 5) Metrics
    arith_int = total_flops / total_bytes
    tflops_s  = error_factor * total_flops / elapsed_s / 1e12

    print(f"Batch={batch} | SeqLen={seq_len}")
    print(f"Elapsed GPU time: {elapsed_s:.4f}s | TFLOP/s: {tflops_s:.1f} | AI: {arith_int:.2f} FLOP/B")

    return {
        "tflops_s": tflops_s,
        "total_flops": total_flops,
        "estimated_memory_bytes": total_bytes,
        "arithmetic_intensity": arith_int,
    }




def estimate_cost_per_1m_tokens(
    tokens_generated: int,
    elapsed_s: float,
    price_per_hour: float = 1.21
) -> float:
    """
    Estimate cost per 1M tokens using throughput and GPU price.
    """
    throughput = tokens_generated / elapsed_s
    price_per_sec = price_per_hour / 3600
    cost_per_token = price_per_sec / throughput
    cost_per_1m = cost_per_token * 1_000_000

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"--------- OUTPUT BREAKDOWN ---------")
        print(f"🧠 Tokens generated: {tokens_generated}")
        print(f"⚡ Throughput: {throughput:.5f} tokens/sec")
        print(f"⏱️ Total time: {elapsed_s:.5f} sec")
        print(f"💸 Cost per 1M tokens: ${cost_per_1m:.5f}")
        print(f"------------------------------------")
    return cost_per_1m



def reset_distributed_and_clear_memory():
    """
    Tear down any existing NCCL process group, kill leftover listeners on master_port,
    unset MASTER_ADDR/MASTER_PORT, and free Python & CUDA memory.
    """
    import os
    import signal
    import subprocess
    import gc
    import torch
    import torch.distributed as dist

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12355")
    

    # 1) Destroy any existing process group
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print("✅ Destroyed process group.")

    # 2) Kill any OS process listening on master_port
    _cleanup_all()

    # 3) Unset environment variables
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)
   

    print("✅ Distributed env torn down and memory cleared.")


# 1) Top-level worker
def _distributed_worker(
    rank: int,
    model_name: str,
    seq_len: int,
    batch_sizes: list[int],
    dtype: torch.dtype,
    sharding: bool ,
    world_size: int,
    ds_config: dict,
    shared_results
):
    # 1️⃣ Tell DeepSpeed/PyTorch what our local rank is
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)  # 👈 Isolate GPU view
    
    # 2️⃣ Bind this process to GPU `rank`
    torch.cuda.set_device(rank)

    # Run the benchmark for this rank
    res = benchmark_batch_sizes(
        model_name=model_name,
        seq_len=seq_len,
        batch_sizes=batch_sizes,
        dtype=dtype,
        sharding=sharding,
        world_size=world_size,
        rank=rank,
        ds_config=ds_config
    )
    # Only rank 0 writes
    if rank == 0 and res is not None:
        shared_results.extend(res)
    elif rank == 0:
        print("❌ Benchmark returned None; skipping.")


# 2) Launcher
def run_distributed_benchmark(
    model_name: str,
    seq_len: int,
    batch_sizes: list[int],
    dtype: torch.dtype,
    sharding: bool,
    world_size: int,
    ds_config: dict,
):
    mp.set_start_method("fork", force=True)
    
    with Manager() as manager:
        shared_results = manager.list()

        mp.spawn(
            _distributed_worker,
            args=(
                model_name,
                seq_len,
                batch_sizes,
                dtype,
                sharding,
                world_size,
                ds_config,
                shared_results
            ),
            nprocs=world_size,
            join=True
        )

        return list(shared_results)


def redirect_output_to_file(rank):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ds_worker_{rank}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        level=logging.WARNING,  # or INFO
        format="%(asctime)s %(levelname)s %(message)s"
    )
