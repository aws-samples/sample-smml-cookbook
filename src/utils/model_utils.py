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

def benchmark_batch_sizes(
    model_name: str,
    seq_len: int,
    min_new_tokens: int,
    batch_sizes,
    dtype=torch.bfloat16,
    sharding: str = "none",
    world_size: int = 1,
    rank: int = 0
):
    """
    Run benchmarks across different batch sizes and return results.

    Initializes the process group for distributed strategies if needed.
    """
    # Initialize distributed process group once
    if not dist.is_available():
        raise RuntimeError("Distributed package is not available")
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )

    results = []

    for batch in batch_sizes:
        if rank == 0:
            print(f"\n🔁 Running batch size = {batch}")

        elapsed_s, tokens_generated, metrics, cost = benchmark_llm(
            model_name=model_name,
            batch=batch,
            seq_len=seq_len,
            min_new_tokens=min_new_tokens,
            dtype=dtype,
            sharding=sharding,
            world_size=world_size
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

    # Destroy process group if this function initialized it
    if dist.is_initialized():
        dist.destroy_process_group()

    return results


def load_sharded_model(
    model_name: str,
    dtype: torch.dtype,
    sharding: Literal["none", "data", "fsdp", "tensor", "pipeline", "device_map"] = "none",
    world_size: int = 1,
    seq_len: int = 1,
    min_new_tokens: int = 1
) -> torch.nn.Module:
    """
    Load a model with the specified sharding strategy.

    Args:
        model_name: HF model ID
        dtype: torch.float16 or torch.bfloat16
        sharding: one of the supported modes
        world_size: number of GPUs/processes

    Returns:
        Initialized and eval()'d model
    """
    redirect_output_to_file(int(os.environ.get("LOCAL_RANK", 0)))

    # Single-GPU, no parallelism
    if sharding == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(torch.cuda.current_device()).eval()

    # DataParallel: replicate model, shard input
    elif sharding == "data":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(torch.cuda.current_device())
        model = torch.nn.DataParallel(model).eval().module

    # FSDP: fully shard parameters across ranks
    elif sharding == "fsdp":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(local_rank)
        model = FSDP(model).eval()

    # DeepSpeed inference: tensor parallel only
    elif sharding == "tensor":
        num_mp = world_size
        max_tokens =  seq_len + min_new_tokens  # force planning for only what we need

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )
        model.eval()

        ds_config = {
            "replace_with_kernel_inject": True,
            "enable_cuda_graph": False,
            "tensor_parallel": {
                "enabled": True,
                "tp_size": num_mp
            },
            # Optional tuning knobs to constrain token planning
            "max_tokens": max_tokens
        }

        model = deepspeed.init_inference(
            model,
            config=ds_config,
            dtype=dtype,
            replace_method="auto",
            replace_with_kernel_inject=True
        )

    # Pipeline parallel unsupported in inference API
    elif sharding == "pipeline":
        raise NotImplementedError(
            "DeepSpeed inference does not support pipeline parallelism; "
            "use deepspeed.runtime.pipe.PipelineModule for pipeline execution."
        )

    # Hugging Face Accelerate: device_map auto-sharding
    elif sharding == "device_map":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto"
        ).eval()

    else:
        raise ValueError(f"Unsupported sharding strategy: {sharding}")

    return model


def benchmark_llm(
    model_name: str,
    batch: int,
    seq_len: int,
    min_new_tokens: int,
    dtype=torch.bfloat16,
    sharding: str = "none",
    world_size: int = 1
):
    """
    Benchmark a single forward pass (prompt + generation) on specified parallelism.
    Returns elapsed GPU seconds, tokens generated, performance metrics, and cost.
    """

    # Load tokenizer and model
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = load_sharded_model(model_name, dtype, sharding, world_size, seq_len, min_new_tokens)
    model.config.use_cache = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # Prepare prompt
    prompt = tok.bos_token + ''.join(random.choices(string.ascii_letters, k=seq_len - 1))
    device = next(model.parameters()).device
    inputs = tok([prompt] * batch, return_tensors="pt", padding=True).to(device)

    # Warm-up
    with torch.inference_mode():
        for _ in range(3):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # Simulate new tokens
    prompt_ids = torch.randint(
        1, tok.vocab_size - 1,
        (batch, seq_len + min_new_tokens),
        device=device
    )

    start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.inference_mode():
        start_evt.record()
        _ = model(prompt_ids)
        end_evt.record()
    torch.cuda.synchronize()

    elapsed_s = start_evt.elapsed_time(end_evt) / 1e3
    tokens_generated = batch * min_new_tokens

    # if os.environ.get("LOCAL_RANK", 0) == 0: 
    metrics = theoretical_tflops(model, min_new_tokens, seq_len, batch, elapsed_s)
    cost = estimate_cost_per_1m_tokens(tokens_generated, elapsed_s)

    # Cleanup
    del model, tok, inputs, prompt_ids
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return elapsed_s, tokens_generated, metrics, cost


def theoretical_tflops(
    model,
    min_new_tokens: int,
    seq_len: int,
    batch: int,
    elapsed_s: float,
    peak_tflops: float = 121.0
):
    """
    Compute FLOPs, bandwidth, and arithmetic intensity for one forward.
    """
    params = sum(p.numel() for p in model.parameters())
    dtype_bytes = torch.finfo(next(model.parameters()).dtype).bits // 8
    tokens = seq_len + min_new_tokens
    hidden = model.config.hidden_size

    flops = 2 * params * batch * tokens
    weight_bytes = params * dtype_bytes
    kv_bytes = batch * seq_len * hidden * dtype_bytes * 2
    out_bytes = batch * min_new_tokens * hidden * dtype_bytes
    bytes_moved = weight_bytes + kv_bytes + out_bytes
    arith_int = flops / bytes_moved
    tflops_s = flops / elapsed_s / 1e12
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Batch={batch} | Seq={seq_len}+{min_new_tokens}")
        print(f"Elapsed GPU time: {elapsed_s:.4f}s | TFLOP/s: {tflops_s:.1f} | AI: {arith_int:.2f} FLOP/B")

    return {
        "local_gflops": tflops_s * 1000,
        "aggregated_gflops": tflops_s * 1000,
        "total_flops": flops,
        "estimated_memory_bytes": bytes_moved,
        "arithmetic_intensity": arith_int,
    }


def estimate_cost_per_1m_tokens(
    tokens_generated: int,
    elapsed: float,
    price_per_hour: float = 1.21
) -> float:
    """
    Estimate and print cost per 1M tokens using throughput and GPU price.
    """
    throughput = tokens_generated / elapsed
    price_per_sec = price_per_hour / 3600
    cost_per_token = price_per_sec / throughput
    cost_per_1m = cost_per_token * 1_000_000

    # Only print on rank 1
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"🧠 Tokens generated: {tokens_generated}")
        print(f"⚡ Throughput: {throughput:.5f} tokens/sec")
        print(f"⏱️ Total time: {elapsed:.5f} sec")
        print(f"💸 Cost per 1M tokens: ${cost_per_1m:.5f}")

    return cost_per_1m

def reset_distributed_and_clear_memory(master_addr: str = "127.0.0.1",
                                       master_port: str = "12355"):
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

    # 1) Destroy any existing process group
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # 2) Kill any OS process listening on master_port
    try:
        pids = subprocess.check_output(
            f"lsof -ti tcp:{master_port}", shell=True
        ).decode().split()
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass

    # 3) Unset environment variables
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)

    # 4) Clear Python GC
    gc.collect()

    # 5) Clear PyTorch/CUDA caches
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("✅ Distributed env torn down and memory cleared.")


# 1) Top-level worker
def _distributed_worker(
    rank: int,
    model_name: str,
    seq_len: int,
    min_new_tokens: int,
    batch_sizes: list[int],
    dtype: torch.dtype,
    sharding: str,
    world_size: int,
    shared_results
):
    # 1️⃣ Tell DeepSpeed/PyTorch what our local rank is
    os.environ["LOCAL_RANK"] = str(rank)
    
    # 2️⃣ Bind this process to GPU `rank`
    torch.cuda.set_device(rank)

    # Run the benchmark for this rank
    res = benchmark_batch_sizes(
        model_name=model_name,
        seq_len=seq_len,
        min_new_tokens=min_new_tokens,
        batch_sizes=batch_sizes,
        dtype=dtype,
        sharding=sharding,
        world_size=world_size,
        rank=rank
    )
    # Only rank 0 writes
    if rank == 0:
        shared_results.extend(res)



# 2) Launcher
def run_distributed_benchmark(
    model_name: str,
    seq_len: int,
    min_new_tokens: int,
    batch_sizes: list[int],
    dtype: torch.dtype,
    sharding: str,
    world_size: int,
):
    mp.set_start_method("fork", force=True)
    
    with Manager() as manager:
        shared_results = manager.list()

        mp.spawn(
            _distributed_worker,
            args=(
                model_name,
                seq_len,
                min_new_tokens,
                batch_sizes,
                dtype,
                sharding,
                world_size,
                shared_results
            ),
            nprocs=world_size,
            join=True
        )

        return list(shared_results)


def _llama_worker(rank, model_name, ds_config_path, max_new_tokens, seq_len, world_size, data_size, shared_results):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")  # use a non-default port
    redirect_output_to_file(rank)

    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config_path)

    # Generate local share of prompts
    prompts_per_rank = data_size // world_size
    remainder = data_size % world_size
    if rank < remainder:
        prompts_per_rank += 1

    prompt = "DeepSpeed is " + "great " * (seq_len // 2)

    local_outputs = []
    start = time.time()
    for _ in range(prompts_per_rank):
        inputs = tokenizer(prompt, return_tensors="pt").to(model_engine.device)
        with torch.no_grad():
            outputs = model_engine.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            local_outputs.append(decoded)
    end = time.time()

    if rank == 0:
        shared_results.append({
            "outputs": local_outputs,
            "elapsed": end - start
        })

def redirect_output_to_file(rank):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ds_worker_{rank}.log")

    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())


def run_deepspeed_inference(model_name: str,
                            ds_config: dict,
                            world_size: int = 2,
                            max_new_tokens: int = 50,
                            seq_len: int = 32,
                            data_size: int = 10):
    mp.set_start_method("fork", force=True)

    with Manager() as manager:
        shared_results = manager.list()
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        ds_config_path = "results/ds_temp_config.json"
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f)

        mp.spawn(
            _llama_worker,
            args=(model_name, ds_config_path, max_new_tokens, seq_len, world_size, data_size, shared_results),
            nprocs=world_size,
            join=True,
        )

        os.remove(ds_config_path)

        # Only rank 0 appends output, so return first element
        result = shared_results[0] if shared_results else {}
        print(f"Inference completed in {result.get('elapsed', 0):.2f} seconds with a World Size={world_size}.")
        return result.get("outputs", [])