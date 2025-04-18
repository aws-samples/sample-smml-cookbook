import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import time
import os
import json
import time, torch, json, importlib
import src.utils.general_utils as gutils
from fvcore.nn import FlopCountAnalysis
import time, random, string
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_batch_sizes(
    model_name: str,
    seq_len: int,
    min_new_tokens: int,
    batch_sizes,
    dtype=torch.bfloat16,
):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.config.use_cache = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    device = torch.device("cuda:0")
    model = model.to(device).eval()

    results = []

    for batch in batch_sizes:
        elapsed_s, tokens_generated = benchmark_llm(model_name,
                         batch=batch, seq_len=seq_len, min_new_tokens=min_new_tokens, dtype=dtype)



        metrics = theoretical_tflops(model, min_new_tokens, seq_len, batch, elapsed_s)
        cost = estimate_cost_per_1m_tokens(tokens_generated, elapsed_s)

        results.append({
            "batch_size": batch,
            "world_size": 1,
            "avg_time_seconds": elapsed_s,
            **metrics,
            "cost_per_1m_tokens": cost,
        })

    return results

def distributed_resnet_inference(rank, world_size, test_cases, output_dir="results"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    results = []

    for cfg in test_cases:
        batch_size = cfg["batch_size"]
        num_batches = cfg.get("num_batches", 20)

        # Ensure the batch size is divisible
        assert batch_size % world_size == 0, f"Batch size {batch_size} must be divisible by world size {world_size}"
        local_batch_size = batch_size // world_size

        # Create local input per rank
        input_tensor = torch.randn(local_batch_size, 3, 224, 224).cuda(rank)

        # Build and wrap model
        model = models.resnet50(weights="DEFAULT").cuda(rank)
        model.eval()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        # Warm-up
        with torch.no_grad():
            _ = model(input_tensor)

        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()

        # Inference loop
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(input_tensor)

        torch.cuda.synchronize()
        dist.barrier()
        end = time.time()

        avg_time = end - start

        # Collect results from rank 0
        result = {
            "world_size": world_size,
            "batch_size_total": batch_size,
            "batch_size_per_gpu": local_batch_size,
            "num_batches": num_batches,
            "avg_time_seconds": round(avg_time, 4)
        }

        results.append(result)

        if rank == 0:
            print(f"✅ PASSED: {world_size} GPU(s) | Batch {batch_size} x {num_batches}")
            print(f"   Inference Time: {avg_time:.2f} sec")
            print("-" * 60)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"distributed_resnet_results_{world_size}gpus.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_file}")

    dist.destroy_process_group()


def distributed_main(rank, world_size, test_cases):
    distributed_resnet_inference(rank, world_size, test_cases)


def theoretical_tflops(model, min_new_tokens, seq_len, batch, elapsed_s):
    # ---------------------------------------------------------------
    # FLOPs and bytes for *one* full forward pass (KV‑cache on)
    # ---------------------------------------------------------------
    params      = sum(p.numel() for p in model.parameters())
    dtype_bytes = torch.finfo(model.dtype).bits // 8          # 2 for BF16
    tokens      = seq_len + min_new_tokens                    # all tokens in the prompt
    hidden      = model.config.hidden_size

    flops = 2 * params * batch * tokens                       # GEMM rules

    # Each layer’s weights are fetched *once* (reuse across batch & tokens)
    weight_bytes = params * dtype_bytes

    # KV cache traffic (read old K/V, write new K/V)
    kv_bytes  = batch * seq_len * hidden * dtype_bytes * 2
    out_bytes = batch * min_new_tokens * hidden * dtype_bytes

    bytes_moved = weight_bytes + kv_bytes + out_bytes
    arith_int   = flops / bytes_moved
    tflops_s    = flops / elapsed_s / 1e12

    print(f"Batch={batch}  Seq={seq_len}  +{min_new_tokens} tokens")
    print(f"Elapsed GPU time: {elapsed_s:.4f}s")
    print(f"TFLOP/s:           {tflops_s:.1f}")
    print(f"Arithmetic Int.:   {arith_int:.2f} FLOP/B")
    return {
        "local_gflops": tflops_s * 1000,
        "aggregated_gflops": tflops_s * 1000,
        "total_flops": flops,
        "estimated_memory_bytes": bytes_moved,
        "arithmetic_intensity": arith_int,
    }



def benchmark_llm(model_name: str,
                  batch: int,
                  seq_len: int,
                  min_new_tokens: int,
                  dtype=torch.bfloat16):
    """
    Returns (elapsed_gpu_seconds, tokens_generated).
    """
    # Load model + tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=dtype)
    model.config.use_cache = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    device = torch.device("cuda:0")
    model = model.to(device).eval()

    # Build prompt
    prompt = tok.bos_token + ''.join(random.choices(
        string.ascii_letters, k=seq_len - 1))
    inputs = tok([prompt] * batch,
                 return_tensors="pt").to(device)

    # Warm‑up
    with torch.inference_mode():
        for _ in range(3):
            _ = model(**inputs)
    torch.cuda.synchronize()

    # Single forward over full prompt + generated tokens
    prompt_ids = torch.randint(1, tok.vocab_size - 1,
                               (batch, seq_len + min_new_tokens),
                               device=device)

    start_evt, end_evt = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.inference_mode():
        start_evt.record()
        _ = model(prompt_ids)
        end_evt.record()
    torch.cuda.synchronize()

    elapsed_s = start_evt.elapsed_time(end_evt) / 1e3
    tokens_generated = batch * min_new_tokens
    return elapsed_s, tokens_generated

def estimate_cost_per_1m_tokens(
    tokens_generated: int,
    elapsed: float,
    price_per_hour: float = 1.21,  # g6.12xlarge (1x L4 GPU)
) -> None:
    """
    Print throughput, latency, and estimated cost per 1K tokens.

    Args:
        throughput_tokens_per_sec: Number of tokens/sec.
        tokens_generated: Total tokens generated.
        elapsed: Time elapsed in seconds.
        price_per_hour: GPU hourly cost (default: L4 on g6.12xlarge).
    """

    throughput       = tokens_generated / elapsed  # tokens per second

    price_per_sec = price_per_hour / 3600
    cost_per_token = price_per_sec / throughput
    cost_per_1m = cost_per_token * 1000000000

    print(f"🧠 Tokens generated: {tokens_generated}")
    print(f"B Batch Size: {tokens_generated / 100}")
    print(f"⚡ Throughput: {throughput:.2f} tokens/sec")
    print(f"⏱️ Total time: {elapsed:.2f} sec")
    print(f"💸 Cost per 1M tokens: ${cost_per_1m:.5f}")
    return cost_per_1m
