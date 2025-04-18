import torch
import time

def distributed_matmul(A_cpu, B_cpu, world_size):
    """
    Distributed matrix multiplication across multiple GPUs.
    Returns the execution time in seconds.
    """
    # Ensure inputs are divisible, or pad A_cpu
    N = A_cpu.size(0)
    chunk_size = (N + world_size - 1) // world_size  # ceiling division
    pad_rows = chunk_size * world_size - N

    if pad_rows > 0:
        pad = torch.zeros(pad_rows, A_cpu.size(1), dtype=A_cpu.dtype)
        A_cpu = torch.cat([A_cpu, pad], dim=0)

    # 1) Split A into chunks
    A_parts = torch.chunk(A_cpu, world_size, dim=0)

    # 2) Move data to GPUs
    A_parts = [A_chunk.contiguous().to(f"cuda:{i}") for i, A_chunk in enumerate(A_parts)]
    B_parts = [B_cpu.contiguous().to(f"cuda:{i}") for i in range(world_size)]

    # 3) Warm-up run (not timed)
    _ = [torch.matmul(A_parts[i], B_parts[i]) for i in range(world_size)]
    for i in range(world_size):
        torch.cuda.synchronize(f"cuda:{i}")

    # 4) Start timing
    start = time.time()

    C_parts = [torch.matmul(A_parts[i], B_parts[i]) for i in range(world_size)]

    # 5) Simulate gathering full result back to CPU
    C_cpu_parts = [C_parts[i].cpu() for i in range(world_size)]
    final_result = torch.cat(C_cpu_parts, dim=0)

    # 6) Sync all GPUs after computation
    for i in range(world_size):
        torch.cuda.synchronize(f"cuda:{i}")
    end = time.time()

    execution_time = end - start

    # Simple verification metric
    result_sum = final_result.sum().item()

    print(f"{world_size}-GPU result sum: {result_sum:.2f}")
    print(f"{world_size}-GPU time: {execution_time:.2f} seconds")
    
    return execution_time
