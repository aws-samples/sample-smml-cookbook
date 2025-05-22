import os
import json
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager
from torch.distributed.tensor import distribute_tensor, DeviceMesh, Shard, Replicate


def _distributed_gemm_worker(rank, world_size, rows, cols, out_cols, shared_results):
    import time
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'

    print(f"[Rank {rank}] Starting process on {device}")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Initialized NCCL process group with world_size={world_size}")

    # Allocate local chunk of A
    rows_per_rank = rows // world_size
    A_local = torch.randn((rows_per_rank, cols), device=device)
    print(f"[Rank {rank}] Created local shard of A: shape={A_local.shape} on {device}")

    # Initialize or receive B
    if rank == 0:
        B = torch.randn((cols, out_cols), device=device)
        print(f"[Rank {rank}] Created full matrix B: shape={B.shape} on {device}")
    else:
        B = torch.empty((cols, out_cols), device=device)
        print(f"[Rank {rank}] Allocated empty matrix B to receive broadcast: shape={B.shape} on {device}")

    # Broadcast B from rank 0 to all ranks
    dist.broadcast(B, src=0)
    print(f"[Rank {rank}] Completed broadcast of B")

    # Local matrix multiply
    print(f"[Rank {rank}] Performing matmul: A_local ({A_local.shape}) @ B ({B.shape})")
    start = time.time()
    out_local = A_local @ B
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"[Rank {rank}] Finished matmul. Output shape: {out_local.shape}. Time: {duration:.3f}s")

    # Prepare all-gather
    out_gathered = [torch.empty_like(out_local) for _ in range(world_size)]
    print(f"[Rank {rank}] Prepared buffers for all_gather")

    # All-gather outputs
    dist.all_gather(out_gathered, out_local)
    print(f"[Rank {rank}] Completed all_gather of local outputs")

    # Concatenate to form full output on all ranks (optional: only on rank 0)
    full_out = torch.cat(out_gathered, dim=0)
    print(f"[Rank {rank}] Full output assembled: shape={full_out.shape} (still on {device})")

    # Save result (on rank 0 only)
    if rank == 0:
        shared_results.append({
            "output_shape": full_out.shape,
            "rows_per_rank": rows_per_rank,
            "device": device
        })
        print(f"[Rank {rank}] Saved result to shared_results")

    # Clean up
    dist.destroy_process_group()
    print(f"[Rank {rank}] Destroyed process group and exiting")

def distributed_gemm():
    world_size = 2
    rows, cols, out_cols = 115_000, 60_000, 1024

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.set_start_method("fork", force=True)

    with Manager() as manager:
        shared_results = manager.list()

        mp.spawn(
            _distributed_gemm_worker,
            args=(world_size, rows, cols, out_cols, shared_results),
            nprocs=world_size,
            join=True
        )

        return list(shared_results)


def parse_shard_pattern(pattern):
    # Returns placements and mesh shape
    pattern = pattern.strip()
    if not pattern:
        return [Replicate(), Replicate()], (2, 2)  # Fully replicated

    parts = pattern.split("_")
    assert len(parts) in (1, 2), f"Invalid pattern: {pattern}"

    I_shard = parts[0] if parts[0].startswith("I") else ""
    J_shard = parts[1] if len(parts) > 1 and parts[1].startswith("J") else (parts[0] if parts[0].startswith("J") else "")

    I_axes = list(I_shard[1:]) if I_shard else []
    J_axes = list(J_shard[1:]) if J_shard else []

    # Disallow conflicting sharding (e.g., I and J both across multiple axes)
    if len(I_axes) > 1 and len(J_axes) > 0:
        raise ValueError(f"Invalid pattern '{pattern}': cannot shard both I and J across different axes")
    if len(J_axes) > 1 and len(I_axes) > 0:
        raise ValueError(f"Invalid pattern '{pattern}': cannot shard both I and J across different axes")
    
    if I_axes == [] and J_axes == []:
        return [Replicate(), Replicate()], (2, 2)  # Fully replicated
     
    placements = [Replicate(),Replicate()]
    for index, axis in enumerate(["y", "x"]):
        if axis in I_axes:
            placements[index] = Shard(0)  # Shard row dimension
        elif axis in J_axes:
            placements[index] = Shard(1)  # Shard col dimension

    return placements, (2, 2) if any(isinstance(p, Shard) for p in placements) else (1, 1)

def distributed_matrix_shard(rank, world_size, test_config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    shard_pattern = test_config["shard_pattern"]
    placements, mesh_shape = parse_shard_pattern(shard_pattern)
    device_mesh = DeviceMesh("cuda", torch.arange(world_size).reshape(mesh_shape))

    global_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4).cuda()
    dist.broadcast(global_tensor, src=0)

    dtensor = distribute_tensor(global_tensor, device_mesh, placements=placements)
    local_tensor = dtensor.to_local().cpu().tolist()
    local_shape = list(dtensor.to_local().shape)

    # Gather all local shards and shapes at rank 0
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_tensor)
    shapes = [None for _ in range(world_size)]
    dist.all_gather_object(shapes, local_shape)

    if rank == 0:
        result = {
            "shard_pattern": shard_pattern,
            "placements": [type(p).__name__ for p in placements],
            "mesh_shape": mesh_shape,
            "local_shapes": shapes,
            "all_ranks": gathered
        }
        filename = f"results/matrix_shard_{world_size}gpus_{shard_pattern or 'replicated'}.json"
        os.makedirs("results", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

    dist.destroy_process_group()


def multi_layer_chain(batch_size, D, F, num_layers=2, warmup_iters=2, iters=5, device=torch.device('cuda')):
    """
    Performs a chain of matrix multiplications in FP16 on GPU:
      1) X: [B, D] x W0: [D, F] -> [B, F]
      2) for layer i in range(1, num_layers): 
           output_{i-1}: [B, F] x Wi: [F, F] -> [B, F]
    Followed by a ReLU activation after each layer. 
    Returns the average time (in seconds) over `iters` iterations.
    """
    # Create input [B, D] in FP16
    x = torch.randn(batch_size, D, dtype=torch.float16, device=device)
    # First weight: [D, F]
    w0 = torch.randn(D, F, dtype=torch.float16, device=device)
    # Subsequent weights: [F, F]
    weights = [w0] + [
        torch.randn(F, F, dtype=torch.float16, device=device)
        for _ in range(num_layers - 1)
    ]

    # Warmup iterations
    for _ in range(warmup_iters):
        tmp = x @ weights[0]
        tmp = torch.relu(tmp)
        for i in range(1, num_layers):
            tmp = tmp @ weights[i]
            tmp = torch.relu(tmp)
        x = tmp
    torch.cuda.synchronize(device)

    # Timed iterations
    times = []
    for _ in range(iters):
        start = time.time()
        tmp = x @ weights[0]
        tmp = torch.relu(tmp)
        for i in range(1, num_layers):
            tmp = tmp @ weights[i]
            tmp = torch.relu(tmp)
        x = tmp
        torch.cuda.synchronize(device)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time

def calculate_chain_flops(batch_size, D, F, num_layers=2):
    """
    Computes the theoretical total FLOPs for the multi-layer chain:
      1st layer: [B, D] x [D, F] => 2 * B * D * F
      Subsequent (num_layers-1) layers: each [B, F] x [F, F] => 2 * B * F * F
    """
    first_layer = 2.0 * batch_size * D * F
    other_layers = (num_layers - 1) * (2.0 * batch_size * F * F)
    return first_layer + other_layers

def calculate_memory_bytes(batch_size, D, F, num_layers=2):
    """
    Estimates the memory traffic (in bytes) for the multi-layer chain. 
    Assumes:
      - Input [B, D] is read once.
      - Weight W0 [D, F] is read once.
      - Each layer writes an output of shape [B, F].
    All tensors are FP16 (2 bytes/element).
    """
    bytes_per_elem = 2.0
    # 1st layer: input + weight + output
    mem_first_layer = (batch_size * D + D * F + batch_size * F) * bytes_per_elem
    # Each subsequent layer: read [B,F] and weight, then write [B,F]
    mem_other_layers = (num_layers - 1) * ((batch_size * F + F * F + batch_size * F) * bytes_per_elem)
    return mem_first_layer + mem_other_layers

def distributed_run_test_suite(world_size, test_cases, warmup_iters=2, iters=5, output_dir="results"):
    """
    Runs the provided test cases on each GPU (process) and aggregates performance across GPUs.
    For each test case, each GPU computes its local GFLOPs and then they are summed via all_reduce.
    Only rank 0 writes the aggregated results to a JSON file.
    """
    results = []
    rank = dist.get_rank()
    
    for cfg in test_cases:
        B = cfg["batch_size"]
        D = cfg["D"]
        F = cfg["F"]
        L = cfg["num_layers"]

        try:
            avg_time = multi_layer_chain(B, D, F, L, warmup_iters=warmup_iters, iters=iters,
                                         device=torch.device(f"cuda:{rank}"))
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                if rank == 0:
                    print(f"Skipping incompatible shape: B={B}, D={D}, F={F}, layers={L}. Error: {e}")
                continue
            else:
                raise

        # Compute per-GPU (local) values.
        local_flops = calculate_chain_flops(B, D, F, L)
        local_mem_bytes = calculate_memory_bytes(B, D, F, L)
        local_ai = local_flops / local_mem_bytes if local_mem_bytes != 0 else float('inf')
        tflops_s = local_flops / avg_time / 1e12

        # Compute aggregated values across GPUs.
        aggregated_flops = local_flops * world_size
        aggregated_mem_bytes = local_mem_bytes * world_size
        aggregated_ai = aggregated_flops / aggregated_mem_bytes if aggregated_mem_bytes != 0 else float('inf')

        # Sum local GFLOPS across GPUs.
        gflops_tensor = torch.tensor([tflops_s], device=torch.device(f"cuda:{rank}"))
        dist.all_reduce(gflops_tensor, op=dist.ReduceOp.SUM)
        aggregated_gflops = gflops_tensor.item()

        result = {
            "batch_size": B,
            "world_size": world_size,
            "D": D,
            "F": F,
            "num_layers": L,
            "avg_time_seconds": avg_time,
            "tflops_s": tflops_s,
            "aggregated_gflops": aggregated_gflops,
            "total_flops": aggregated_flops,                # now aggregated across GPUs
            "estimated_memory_bytes": aggregated_mem_bytes,   # aggregated memory traffic
            "arithmetic_intensity": aggregated_ai             # aggregated arithmetic intensity
        }
        results.append(result)
        if rank == 0:
            print(f"PASSED: B={B}, D={D}, F={F}, layers={L}")
            print(f"  Local TFLOPs: {tflops_s:.2f}  |  Aggregated GFLOPs: {aggregated_gflops:.2f}")
            print(f"  Total FLOPs: {aggregated_flops:.2e}  |  Estimated Memory: {aggregated_mem_bytes:.2e}")
            print(f"  Arithmetic Intensity: {aggregated_ai:.3f}")
            print("-" * 80)
    
    # Only rank 0 writes the results to a JSON file.
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"distributed_test_suite_results_{world_size}gpus.json")
        with open(out_file, "w") as jf:
            json.dump(results, jf, indent=2)
        print(f"\nTest suite complete. Results saved to {out_file}")
    return results

def distributed_main(rank, world_size, test_cases, warmup_iters=2, iters=5):
    # Set up the distributed environment.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank} running on GPU {rank}")

    distributed_run_test_suite(world_size, test_cases, warmup_iters, iters)

    dist.destroy_process_group()

if __name__ == "__main__":
    # Define a default test suite.
    default_test_cases = [
        {"batch_size": 1,   "D": 4096,  "F": 4096,  "num_layers": 5},
        {"batch_size": 128, "D": 4096,  "F": 4096,  "num_layers": 5},
        {"batch_size": 256, "D": 4096,  "F": 4096,  "num_layers": 5},
        {"batch_size": 32,  "D": 40960, "F": 40960, "num_layers": 5},
        {"batch_size": 128, "D": 40960, "F": 40960, "num_layers": 5},
    ]
    # Set world_size to the desired number of GPUs (e.g., 2, 3, or 4).
    world_size = 2  # Change this value to test scaling with more GPUs

    mp.spawn(distributed_main, args=(world_size, default_test_cases), nprocs=world_size, join=True)
