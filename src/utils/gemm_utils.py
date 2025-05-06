# src/utils/gemm_utils.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Manager
import os

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
