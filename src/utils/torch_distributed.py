import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import time
import os
import json

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


if __name__ == "__main__":
    test_cases = [{"batch_size": 256, "num_batches": 20}]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    mp.spawn(distributed_main, args=(world_size, test_cases), nprocs=world_size, join=True)
