import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DeviceMesh, Shard, Replicate
import json

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
