import torch
import time
import json
import os

def multi_layer_chain(batch_size, D, F, num_layers=2, warmup_iters=2, iters=5):
    """
    Performs a chain of matrix multiplications in FP16 on GPU:
      1) X: [B, D] x W0: [D, F] -> [B, F]
      2) for layer i in range(1, num_layers): 
           output_{i-1}: [B, F] x Wi: [F, F] -> [B, F]
    Followed by a ReLU activation after each layer. 
    Returns average time in seconds.

    Raises RuntimeError if shapes cannot be multiplied.
    """
    device = 'cuda'

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
    torch.cuda.synchronize()

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
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time

def calculate_chain_flops(batch_size, D, F, num_layers=2):
    """
    Theoretical total FLOPs for this multi-layer chain:
      1st layer: [B, D] x [D, F] => 2 * B * D * F
      next (num_layers-1) layers: each [B, F] x [F, F] => 2 * B * F * F
    """
    first_layer = 2.0 * batch_size * D * F
    other_layers = (num_layers - 1) * (2.0 * batch_size * F * F)
    return first_layer + other_layers

def calculate_memory_bytes(batch_size, D, F, num_layers=2):
    """
    A rough (naive) estimate of the memory traffic in bytes, ignoring 
    caching details. If needed, adapt to reflect extra reads/writes or 
    CPU round-trips. For now:
      - input [B, D], read once
      - w0 [D, F], read once
      - output of layer 0 => [B, F], written
      - each subsequent layer reads [B, F] and [F, F] 
        => writes [B, F]
    All are FP16 => 2 bytes/element.
    """
    bytes_per_elem = 2.0
    # 1st layer: read input ([B,D]) + read w0 ([D,F]) + write out ([B,F])
    mem_first_layer = (batch_size * D + D * F + batch_size * F) * bytes_per_elem
    # Other layers: each reads [B,F] + [F,F], writes [B,F]
    mem_other_layers = (num_layers - 1) * ((batch_size * F + F * F + batch_size * F) * bytes_per_elem)
    return mem_first_layer + mem_other_layers

def run_test_suite(test_cases, warmup_iters=2, iters=5, output_dir="results"):
    """
    Runs the provided test suite of configurations.

    Parameters:
      - test_cases: List of dictionaries, each containing keys:
          "batch_size", "D", "F", and "num_layers"
      - warmup_iters: Number of warmup iterations before timing
      - iters: Number of timed iterations
      - output_dir: Directory where results JSON will be saved

    Returns:
      - results: List of result dictionaries for each successful test case.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for cfg in test_cases:
        B = cfg["batch_size"]
        D = cfg["D"]
        F = cfg["F"]
        L = cfg["num_layers"]

        # Attempt to run the chain
        try:
            avg_time = multi_layer_chain(B, D, F, L, warmup_iters=warmup_iters, iters=iters)
        except RuntimeError as e:
            # Catch shape mismatch or any other matmul error
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                print(f"Skipping incompatible shape: B={B}, D={D}, F={F}, layers={L}. Error: {e}")
                continue
            else:
                raise

        # Compute theoretical FLOPs, memory traffic, and arithmetic intensity (AI)
        flops = calculate_chain_flops(B, D, F, L)
        mem_bytes = calculate_memory_bytes(B, D, F, L)
        ai = flops / mem_bytes if mem_bytes != 0 else float('inf')

        # Record results for this configuration
        result = {
            "batch_size": B,
            "D": D,
            "F": F,
            "num_layers": L,
            "avg_time_seconds": avg_time,
            "total_flops": flops,
            "estimated_memory_bytes": mem_bytes,
            "arithmetic_intensity": ai
        }
        results.append(result)

        print(
            f"PASSED: B={B}, D={D}, F={F}, layers={L}\n"
            f"  Time: {avg_time * 1e3:.3f} ms  |  "
            f"FLOPs={flops:.2e}  |  Bytes={mem_bytes:.2e}  |  AI={ai:.3f}"
        )
        print("-" * 80)
    
    # Save results to JSON file
    out_file = os.path.join(output_dir, "test_suite_results.json")
    with open(out_file, "w") as jf:
        json.dump(results, jf, indent=2)
    
    print(f"\nTest suite complete. Results saved to {out_file}")
    return results

if __name__ == "__main__":
    # If run as a script, you can provide a default test suite here.
    default_test_cases = [
        # Example configurations for testing:
        {"batch_size": 1,   "D": 4096,  "F": 4096,  "num_layers": 5},
        {"batch_size": 128, "D": 4096,  "F": 4096,  "num_layers": 5},
        {"batch_size": 256, "D": 4096,  "F": 4096,  "num_layers": 5},
        {"batch_size": 32,  "D": 40960, "F": 40960, "num_layers": 5},
        {"batch_size": 128, "D": 40960, "F": 40960, "num_layers": 5},
    ]
    run_test_suite(default_test_cases)
