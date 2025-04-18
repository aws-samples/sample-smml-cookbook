import time
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

# Parameters for GEMM
M, N, K = 1024, 1024, 1024


@nki.jit
def gemm_kernel(A, B):
    # A: [M, K], B: [K, N]
    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    for m in range(M):
        for n in range(N):
            acc = nl.scalar(0.0, dtype=A.dtype)
            for k in range(K):
                acc += A[m, k] * B[k, n]
            C[m, n] = acc

    return C


# Initialize random inputs
a_np = np.random.rand(M, K).astype(np.float32)
b_np = np.random.rand(K, N).astype(np.float32)

# Move to Neuron buffers
a_dev = nki.array(a_np)
b_dev = nki.array(b_np)

# Warm-up
_ = gemm_kernel(a_dev, b_dev)

# Time it
start = time.time()
_ = gemm_kernel(a_dev, b_dev)
end = time.time()

latency = end - start
flops = 2 * M * N * K
tflops = flops / (latency * 1e12)

# Arithmetic intensity
input_bytes = a_np.nbytes + b_np.nbytes
output_bytes = M * N * 4  # float32
ai = flops / (input_bytes + output_bytes)

print(f"Latency: {latency:.6f} sec")
print(f"Estimated TFLOPs: {tflops:.2f}")
print(f"Arithmetic Intensity: {ai:.2f} FLOPs/byte")
