"""SOTA comparison baseline: torch.mm (cuBLAS) FP32 GEMM on the local RTX 5090.

Mirrors the timing convention of kernels/gemm/gemm.cu's --bench mode
(cudaEvent timing, warmup=10, iters=50, median in ms, BENCH_MS line) so
numbers are directly comparable to the custom kernel's own benchmark output.

TF32 is explicitly disabled -- cuBLAS silently uses TF32 tensor-core matmul
for FP32 inputs by default on Ampere+ GPUs, which would make this an unfair
(faster, lower-precision) comparison against a true FP32 custom kernel.
"""

import statistics

import torch

torch.backends.cuda.matmul.allow_tf32 = False
print(f"torch.backends.cuda.matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")

FP32_PEAK_TFLOPS = 104.8  # RTX 5090 FP32 (non-tensor-core) peak

SHAPES = [
    (100, 300, 200),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
]

WARMUP = 10
ITERS = 50


def bench_shape(M, K, N):
    A = torch.rand(M, K, dtype=torch.float32, device="cuda")
    B = torch.rand(K, N, dtype=torch.float32, device="cuda")

    for _ in range(WARMUP):
        torch.mm(A, B)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(A, B)
        stop.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(stop))

    median_ms = statistics.median(times_ms)
    tflops = 2 * M * N * K / (median_ms / 1000) / 1e12
    utilization_pct = tflops / FP32_PEAK_TFLOPS * 100

    print(f"--- shape M={M} K={K} N={N} ---")
    print(f"BENCH_MS {median_ms:.6f}")
    print(f"TFLOPS {tflops:.6f}")
    print(f"UTILIZATION_PCT {utilization_pct:.4f}")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    for M, K, N in SHAPES:
        bench_shape(M, K, N)


if __name__ == "__main__":
    main()
