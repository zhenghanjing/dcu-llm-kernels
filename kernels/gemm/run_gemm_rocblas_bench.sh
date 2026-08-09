#!/bin/sh
# run_gemm_rocblas_bench.sh -- real-hardware rocBLAS SOTA baseline for
# rocblas_sgemm_bench.cu (hipEventElapsedTime timing, default warmup=10/
# iters=50, one BENCH_MS line per shape). Shapes match run_gemm_dcu_bench.sh
# exactly so the two BENCH_MS series can be placed side by side in the same
# table:
#   (100,300,200)    -- non-2^n / small
#   (1024,1024,1024) -- large
#   (4096,4096,4096) -- extreme
#
# This bench has no dependency on kernels/common/ (rocblas_sgemm_bench.cu
# includes hip_runtime.h and rocblas.h directly, no gpu_compat.h), so unlike
# run_gemm_dcu_bench.sh it does not need the multi-top-level-dir/path-prefix
# zip layout. Paths here assume the single-top-level-dir convention instead:
# the submit zip's only top-level entry is this gemm/ directory, the portal
# auto-cd's into it, so the compiled binary sits right next to this script
# with no path prefix needed.
echo "=== rocblas_sgemm_bench (100,300,200) ==="
./rocblas_sgemm_bench 100 300 200

echo "=== rocblas_sgemm_bench (1024,1024,1024) ==="
./rocblas_sgemm_bench 1024 1024 1024

echo "=== rocblas_sgemm_bench (4096,4096,4096) ==="
./rocblas_sgemm_bench 4096 4096 4096
