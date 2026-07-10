#!/bin/sh
# run_gemm_dcu_bench.sh -- real-hardware performance benchmark for gemm.cu,
# using its built-in --bench mode (hipEventElapsedTime timing, default
# warmup=10/iters=50). Shapes match dcu_perf.md's RTX5090 baseline exactly
# for direct comparison:
#   (100,300,200)    -- non-2^n / small, grid under-filled on RTX5090
#   (1024,1024,1024) -- large
#   (4096,4096,4096) -- extreme, never run on real DCU before (see
#                       docs/dcu_portability_review.md "仍待验证")
# The extreme shape is also wrapped in dcu_monitor.sh to confirm the DCU
# core is actually loaded for the heaviest case; the other two are left
# unwrapped to keep their BENCH_MS readings free of any sampler overhead.
#
# Paths assume cwd == zip's remote_dir (common/ and gemm/ as sibling
# top-level dirs, same convention as run_gemm_large_tile.sh).
echo "=== gemm --bench (100,300,200) ==="
./gemm/gemm --bench 100 300 200

echo "=== gemm --bench (1024,1024,1024) ==="
./gemm/gemm --bench 1024 1024 1024

echo "=== gemm --bench (4096,4096,4096), DCU_UTIL wrapped ==="
sh common/dcu_monitor.sh ./gemm/gemm --bench 4096 4096 4096
