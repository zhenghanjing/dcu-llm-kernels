#!/bin/sh
# run_gemm_dispatch_verify.sh -- real-hardware verification for the new
# auto-dispatch gemm.cu (template<BM,BN,TM,TN,DoubleBuffer> gemm_tiled,
# runtime-picked between the 64-tile and 128-tile variants via
# use_large_tile()/launch_gemm()). Local RTX 5090 data alone isn't enough
# here for two reasons: (1) both dispatch branches need to actually be
# exercised and produce correct output on the real DCU (gfx906,
# wavefront=64), not just on the 5090 dev box; (2) this repo has a
# documented precedent (bank conflict A/B, .claude/skills/dcu_perf.md #8)
# where the 5090 and DCU disagreed on which config was faster, so the
# local ~12.1% win at 4096^3 (the 128-tile dispatch branch) needs its own
# real-hardware confirmation, not an assumption that it carries over.
#
# Three parts, in order:
#   1. Correctness on all 4 existing local test_data tiers -- boundary
#      (M=K=N=1), non2n (100,300,200), large (256,256,256 -- NOTE: despite
#      the directory name this is 256^3, NOT the 1024^3 "large" tier used
#      elsewhere in this repo's perf shape matrix; see the shape printed in
#      each ref_C_preview.txt), extreme (4096,4096,4096). By the dispatch
#      threshold (ceil(M/128)*ceil(N/128)/170 >= 4), boundary/non2n/large
#      all route to the small (64-tile) kernel and extreme is the only one
#      of the four that routes to the large (128-tile) kernel -- so this
#      set exercises both dispatch branches. Compare each printed
#      "C[0:N] = ..." line against the matching
#      test_data/<tier>/ref_C_preview.txt by hand.
#   2. Performance: --bench on (100,300,200)/(1024,1024,1024)/
#      (4096,4096,4096) -- same shapes as run_gemm_dcu_bench.sh -- for the
#      new dispatch binary's real BENCH_MS numbers on DCU.
#   3. rocBLAS reference: same 3 shapes via rocblas_sgemm_bench (already
#      verified compiling+linking on this DTK install, see
#      kernels/common/lib_probe/run_lib_probe.sh), so the new/old kernel
#      numbers and rocBLAS's can go into the same utilization-% table.
#
# Paths assume cwd == zip's remote_dir (common/ and gemm/ as sibling
# top-level dirs -- gemm.cu needs common/gpu_compat.h -- same convention as
# run_gemm_dcu_bench.sh).

echo "=== [1/3] Correctness: boundary (M=K=N=1, expect SMALL tile) ==="
./gemm/gemm 1 1 1 gemm/test_data/boundary/A.bin gemm/test_data/boundary/B.bin

echo "=== [1/3] Correctness: non2n (100,300,200, expect SMALL tile) ==="
./gemm/gemm 100 300 200 gemm/test_data/non2n/A.bin gemm/test_data/non2n/B.bin

echo "=== [1/3] Correctness: large (256,256,256, expect SMALL tile) ==="
./gemm/gemm 256 256 256 gemm/test_data/large/A.bin gemm/test_data/large/B.bin

echo "=== [1/3] Correctness: extreme (4096,4096,4096, expect LARGE tile) ==="
./gemm/gemm 4096 4096 4096 gemm/test_data/extreme/A.bin gemm/test_data/extreme/B.bin

echo "=== [2/3] Perf: gemm --bench (100,300,200) ==="
./gemm/gemm --bench 100 300 200

echo "=== [2/3] Perf: gemm --bench (1024,1024,1024) ==="
./gemm/gemm --bench 1024 1024 1024

echo "=== [2/3] Perf: gemm --bench (4096,4096,4096) ==="
./gemm/gemm --bench 4096 4096 4096

echo "=== [3/3] rocBLAS: rocblas_sgemm_bench (100,300,200) ==="
./gemm/rocblas_sgemm_bench 100 300 200

echo "=== [3/3] rocBLAS: rocblas_sgemm_bench (1024,1024,1024) ==="
./gemm/rocblas_sgemm_bench 1024 1024 1024

echo "=== [3/3] rocBLAS: rocblas_sgemm_bench (4096,4096,4096) ==="
./gemm/rocblas_sgemm_bench 4096 4096 4096
