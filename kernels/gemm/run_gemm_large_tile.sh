#!/bin/sh
# gemm_large_tile real-hardware sweep.
# Reuses gemm/test_data/{boundary,non2n,large} -- the same A/B binaries
# already validated against kernels/gemm/gemm.cu on real DCU (see
# docs/dcu_portability_review.md), since gemm_large_tile takes the same
# <M> <K> <N> <A.bin> <B.bin> interface and computes the same math.
#
# Sweeps all four --tile configs per shape, including xlarge (64 KiB dynamic
# shared memory -- the >48KiB opt-in / gfx906-LDS-budget-query path from
# .claude/skills/dcu_hip_porting.md sections 3-4). This is the first time
# that path runs under hipcc on real hardware for this kernel; local nvcc
# runs only proved the CUDA branch.
#
# No file_C output arg (matches run_gemm.sh / run_gemm_stress.sh): only the
# printed "C[0:N] = ..." preview line is compared, against the local
# reference preview generated on the dev box before submission.
for tile in small medium large xlarge; do
  echo "=== gemm_large_tile --tile=$tile boundary (M=1 K=1 N=1) ==="
  ./gemm/gemm_large_tile --tile=$tile 1 1 1 gemm/test_data/boundary/A.bin gemm/test_data/boundary/B.bin

  echo "=== gemm_large_tile --tile=$tile non-2^n (M=100 K=300 N=200) ==="
  ./gemm/gemm_large_tile --tile=$tile 100 300 200 gemm/test_data/non2n/A.bin gemm/test_data/non2n/B.bin

  echo "=== gemm_large_tile --tile=$tile large (M=256 K=256 N=256) ==="
  ./gemm/gemm_large_tile --tile=$tile 256 256 256 gemm/test_data/large/A.bin gemm/test_data/large/B.bin
done
