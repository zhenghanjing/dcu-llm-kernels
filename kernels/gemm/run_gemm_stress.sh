#!/bin/sh
# GEMM stress sweep: boundary / non-2^n / large shapes from tests/stress_test.py's
# TESTS list. Extreme (4096^3) intentionally skipped this round -- too heavy,
# handle separately. No file_C output arg (matches run_gemm.sh): only the
# printed "C[0:N] = ..." preview line is compared, against ref_C_preview.txt.
echo "=== gemm boundary (M=1 K=1 N=1) ==="
./gemm/gemm 1 1 1 gemm/test_data/boundary/A.bin gemm/test_data/boundary/B.bin

echo "=== gemm non-2^n (M=100 K=300 N=200) ==="
./gemm/gemm 100 300 200 gemm/test_data/non2n/A.bin gemm/test_data/non2n/B.bin

echo "=== gemm large (M=256 K=256 N=256) ==="
./gemm/gemm 256 256 256 gemm/test_data/large/A.bin gemm/test_data/large/B.bin
