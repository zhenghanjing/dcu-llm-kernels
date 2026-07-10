#!/bin/sh
# run_gemm_ab_bench.sh -- real-hardware A/B data collection for two isolated
# GEMM experiments that have never been run on DCU (wavefront=64) before:
#   1. bank conflict (BN=64 baseline vs BN=32 conflict-free, see gemm.cu:43-60
#      and dcu_perf.md section 3 -- that A/B was only ever done on RTX5090,
#      warp=32)
#   2. accumulator precision (double baseline vs float, see dcu_numerics.md
#      and dcu_perf.md section 7 -- never isolated on ANY platform; the
#      "DCU is ~3x faster than RTX5090 overall" result is only indirect
#      evidence, not a real A/B)
#
# This script does NOT compute an average or stddev -- it prints every raw
# BENCH_MS line from 15 independently-launched processes per (binary, shape)
# so the analysis step afterwards can compute mean/stddev from the untouched
# raw numbers, not from a number this script already summarized.
#
# Paths assume cwd == the zip's remote_dir (common/ and gemm/ as sibling
# top-level dirs, same convention as run_gemm_dcu_bench.sh /
# run_gemm_large_tile_bench_monitor.sh).
#
# Binaries expected (built by the portal's compile-command field):
#   gemm/gemm        -- baseline: BN=64, double accumulator (unchanged default)
#   gemm/gemm_bn32   -- BN=32 (bank-conflict-free), double accumulator
#   gemm/gemm_accf   -- BN=64, float accumulator

BINARIES="gemm gemm_bn32 gemm_accf"
SHAPES="100x300x200 1024x1024x1024 4096x4096x4096"

for bin in $BINARIES; do
    for shape in $SHAPES; do
        M=$(echo "$shape" | cut -d x -f1)
        K=$(echo "$shape" | cut -d x -f2)
        N=$(echo "$shape" | cut -d x -f3)
        echo "=== BENCH BINARY=$bin SHAPE=($M,$K,$N) ==="
        i=1
        while [ "$i" -le 15 ]; do
            echo "--- run $i/15 ---"
            ./gemm/"$bin" --bench "$M" "$K" "$N"
            i=$((i + 1))
        done
    done
done

echo "=== DCU_UTIL check: 4096x4096x4096, each binary, dcu_monitor-wrapped ==="
for bin in $BINARIES; do
    echo "--- BINARY=$bin ---"
    sh common/dcu_monitor.sh ./gemm/"$bin" --bench 4096 4096 4096
done
