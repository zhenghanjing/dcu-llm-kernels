#!/bin/sh
# run_gemm_large_tile_bench_monitor.sh -- real-hardware test combining two
# things at once: (1) gemm_large_tile.cu's --bench mode (already validated
# for correctness on this DCU, see docs/dcu_portability_review.md), run
# with enough iterations for sustained load, wrapped in (2) the new
# dcu_monitor.sh resource-usage sampler, to find out for the first time
# whether rocm-smi shows real DCU core utilization during an actual kernel
# run on this specific machine.
#
# Paths below assume cwd == the zip's remote_dir (common/ and gemm/ as
# sibling top-level directories -- same convention as run_gemm_large_tile.sh).
sh common/dcu_monitor.sh ./gemm/gemm_large_tile --bench --tile=xlarge 1024 1024 1024 10 2000
