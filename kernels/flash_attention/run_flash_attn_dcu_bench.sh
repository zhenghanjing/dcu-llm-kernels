#!/bin/sh
# run_flash_attn_dcu_bench.sh -- real-hardware performance benchmark for
# flash_attn.cu, using its built-in --bench mode (hipEventElapsedTime
# timing, default warmup=10/iters=50). Shapes match dcu_perf.md's grid-size
# analysis table exactly: (100,64) non-2^n, (512,64) large, (4096,128)
# extreme -- the last one has never been run end-to-end on real DCU before
# (only its 64KiB shared-memory boundary was tested separately via the
# smem_risk case at seq_len=128,head_dim=128 -- same smem footprint, but
# untested grid/global-memory scale). Extreme shape wrapped in
# dcu_monitor.sh to confirm DCU core utilization.
#
# Paths assume cwd == zip's remote_dir (common/ and flash_attention/ as
# sibling top-level dirs, same convention as run_gemm_dcu_bench.sh).
echo "=== flash_attn --bench (100,64) ==="
./flash_attention/flash_attn --bench 100 64

echo "=== flash_attn --bench (512,64) ==="
./flash_attention/flash_attn --bench 512 64

echo "=== flash_attn --bench (4096,128), DCU_UTIL wrapped ==="
sh common/dcu_monitor.sh ./flash_attention/flash_attn --bench 4096 128
