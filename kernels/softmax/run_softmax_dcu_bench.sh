#!/bin/sh
# run_softmax_dcu_bench.sh -- real-hardware performance benchmark for
# softmax.cu, using its built-in --bench mode (hipEventElapsedTime timing,
# default warmup=10/iters=50). Shapes match the ones already used for
# real-hardware correctness stress testing (see docs/dcu_portability_review.md
# "压力测试记录"): (100,300) non-2^n, (128,1024) large. The large shape is
# also wrapped in dcu_monitor.sh to confirm DCU core utilization.
#
# Paths assume cwd == zip's remote_dir (common/ and softmax/ as sibling
# top-level dirs, same convention as run_gemm_dcu_bench.sh).
echo "=== softmax --bench (100,300) ==="
./softmax/softmax --bench 100 300

echo "=== softmax --bench (128,1024), DCU_UTIL wrapped ==="
sh common/dcu_monitor.sh ./softmax/softmax --bench 128 1024
