#!/bin/sh
# gpu_monitor.sh -- wrap a kernel invocation and sample nvidia-smi in the
# background to confirm the GPU's SM cores / memory controller were actually
# exercised while it ran (not just "process exited 0").
#
# Usage:
#   ./gpu_monitor.sh <cmd> [args...]
# Example:
#   ./gpu_monitor.sh ./gemm_large_tile --bench --tile=xlarge 1024 1024 1024
#
# The wrapped command's own stdout/stderr pass through unchanged. After it
# exits, a summary line is appended:
#   GPU_UTIL_SUMMARY samples=<N> max_sm_util=<X>% avg_sm_util=<Y>% max_mem_used=<Z>MiB
#
# IMPORTANT CAVEAT: nvidia-smi samples at ~1 Hz here (-l 1). A single
# correctness run of a small GEMM/Softmax/FlashAttn shape finishes in well
# under a millisecond -- far shorter than the sampling interval -- so
# max_sm_util=0% from a quick single-shot run does NOT mean the kernel didn't
# use the GPU; it means the sampler simply never landed a sample during the
# (very short) window the kernel was actually running. For a meaningful
# reading, wrap something with sustained duration: a --bench loop (see
# gemm_large_tile.cu's --bench mode: warmup+iters, default 10+50), a large
# shape, or a shell loop that repeats a correctness run many times back to
# back.
#
# Requires: nvidia-smi on PATH (NVIDIA/CUDA dev box only -- this script is
# not meant to run on the DCU side; see dcu_monitor.sh for that, once it
# exists).

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <cmd> [args...]" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu_monitor.sh: nvidia-smi not found on PATH -- is this the NVIDIA dev box?" >&2
    "$@"
    exit $?
fi

LOG=$(mktemp)

nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
    --format=csv,noheader,nounits -l 1 > "$LOG" 2>/dev/null &
SMI_PID=$!

"$@"
STATUS=$?

# Give the sampler one more tick before killing it, then stop it.
sleep 1
kill "$SMI_PID" 2>/dev/null
wait "$SMI_PID" 2>/dev/null

MAX_SM=0
SUM_SM=0
N=0
MAX_MEM=0

while IFS=',' read -r sm mem memused; do
    # nvidia-smi.exe on Windows emits CRLF line endings; strip both spaces
    # and any stray \r explicitly rather than relying on a shell's command
    # substitution to normalize it away (that's an MSYS/Git-Bash-specific
    # side effect on this dev box, not something to depend on).
    sm=$(printf '%s' "$sm" | tr -d ' \r')
    memused=$(printf '%s' "$memused" | tr -d ' \r')
    case "$sm" in ''|*[!0-9]*) continue ;; esac
    N=$((N + 1))
    SUM_SM=$((SUM_SM + sm))
    if [ "$sm" -gt "$MAX_SM" ]; then MAX_SM=$sm; fi
    case "$memused" in ''|*[!0-9]*) memused=0 ;; esac
    if [ "$memused" -gt "$MAX_MEM" ]; then MAX_MEM=$memused; fi
done < "$LOG"

if [ "$N" -gt 0 ]; then
    AVG_SM=$((SUM_SM / N))
else
    AVG_SM=0
fi

echo "GPU_UTIL_SUMMARY samples=$N max_sm_util=${MAX_SM}% avg_sm_util=${AVG_SM}% max_mem_used=${MAX_MEM}MiB"

# Two distinct ambiguous cases, both worth flagging -- not just N==0.
# nvidia-smi's first "-l 1" sample fires immediately at launch, and this
# script sleeps 1s after the wrapped command exits before killing the
# sampler, so N==0 is actually rare in practice: a quick single-shot
# correctness run typically still lands 1-2 samples, all reading 0%
# (confirmed empirically: a sub-millisecond gemm_large_tile correctness
# call produced samples=2, max_sm_util=0%, and would have printed no
# warning under the old N==0-only check). max_sm_util==0% with N>0 is the
# common, uninformative case that actually needs flagging, not just N==0.
if [ "$N" -eq 0 ]; then
    echo "GPU_UTIL_SUMMARY warning: 0 samples captured -- command finished" \
         "faster than the sampler could take even one reading; this says" \
         "nothing about whether the GPU was used. Wrap a --bench loop or" \
         "larger shape instead." >&2
elif [ "$MAX_SM" -eq 0 ]; then
    echo "GPU_UTIL_SUMMARY warning: $N sample(s) captured but max_sm_util=0%" \
         "-- most likely the command ran and finished in between 1Hz" \
         "sampling ticks (typical for a single quick correctness run), not" \
         "proof the GPU was idle. Wrap a --bench loop or larger shape for a" \
         "meaningful reading." >&2
fi

rm -f "$LOG"
exit "$STATUS"
