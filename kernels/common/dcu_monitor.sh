#!/bin/sh
# dcu_monitor.sh -- wrap a kernel invocation on the real DCU machine and
# sample `rocm-smi` in the background to confirm the DCU cores were
# actually exercised while it ran (companion to gpu_monitor.sh, which does
# the same thing on the local NVIDIA dev box via nvidia-smi).
#
# Calibrated against a real probe of this specific install (not guessed):
# ROCM-SMI version 1.4.3 (rocm_smi.py), DTK 22.10.1, 4x DCU visible as
# card1..card4 (1-indexed, not 0-indexed). Confirmed real output shape:
#
#   $ rocm-smi --showuse --showmemuse --csv
#   device,DCU use (%),DCU memory use (%)
#   card1,0,0
#   card2,0,0
#   card3,0,0
#   card4,0,0
#
# Unlike nvidia-smi, this rocm-smi build has no built-in continuous-sampling
# flag (no equivalent of -l 1), so this script loops the command itself.
# It is ALSO a Python CLI (rocm_smi.py), not a lightweight C binary -- each
# invocation carries real process-startup + driver-query overhead (the
# probe run of ~6 different rocm-smi invocations took ~8s total), so the
# achievable sampling rate here is much coarser than nvidia-smi's ~1Hz on
# the local box. Don't assume sub-second resolution.
#
# Also note (CONFIRMED, not just theorized -- see the gemm_large_tile
# --bench real-hardware test that validated this script): --gres=dcu:1
# allocates exactly one DCU per job, but rocm-smi still reports all 4
# physical cards regardless of allocation. The job's actual card shows up
# as the only one with non-zero use%; the other three read a steady 0%.
# That's why this script reports a per-card breakdown instead of a single
# aggregate number -- read whichever card is non-zero, not "card1" by
# convention (the allocator could hand out any of the 4 on a different run).
#
# Usage:
#   ./dcu_monitor.sh <cmd> [args...]
# Example:
#   ./dcu_monitor.sh ./gemm/gemm_large_tile --bench --tile=xlarge 1024 1024 1024 10 2000

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <cmd> [args...]" >&2
    exit 1
fi

if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "dcu_monitor.sh: rocm-smi not found on PATH -- is this the DCU machine?" >&2
    "$@"
    exit $?
fi

LOG=$(mktemp)

# Filter with grep '^card' rather than tail -n +2: real output has a
# leading blank line before the header (confirmed empirically -- a first
# version of this script used tail -n +2 assuming the header was line 1,
# which actually left the header itself in the log because the blank line
# was line 1 instead. That header line has 3 comma-separated fields too, so
# awk silently counted it as a fake "device" card with 0% everywhere.
# Filtering by the card prefix sidesteps blank-line/header positioning
# entirely instead of counting fixed line numbers.
( while true; do
    rocm-smi --showuse --showmemuse --csv 2>/dev/null | grep '^card'
  done ) > "$LOG" 2>/dev/null &
SAMPLER_PID=$!

"$@"
STATUS=$?

kill "$SAMPLER_PID" 2>/dev/null
wait "$SAMPLER_PID" 2>/dev/null

echo "--- DCU_UTIL raw samples ---"
cat "$LOG"

awk -F',' '
    NF < 3 { next }
    $1 !~ /^card[0-9]+$/ { next }
    {
        card = $1; use = $2 + 0; mem = $3 + 0
        n[card]++
        sum[card] += use
        summem[card] += mem
        if (use > max[card]) max[card] = use
        if (mem > maxmem[card]) maxmem[card] = mem
    }
    END {
        for (c in n) {
            avg = sum[c] / n[c]
            avgmem = summem[c] / n[c]
            printf "DCU_UTIL_SUMMARY card=%s samples=%d max_use=%d%% avg_use=%.1f%% max_memuse=%d%% avg_memuse=%.1f%%\n", c, n[c], max[c], avg, maxmem[c], avgmem
        }
    }
' "$LOG"

TOTAL_SAMPLES=$(wc -l < "$LOG" | tr -d ' ')
if [ "$TOTAL_SAMPLES" -eq 0 ]; then
    echo "DCU_UTIL_SUMMARY warning: 0 samples captured -- command finished before even one rocm-smi invocation completed (each invocation has real Python-process overhead here). Wrap something with longer sustained duration (bigger --bench iters)." >&2
fi

rm -f "$LOG"
exit "$STATUS"
