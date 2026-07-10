#!/bin/sh
# run_rocm_probe.sh -- one-shot real-hardware probe of rocm-smi's actual
# CLI surface on this DTK install. We don't yet know which flags/CSV schema
# this specific rocm-smi version supports (DTK 22.10.1's rocm-smi predates
# a lot of upstream ROCm CLI changes), and the real-hardware portal is a
# one-shot submit/collect-stdout/cleanup batch job with no interactive
# shell -- so instead of guessing flags for the real monitoring wrapper and
# finding out it silently produced empty/garbage output, run every
# candidate invocation once here and print everything to stdout for a human
# to read. No kernel involved, no GPU load generated on purpose -- this is
# just discovering the tool's interface.
echo "=== rocm-smi --version ==="
rocm-smi --version 2>&1
echo
echo "=== rocm-smi (no args, default summary) ==="
rocm-smi 2>&1
echo
echo "=== rocm-smi --showuse --showmemuse ==="
rocm-smi --showuse --showmemuse 2>&1
echo
echo "=== rocm-smi --showuse --showmemuse --csv ==="
rocm-smi --showuse --showmemuse --csv 2>&1
echo
echo "=== rocm-smi -a (full dump) ==="
rocm-smi -a 2>&1
echo
echo "=== rocm-smi --help (first 100 lines) ==="
rocm-smi --help 2>&1 | head -100
