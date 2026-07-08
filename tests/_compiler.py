"""
tests/_compiler.py — shared compiler-detection helper for stress_test.py and
benchmark.py.

Prefers hipcc (DCU/DTK toolchain) when present on PATH; otherwise falls back
to the nvcc + -arch=sm_120 path used throughout local CUDA development on
this machine. The nvcc fallback is unchanged from before this file existed,
so existing CUDA workflows keep working exactly as-is when hipcc isn't
installed.
"""

import shutil


def pick_compiler():
    """Return (compiler_cmd, label) for the available toolchain.

    compiler_cmd is a list of argv tokens (compiler + flags, no -o/output/
    source yet); label is a short string for log/error messages.
    """
    if shutil.which("hipcc"):
        # TODO(DCU real-hardware bring-up): hipcc found, but the correct
        # architecture flag (e.g. --offload-arch=gfx906) depends on the
        # installed DTK version and hasn't been confirmed on real hardware
        # yet — see docs/dcu_portability_review.md. Deliberately NOT
        # guessing/hardcoding one here; compiling with no arch flag lets
        # hipcc fall back to its own default target until this is verified.
        return ["hipcc", "-O2"], "hipcc"
    return ["nvcc", "-O2", "-arch=sm_120"], "nvcc"
