#!/usr/bin/env python3
"""
tests/benchmark.py — Performance benchmark for GEMM, Softmax, and Flash Attention
kernels vs. their PyTorch reference ops.

Usage:
    python tests/benchmark.py [--skip-compile]

Options:
    --skip-compile   Use existing binaries without recompiling.

Methodology:
  - 3 shapes per operator, reusing the non-2^n / large / extreme tiers from
    tests/stress_test.py (softmax and flash don't have an "extreme" tier there,
    so one is defined here in the same spirit: push size well past "large").
  - Warmup 10 iterations + 50 timed iterations; report the MEDIAN, not the mean,
    so a handful of slow outlier runs (OS scheduling jitter, first-touch page
    faults, etc.) don't skew the number.
  - Our kernels are timed with cudaEventRecord entirely inside the compiled
    binary (see the --bench mode added to each .cu file), so CPU-side process
    spawn / file I/O never enters the measurement.
  - PyTorch ops are timed the equivalent way, with torch.cuda.Event, in-process.

Operator interfaces (bench mode):
  GEMM     : ./gemm       --bench <M> <K> <N>        [warmup] [iters]
  Softmax  : ./softmax    --bench <M> <N>            [warmup] [iters]
  FlashAttn: ./flash_attn --bench <seq_len> <d>       [warmup] [iters]
  Each prints a single line: "BENCH_MS <median_ms>"
"""

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from _compiler import pick_compiler

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Windows consoles default to a legacy code page (e.g. cp1252) that can't
# encode the Chinese column headers below; force UTF-8 stdout/stderr.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EXT = ".exe" if sys.platform == "win32" else ""

BIN = {
    "gemm":    os.path.join(_ROOT, "kernels", "gemm",            f"gemm{_EXT}"),
    "softmax": os.path.join(_ROOT, "kernels", "softmax",         f"softmax{_EXT}"),
    "flash":   os.path.join(_ROOT, "kernels", "flash_attention", f"flash_attn{_EXT}"),
}
SRC = {
    "gemm":    os.path.join(_ROOT, "kernels", "gemm",            "gemm.cu"),
    "softmax": os.path.join(_ROOT, "kernels", "softmax",         "softmax.cu"),
    "flash":   os.path.join(_ROOT, "kernels", "flash_attention", "flash_attn.cu"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP = 10
ITERS  = 50

# ---------------------------------------------------------------------------
# Test matrix: 3 shapes per operator (non-2^n / large / extreme tiers,
# consistent with tests/stress_test.py's categories)
# Shape: GEMM=(M,K,N)  Softmax=(M,N)  Flash=(seq_len,head_dim)
# ---------------------------------------------------------------------------
BENCH_MATRIX = [
    ("non-2^n", "gemm",    (100, 300, 200)),
    ("large",   "gemm",    (1024, 1024, 1024)),
    ("extreme", "gemm",    (4096, 4096, 4096)),

    ("non-2^n", "softmax", (100, 300)),
    ("large",   "softmax", (512, 4096)),
    ("extreme", "softmax", (8192, 8192)),

    ("non-2^n", "flash",   (100, 64)),
    ("large",   "flash",   (512, 64)),
    ("extreme", "flash",   (4096, 128)),
]

OP_LABEL = {"gemm": "GEMM", "softmax": "Softmax", "flash": "FlashAttn"}

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def compile_kernels(skip: bool) -> dict:
    """Compile all three kernels with the detected toolchain (hipcc if present
    on PATH, else nvcc -O2 -arch=sm_120 — see tests/_compiler.py)."""
    ok = {}
    print("Compiling kernels...")

    if not skip:
        compiler_cmd, compiler_label = pick_compiler()
        print(f"  using compiler: {compiler_label}")

    for op in ("gemm", "softmax", "flash"):
        if skip:
            found = os.path.isfile(BIN[op])
            ok[op] = found
            print(f"  [{'OK  ' if found else 'FAIL'}] {op:<8}  "
                  f"{'binary found' if found else 'MISSING'}")
            continue

        out_base = BIN[op][:-4] if BIN[op].endswith(".exe") else BIN[op]
        cmd = [*compiler_cmd, "-o", out_base, SRC[op]]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            dt = time.perf_counter() - t0
            if r.returncode == 0:
                ok[op] = True
                print(f"  [OK  ] {op:<8}  ({dt:.1f}s)")
            else:
                ok[op] = False
                lines = [l for l in r.stderr.splitlines() if l.strip()]
                print(f"  [FAIL] {op:<8}  {lines[-1][:90] if lines else '(no stderr)'}")
        except FileNotFoundError:
            ok[op] = False
            print(f"  [FAIL] {op:<8}  {compiler_label} not found in PATH")
        except subprocess.TimeoutExpired:
            ok[op] = False
            print(f"  [FAIL] {op:<8}  compilation timed out (>180 s)")
    print()
    return ok

# ---------------------------------------------------------------------------
# Our kernel: GPU-timed via the binary's own --bench mode (cudaEvent-based)
# ---------------------------------------------------------------------------
_BENCH_RE = re.compile(r"BENCH_MS\s+([0-9.eE+-]+)")

def bench_ours(op: str, shape: tuple) -> float:
    """Run `<binary> --bench <shape...> WARMUP ITERS`, return median ms."""
    args = [BIN[op], "--bench", *map(str, shape), str(WARMUP), str(ITERS)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=600, check=True)
    m = _BENCH_RE.search(r.stdout)
    if not m:
        raise RuntimeError(f"could not parse BENCH_MS from output: {r.stdout!r}")
    return float(m.group(1))

# ---------------------------------------------------------------------------
# PyTorch reference: timed in-process with torch.cuda.Event
# ---------------------------------------------------------------------------
def _time_cuda(fn, warmup: int, iters: int) -> float:
    """Run fn() warmup+iters times, return median elapsed ms over the timed iters."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def bench_pytorch(op: str, shape: tuple) -> float:
    torch.manual_seed(42)

    if op == "gemm":
        M, K, N = shape
        A = torch.rand(M, K, dtype=torch.float32, device=DEVICE)
        B = torch.rand(K, N, dtype=torch.float32, device=DEVICE)
        return _time_cuda(lambda: torch.matmul(A, B), WARMUP, ITERS)

    if op == "softmax":
        M, N = shape
        X = torch.rand(M, N, dtype=torch.float32, device=DEVICE)
        return _time_cuda(lambda: torch.softmax(X, dim=-1), WARMUP, ITERS)

    if op == "flash":
        seq_len, d = shape
        Q = torch.rand(1, 1, seq_len, d, dtype=torch.float32, device=DEVICE)
        K = torch.rand(1, 1, seq_len, d, dtype=torch.float32, device=DEVICE)
        V = torch.rand(1, 1, seq_len, d, dtype=torch.float32, device=DEVICE)
        return _time_cuda(
            lambda: F.scaled_dot_product_attention(Q, K, V), WARMUP, ITERS
        )

    raise ValueError(op)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def shape_str(shape: tuple) -> str:
    return "(" + ", ".join(map(str, shape)) + ")"


def main():
    parser = argparse.ArgumentParser(description="GEMM/Softmax/FlashAttn benchmark")
    parser.add_argument("--skip-compile", action="store_true",
                         help="Skip recompilation and use existing binaries")
    args = parser.parse_args()

    if DEVICE.type != "cuda":
        print("CUDA not available; benchmarking requires a GPU.")
        sys.exit(1)

    compile_ok = compile_kernels(skip=args.skip_compile)

    rows = []
    for category, op, shape in BENCH_MATRIX:
        label = f"{OP_LABEL[op]} {shape_str(shape)} [{category}]"
        print(f"Benchmarking {label} ...", flush=True)

        if not compile_ok.get(op, False) or not os.path.isfile(BIN[op]):
            rows.append((op, category, shape, None, None, "binary unavailable"))
            continue

        try:
            ours_ms = bench_ours(op, shape)
            torch_ms = bench_pytorch(op, shape)
            rows.append((op, category, shape, ours_ms, torch_ms, ""))
        except Exception as exc:
            rows.append((op, category, shape, None, None, str(exc)[:60]))

    print_table(rows)


def print_table(rows):
    W = dict(op=10, shape=20, cat=9, ours=18, torch=18, ratio=12)
    hdr = " | ".join(f"{h:<{w}}" for h, w in zip(
        ["算子", "shape", "档位", "我们的实现(ms)", "PyTorch(ms)", "相对倍数"],
        W.values(),
    ))
    sep = "-+-".join("-" * w for w in W.values())

    print()
    print(hdr)
    print(sep)
    for op, category, shape, ours_ms, torch_ms, note in rows:
        if ours_ms is None:
            print(" | ".join(f"{v:<{w}}" for v, w in zip(
                [OP_LABEL[op], shape_str(shape), category, "-", "-", f"ERROR: {note}"],
                W.values(),
            )))
            continue
        ratio = ours_ms / torch_ms
        print(" | ".join(f"{v:<{w}}" for v, w in zip(
            [OP_LABEL[op], shape_str(shape), category,
             f"{ours_ms:.4f}", f"{torch_ms:.4f}", f"{ratio:.2f}x"],
            W.values(),
        )))
    print(sep)
    print("相对倍数 = 我们的实现耗时 / PyTorch耗时 (越接近1越好, >1 表示比PyTorch慢)")


if __name__ == "__main__":
    main()
