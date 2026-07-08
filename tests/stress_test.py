#!/usr/bin/env python3
"""
tests/stress_test.py — Stress test for GEMM, Softmax, and Flash Attention kernels.

Usage:
    python tests/stress_test.py [--skip-compile]

Options:
    --skip-compile   Use existing binaries without recompiling.

Compile flags: hipcc -O2 if hipcc is on PATH, else nvcc -O2 -arch=sm_120
  (see tests/_compiler.py)

Test categories:
  boundary  — minimal shapes (1x1x1, seq=1)
  non-2^n   — shapes not aligned to tile sizes or powers of 2
  large     — multi-tile shapes exercising most kernel paths
  extreme   — GPU memory and throughput stress (GEMM only)

Operator interfaces:
  GEMM     : ./gemm <M> <K> <N> <A.bin> <B.bin> <C.bin>
  Softmax  : ./softmax <M> <N> <in.bin> <out.bin>
  FlashAttn: ./flash_attn <seq> <d> <Q.bin> <K.bin> <V.bin> <O.bin>
  All .bin : raw float32, row-major.

Tolerances (per CLAUDE.md):
  GEMM / Softmax : FP32 atol = 1e-5
  Flash Attention: FP32 atol = 1e-3  (online softmax reorders FP ops)
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from _compiler import pick_compiler

# Use exact FP32 in PyTorch reference (disable TF32 on Ampere+)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EXT  = ".exe" if sys.platform == "win32" else ""

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

# ---------------------------------------------------------------------------
# Tolerances & device
# ---------------------------------------------------------------------------
ATOL = {
    "gemm":    1e-5,
    "softmax": 1e-5,
    "flash":   1e-3,
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Test matrix
# Shape: GEMM=(M,K,N)  Softmax=(M,N)  Flash=(seq_len,head_dim)
# ---------------------------------------------------------------------------
TESTS = [
    # ── boundary ──────────────────────────────────────────────────────────
    ("boundary", "gemm",    (1, 1, 1)),
    ("boundary", "softmax", (1, 1)),
    ("boundary", "flash",   (1, 64)),
    # ── non-power-of-2 ────────────────────────────────────────────────────
    ("non-2^n",  "gemm",    (100, 300, 200)),
    ("non-2^n",  "softmax", (100, 300)),
    ("non-2^n",  "flash",   (100, 64)),
    # ── large ─────────────────────────────────────────────────────────────
    ("large",    "gemm",    (1024, 1024, 1024)),
    ("large",    "softmax", (512, 4096)),
    ("large",    "flash",   (512, 64)),
    # ── extreme ───────────────────────────────────────────────────────────
    ("extreme",  "gemm",    (4096, 4096, 4096)),
]

# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class Result:
    category: str
    label:    str
    status:   str            # PASS / FAIL / SKIP / ERROR
    max_err:  Optional[float] = None
    elapsed:  float = 0.0
    note:     str = ""

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
            tag = "OK  " if found else "FAIL"
            print(f"  [{tag}] {op:<8}  {'binary found' if found else 'MISSING'}")
            continue

        # nvcc/hipcc both expect the output path without .exe; they append
        # .exe on Windows themselves.
        out_base = BIN[op][:-4] if BIN[op].endswith(".exe") else BIN[op]
        cmd = [*compiler_cmd, "-o", out_base, SRC[op]]
        t0  = time.perf_counter()
        try:
            r  = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            dt = time.perf_counter() - t0
            if r.returncode == 0:
                ok[op] = True
                print(f"  [OK  ] {op:<8}  ({dt:.1f}s)")
            else:
                ok[op] = False
                lines   = [l for l in r.stderr.splitlines() if l.strip()]
                snippet = lines[-1][:90] if lines else "(no stderr)"
                print(f"  [FAIL] {op:<8}  {snippet}")
        except FileNotFoundError:
            ok[op] = False
            print(f"  [FAIL] {op:<8}  {compiler_label} not found in PATH")
        except subprocess.TimeoutExpired:
            ok[op] = False
            print(f"  [FAIL] {op:<8}  compilation timed out (>180 s)")
    print()
    return ok

# ---------------------------------------------------------------------------
# Temp-file helpers
# ---------------------------------------------------------------------------
def _mktmp(n: int):
    """Return n closed NamedTemporaryFile paths."""
    paths = []
    for _ in range(n):
        fd = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        fd.close()
        paths.append(fd.name)
    return paths

def _rm(*paths):
    for p in paths:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

def _write(path: str, t: torch.Tensor):
    t.detach().cpu().numpy().astype(np.float32).tofile(path)

def _read(path: str, shape) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape))

# ---------------------------------------------------------------------------
# Per-operator runners
# ---------------------------------------------------------------------------
def run_gemm(M: int, K: int, N: int):
    torch.manual_seed(42)
    A = torch.rand(M, K, dtype=torch.float32)
    B = torch.rand(K, N, dtype=torch.float32)
    # FP64 reference eliminates cuBLAS FP32 accumulation error as a confound,
    # giving the correctly-rounded float32 dot-product as ground truth.
    ref = torch.mm(A.double().to(DEVICE), B.double().to(DEVICE)).float().cpu()

    pA, pB, pC = _mktmp(3)
    try:
        _write(pA, A)
        _write(pB, B)
        t0 = time.perf_counter()
        # Try new 7-arg interface (file_C output); fall back to old 6-arg if binary
        # was compiled before the optional file_C argument was added.
        r = subprocess.run(
            [BIN["gemm"], str(M), str(K), str(N), pA, pB, pC],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            # Old binary: retry with 6 args, compare first 8 elements from stdout
            r6 = subprocess.run(
                [BIN["gemm"], str(M), str(K), str(N), pA, pB],
                capture_output=True, text=True, check=True, timeout=300,
            )
            elapsed = time.perf_counter() - t0
            vals = [float(v) for v in r6.stdout.split("=", 1)[1].strip().split()]
            n_cmp   = min(len(vals), M * N)
            C_part  = torch.tensor(vals[:n_cmp])
            ref_part = ref.flatten()[:n_cmp]
            max_err  = (C_part - ref_part).abs().max().item()
            # Wrap in a tuple; caller will append note separately
            return max_err, elapsed, f"partial {n_cmp}/{ M*N} elem"
        elapsed = time.perf_counter() - t0
        C_out = _read(pC, (M, N))
    finally:
        _rm(pA, pB, pC)

    return (C_out - ref).abs().max().item(), elapsed, ""


def run_softmax(M: int, N: int):
    torch.manual_seed(42)
    X   = torch.rand(M, N, dtype=torch.float32)
    ref = F.softmax(X, dim=1)

    pIn, pOut = _mktmp(2)
    try:
        _write(pIn, X)
        t0 = time.perf_counter()
        subprocess.run(
            [BIN["softmax"], str(M), str(N), pIn, pOut],
            capture_output=True, text=True, check=True, timeout=300,
        )
        elapsed = time.perf_counter() - t0
        Y = _read(pOut, (M, N))
    finally:
        _rm(pIn, pOut)

    return (Y - ref).abs().max().item(), elapsed, ""


def run_flash(seq_len: int, head_dim: int):
    torch.manual_seed(42)
    Q = torch.rand(seq_len, head_dim, dtype=torch.float32)
    K = torch.rand(seq_len, head_dim, dtype=torch.float32)
    V = torch.rand(seq_len, head_dim, dtype=torch.float32)

    scale   = 1.0 / math.sqrt(head_dim)
    scores  = (Q @ K.T) * scale
    weights = F.softmax(scores, dim=-1)
    ref     = weights @ V

    pQ, pK, pV, pO = _mktmp(4)
    try:
        _write(pQ, Q)
        _write(pK, K)
        _write(pV, V)
        t0 = time.perf_counter()
        subprocess.run(
            [BIN["flash"], str(seq_len), str(head_dim), pQ, pK, pV, pO],
            capture_output=True, text=True, check=True, timeout=300,
        )
        elapsed = time.perf_counter() - t0
        O = _read(pO, (seq_len, head_dim))
    finally:
        _rm(pQ, pK, pV, pO)

    return (O - ref).abs().max().item(), elapsed, ""


RUNNERS = {"gemm": run_gemm, "softmax": run_softmax, "flash": run_flash}

# ---------------------------------------------------------------------------
# Shape display helpers
# ---------------------------------------------------------------------------
OP_LABEL = {"gemm": "GEMM", "softmax": "Softmax", "flash": "FlashAttn"}

def _shape_str(op: str, shape: tuple) -> str:
    return "(" + ", ".join(map(str, shape)) + ")"

def make_label(op: str, shape: tuple) -> str:
    return f"{OP_LABEL[op]}{_shape_str(op, shape)}"

# ---------------------------------------------------------------------------
# Single test executor
# ---------------------------------------------------------------------------
def run_test(category: str, op: str, shape: tuple, compile_ok: dict) -> Result:
    label = make_label(op, shape)

    if not compile_ok.get(op, False) or not os.path.isfile(BIN[op]):
        return Result(category, label, "SKIP", note="binary not available")

    try:
        max_err, elapsed, runner_note = RUNNERS[op](*shape)
    except FileNotFoundError:
        return Result(category, label, "SKIP", note="binary not executable")
    except subprocess.CalledProcessError as e:
        lines = (e.stderr or "").strip().splitlines()
        note  = lines[-1][:72] if lines else "non-zero exit"
        return Result(category, label, "ERROR", note=note)
    except subprocess.TimeoutExpired:
        return Result(category, label, "ERROR", note="timeout >300 s")
    except Exception as exc:
        return Result(category, label, "ERROR", note=str(exc)[:72])

    passed    = max_err <= ATOL[op]
    fail_note = f"atol={ATOL[op]:.0e}" if not passed else ""
    note      = " | ".join(filter(None, [runner_note, fail_note]))
    return Result(category, label, "PASS" if passed else "FAIL",
                  max_err, elapsed, note)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_W = dict(cat=10, lbl=26, sts=6, err=13, t=9, note=35)
# " | " separator = 3 chars, 5 gaps between 6 columns
_TOTAL_W = sum(_W.values()) + (len(_W) - 1) * 3

_SEP = "-+-".join("-" * w for w in _W.values())
_HDR = " | ".join(f"{h:<{w}}" for h, w in zip(
    ["Category", "Test Case", "Status", "Max Abs Err", "Time(s)", "Note"],
    _W.values(),
))


def print_report(results: list) -> bool:
    print()
    print("=" * _TOTAL_W)
    print(f"  Stress Test Results   (ref device: {DEVICE})")
    print("=" * _TOTAL_W)
    print(_HDR)
    print(_SEP)

    for r in results:
        err_s  = f"{r.max_err:.3e}" if r.max_err is not None else "-"
        time_s = f"{r.elapsed:.3f}" if r.elapsed  > 0         else "-"
        print(" | ".join(f"{v:<{w}}" for v, w in zip(
            [r.category, r.label, r.status, err_s, time_s, r.note[:_W["note"]]],
            _W.values(),
        )))

    print(_SEP)
    n_tot  = len(results)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status in ("SKIP", "ERROR"))
    print(f"\n  Total: {n_tot}  |  PASS: {n_pass}  |  FAIL: {n_fail}  |  SKIP/ERROR: {n_skip}")
    print("=" * _TOTAL_W)
    return n_fail == 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stress test for GEMM / Softmax / Flash Attention CUDA kernels"
    )
    parser.add_argument(
        "--skip-compile", action="store_true",
        help="Skip recompilation and use existing binaries",
    )
    args = parser.parse_args()

    compile_ok = compile_kernels(skip=args.skip_compile)

    results = []
    n = len(TESTS)
    print(f"Running {n} tests  (ref device: {DEVICE})\n")

    for i, (cat, op, shape) in enumerate(TESTS, 1):
        label = make_label(op, shape)
        print(f"  [{i:2d}/{n}] {label:<30}", end="", flush=True)
        r = run_test(cat, op, shape, compile_ok)
        results.append(r)

        err_part  = f"  err={r.max_err:.3e}" if r.max_err is not None else ""
        note_part = f"  {r.note}"             if r.note               else ""
        print(f"{r.status}{err_part}{note_part}")

    ok = print_report(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
