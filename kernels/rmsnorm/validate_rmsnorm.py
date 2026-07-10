"""
validate_rmsnorm.py — compare rmsnorm kernel output against PyTorch reference.

Usage:
    python validate_rmsnorm.py <M> <N>

Flow:
  1. Generate X[M, N] and weight[N] in Python (torch, seed=42).
  2. Write X and weight to temp binary files (raw float32, row-major).
  3. Run the compiled rmsnorm binary: rmsnorm M N file_X file_W file_out
  4. Read the output binary back as a numpy/torch array.
  5. Compare element-wise against:
       ref = x / sqrt(mean(x^2, dim=-1) + eps) * weight
     Cross-checked with torch.nn.functional.rms_norm.

Tolerance: FP32 atol = 1e-5  (per project validation standard)

eps = 1e-6 (LLaMA-style default), matched exactly against the EPS constant
in rmsnorm.cu so the kernel and this reference always compare apples to
apples.
"""

import sys
import subprocess
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

ATOL_FP32 = 1e-5
EPS = 1e-6
_base  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rmsnorm")
BINARY = _base + ".exe" if sys.platform == "win32" else _base


def run_binary(M: int, N: int, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Write X, W to temp files, invoke the kernel, read and return Y."""
    if not os.path.isfile(BINARY):
        raise FileNotFoundError(
            f"Binary '{BINARY}' not found. "
            f"Compile first:  nvcc -O2 -o {_base} {_base}.cu"
        )

    fds = [tempfile.NamedTemporaryFile(suffix=".bin", delete=False) for _ in range(3)]
    paths = [fd.name for fd in fds]
    for fd in fds:
        fd.close()
    path_x, path_w, path_out = paths

    try:
        X.numpy().astype(np.float32).tofile(path_x)
        W.numpy().astype(np.float32).tofile(path_w)

        result = subprocess.run(
            [BINARY, str(M), str(N), path_x, path_w, path_out],
            capture_output=True, text=True, check=True
        )
        print(f"  kernel stdout: {result.stdout.strip()}")

        Y_np = np.fromfile(path_out, dtype=np.float32).reshape(M, N)
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    return torch.from_numpy(Y_np.copy())


def reference_rmsnorm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Explicit RMSNorm formula: x / sqrt(mean(x^2, dim=-1) + eps) * weight."""
    mean_sq = X.pow(2).mean(dim=-1, keepdim=True)
    return X / torch.sqrt(mean_sq + EPS) * W


def validate(M: int, N: int):
    torch.manual_seed(42)
    X = torch.rand(M, N, dtype=torch.float32)
    W = torch.rand(N, dtype=torch.float32)

    ref = reference_rmsnorm(X, W)

    # Cross-check ref against PyTorch's built-in rms_norm (should be ~0 diff)
    ref_builtin = F.rms_norm(X, (N,), weight=W, eps=EPS)
    builtin_gap = (ref - ref_builtin).abs().max().item()
    print(f"  ref vs torch F.rms_norm max diff = {builtin_gap:.2e}  (expected ~0)")

    try:
        kernel_out = run_binary(M, N, X, W)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        n_show = min(N, 8)
        print(f"  ref Y[row=0, 0:{n_show}] = {ref[0, :n_show].tolist()}")
        return None

    diff     = (kernel_out - ref).abs()
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    passed   = max_err <= ATOL_FP32

    n_show = min(N, 8)
    print(f"M={M} N={N}")
    print(f"  ref    Y[row=0, 0:{n_show}] = {ref[0, :n_show].tolist()}")
    print(f"  kernel Y[row=0, 0:{n_show}] = {kernel_out[0, :n_show].tolist()}")
    print(f"  max absolute error           = {max_err:.2e}  (atol={ATOL_FP32:.0e})")
    print(f"  mean absolute error          = {mean_err:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <M> <N>")
        sys.exit(1)

    M, N = int(sys.argv[1]), int(sys.argv[2])
    ok = validate(M, N)
    sys.exit(0 if ok else 1)
