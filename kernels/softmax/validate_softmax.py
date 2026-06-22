"""
validate_softmax.py — compare softmax kernel output against PyTorch reference.

Usage:
    python validate_softmax.py <M> <N>

Flow:
  1. Generate X[M, N] in Python (torch, seed=42).
  2. Write X to a temp binary file (raw float32, row-major).
  3. Run the compiled softmax binary: softmax M N file_in file_out
  4. Read the output binary back as a numpy/torch array.
  5. Compare element-wise against torch.nn.functional.softmax(X, dim=1).

Tolerance: FP32 atol = 1e-5  (per project validation standard)
"""

import sys
import subprocess
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

ATOL_FP32 = 1e-5
_base  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "softmax")
BINARY = _base + ".exe" if sys.platform == "win32" else _base


def run_binary(M: int, N: int, X: torch.Tensor) -> torch.Tensor:
    """Write X to a temp file, invoke the kernel, read and return Y."""
    if not os.path.isfile(BINARY):
        raise FileNotFoundError(
            f"Binary '{BINARY}' not found. "
            f"Compile first:  nvcc -O2 -o {_base} {_base}.cu"
        )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fin, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fout:
        path_in  = fin.name
        path_out = fout.name

    try:
        # Write input
        X.numpy().astype(np.float32).tofile(path_in)

        result = subprocess.run(
            [BINARY, str(M), str(N), path_in, path_out],
            capture_output=True, text=True, check=True
        )
        print(f"  kernel stdout: {result.stdout.strip()}")

        # Read output
        Y_np = np.fromfile(path_out, dtype=np.float32).reshape(M, N)
    finally:
        os.unlink(path_in)
        os.unlink(path_out)

    return torch.from_numpy(Y_np)


def validate(M: int, N: int):
    torch.manual_seed(42)
    X = torch.rand(M, N, dtype=torch.float32)

    ref = F.softmax(X, dim=1)   # shape [M, N]

    try:
        kernel_out = run_binary(M, N, X)
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
