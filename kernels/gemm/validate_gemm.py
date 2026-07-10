"""
validate_gemm.py — compare gemm kernel output against PyTorch reference.

Usage:
    python validate_gemm.py <M> <K> <N> [binary_path]

    binary_path : optional path to the compiled gemm binary. Defaults to
                  the hardcoded BINARY path (kernels/gemm/gemm[.exe]) when
                  omitted, so existing call sites keep working unchanged.

Flow:
  1. Generate A[M,K] and B[K,N] in Python (torch, seed=42).
  2. Write them to temp binary files (raw float32, row-major).
  3. Run the compiled gemm binary with those files as input.
  4. Parse the kernel's printed C[0:8].
  5. Compare against torch.mm(A, B) with FP32 atol=1e-5.

Tolerance: FP32 atol = 1e-5  (per project validation standard)
"""

import sys
import subprocess
import os
import tempfile
import numpy as np
import torch

ATOL_FP32 = 1e-5
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm")
BINARY = _base + ".exe" if sys.platform == "win32" else _base


def run_binary(M: int, K: int, N: int, A: torch.Tensor, B: torch.Tensor,
               binary: str = BINARY):
    """Write A, B to temp files, invoke the kernel, return parsed C values."""
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"Binary '{binary}' not found. "
            f"Compile first:  nvcc -O2 -o {_base} {_base}.cu"
        )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fa, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fb:
        path_a, path_b = fa.name, fb.name

    try:
        A.numpy().astype(np.float32).tofile(path_a)
        B.numpy().astype(np.float32).tofile(path_b)

        result = subprocess.run(
            [binary, str(M), str(K), str(N), path_a, path_b],
            capture_output=True, text=True, check=True
        )
    finally:
        os.unlink(path_a)
        os.unlink(path_b)

    # Expected stdout: "C[0:8] = 1.234567 2.345678 ..."
    line = result.stdout.strip()
    values_str = line.split("=", 1)[1].strip()
    return [float(v) for v in values_str.split()]


def validate(M: int, K: int, N: int, binary: str = BINARY):
    torch.manual_seed(42)
    A = torch.rand(M, K, dtype=torch.float32)
    B = torch.rand(K, N, dtype=torch.float32)
    # FP64 reference gives the correctly-rounded float32 result;
    # pairs with the kernel's double accumulator for a fair comparison.
    C_ref = torch.mm(A.double(), B.double()).float()

    n_print = min(M * N, 8)
    ref_values = C_ref.flatten()[:n_print]

    try:
        kernel_values = run_binary(M, K, N, A, B, binary=binary)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        print(f"  ref C[0:{n_print}] = {ref_values.tolist()}")
        return None

    kernel_tensor = torch.tensor(kernel_values, dtype=torch.float32)

    max_err = (kernel_tensor - ref_values).abs().max().item()
    passed = max_err <= ATOL_FP32

    print(f"M={M} K={K} N={N}")
    print(f"  ref    C[0:{n_print}] = {ref_values.tolist()}")
    print(f"  kernel C[0:{n_print}] = {kernel_tensor.tolist()}")
    print(f"  max absolute error    = {max_err:.2e}  (atol={ATOL_FP32:.0e})")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(f"Usage: python {sys.argv[0]} <M> <K> <N> [binary_path]")
        sys.exit(1)

    M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    binary = sys.argv[4] if len(sys.argv) == 5 else BINARY
    ok = validate(M, K, N, binary=binary)
    sys.exit(0 if ok is None or ok else 1)
