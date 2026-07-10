"""
validate_gemm_large_tile.py — compare gemm_large_tile kernel output against
PyTorch reference, across the kernel's runtime-selectable tile configs
(small/medium/large/xlarge -- xlarge's dynamic shared memory exceeds 48 KiB).

Usage:
    python validate_gemm_large_tile.py <M> <K> <N> [tile]
    tile: small|medium|large|xlarge (default: sweep all four)

Reuses the validation logic and tolerance from validate_gemm.py:
  1. Generate A[M,K] and B[K,N] in Python (torch, seed=42).
  2. Write them to temp binary files (raw float32, row-major).
  3. Run the compiled gemm_large_tile binary with those files as input,
     once per tile config, via --tile=<name>.
  4. Parse the kernel's printed C[0:8].
  5. Compare against torch.mm(A.double(), B.double()).float() (double
     precision reference, matching the kernel's double accumulator) with
     FP32 atol=1e-5.

Tolerance: FP32 atol = 1e-5  (per project validation standard)
"""

import sys
import subprocess
import os
import tempfile
import numpy as np
import torch

ATOL_FP32 = 1e-5
TILE_CONFIGS = ["small", "medium", "large", "xlarge"]
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_large_tile")
BINARY = _base + ".exe" if sys.platform == "win32" else _base


def run_binary(tile: str, M: int, K: int, N: int, A: torch.Tensor, B: torch.Tensor):
    """Write A, B to temp files, invoke the kernel with --tile=<tile>, return parsed C values."""
    if not os.path.isfile(BINARY):
        raise FileNotFoundError(
            f"Binary '{BINARY}' not found. "
            f"Compile first:  nvcc -O2 -o {_base} {_base}.cu"
        )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fa, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fb:
        path_a, path_b = fa.name, fb.name

    try:
        A.numpy().astype(np.float32).tofile(path_a)
        B.numpy().astype(np.float32).tofile(path_b)

        result = subprocess.run(
            [BINARY, f"--tile={tile}", str(M), str(K), str(N), path_a, path_b],
            capture_output=True, text=True, check=True
        )
    finally:
        os.unlink(path_a)
        os.unlink(path_b)

    # Expected stdout: "C[0:8] = 1.234567 2.345678 ..." (tile-config info
    # goes to stderr in gemm_large_tile.cu so it doesn't interfere here).
    line = result.stdout.strip()
    values_str = line.split("=", 1)[1].strip()
    return [float(v) for v in values_str.split()]


def validate_one(tile: str, M: int, K: int, N: int, A, B, ref_values, n_print):
    try:
        kernel_values = run_binary(tile, M, K, N, A, B)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return None

    kernel_tensor = torch.tensor(kernel_values, dtype=torch.float32)
    max_err = (kernel_tensor - ref_values).abs().max().item()
    passed = max_err <= ATOL_FP32

    print(f"tile={tile:6s} M={M} K={K} N={N}")
    print(f"  ref    C[0:{n_print}] = {ref_values.tolist()}")
    print(f"  kernel C[0:{n_print}] = {kernel_tensor.tolist()}")
    print(f"  max absolute error    = {max_err:.2e}  (atol={ATOL_FP32:.0e})")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def validate(M: int, K: int, N: int, tile: str = None):
    torch.manual_seed(42)
    A = torch.rand(M, K, dtype=torch.float32)
    B = torch.rand(K, N, dtype=torch.float32)
    # FP64 reference gives the correctly-rounded float32 result;
    # pairs with the kernel's double accumulator for a fair comparison.
    C_ref = torch.mm(A.double(), B.double()).float()

    n_print = min(M * N, 8)
    ref_values = C_ref.flatten()[:n_print]

    tiles = [tile] if tile else TILE_CONFIGS
    results = [validate_one(t, M, K, N, A, B, ref_values, n_print) for t in tiles]

    if all(r is None for r in results):
        return None
    return all(r for r in results if r is not None)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(f"Usage: python {sys.argv[0]} <M> <K> <N> [tile]")
        print(f"  tile: {'|'.join(TILE_CONFIGS)} (default: sweep all)")
        sys.exit(1)

    M, K, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    tile_arg = sys.argv[4] if len(sys.argv) == 5 else None
    ok = validate(M, K, N, tile_arg)
    sys.exit(0 if ok is None or ok else 1)
