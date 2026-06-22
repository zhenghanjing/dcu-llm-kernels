"""
validate_flash_attn.py — compare flash_attn kernel against PyTorch reference.

Usage:
    python validate_flash_attn.py <seq_len> <head_dim>

Flow:
  1. Generate Q, K, V with torch.rand (seed=42), shape [seq_len, head_dim].
  2. Write each to a temp binary file (raw float32, row-major).
  3. Run:  flash_attn seq_len head_dim Q.bin K.bin V.bin O.bin
  4. Read O.bin back as a torch tensor.
  5. Compare element-wise against:
       scores  = (Q @ K.T) / sqrt(head_dim)   # [seq, seq]
       weights = softmax(scores, dim=-1)        # [seq, seq]
       ref     = weights @ V                    # [seq, d]
     Cross-checked with F.scaled_dot_product_attention.
  6. Report max/mean absolute error and PASS/FAIL.

Tolerance: FP32 atol = 1e-3
  Flash Attention tiles the computation and applies online softmax, which
  reorders floating-point operations relative to the naive reference.
  For typical inputs (seq <= 512, d <= 128) the error is < 1e-4; we set
  atol=1e-3 to account for pathological cases with large seq_len.
"""

import sys
import subprocess
import os
import tempfile
import math
import numpy as np
import torch
import torch.nn.functional as F

ATOL_FP32 = 1e-3
_base  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash_attn")
BINARY = _base + ".exe" if sys.platform == "win32" else _base


def run_binary(seq_len: int, head_dim: int,
               Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Write Q/K/V to temp files, invoke kernel, read and return O."""
    if not os.path.isfile(BINARY):
        raise FileNotFoundError(
            f"Binary '{BINARY}' not found. "
            f"Compile first:  nvcc -O2 -o {_base} {_base}.cu"
        )

    # Create four temp files: Q, K, V inputs and O output
    fds = [tempfile.NamedTemporaryFile(suffix=".bin", delete=False) for _ in range(4)]
    paths = [fd.name for fd in fds]
    for fd in fds:
        fd.close()
    pQ, pK, pV, pO = paths

    try:
        Q.numpy().astype(np.float32).tofile(pQ)
        K.numpy().astype(np.float32).tofile(pK)
        V.numpy().astype(np.float32).tofile(pV)

        result = subprocess.run(
            [BINARY, str(seq_len), str(head_dim), pQ, pK, pV, pO],
            capture_output=True, text=True, check=True
        )
        print(f"  kernel stdout: {result.stdout.strip()}")

        O_np = np.fromfile(pO, dtype=np.float32).reshape(seq_len, head_dim)
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass

    return torch.from_numpy(O_np.copy())


def reference_attention(Q: torch.Tensor, K: torch.Tensor,
                        V: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Explicit scaled dot-product attention (numerically stable softmax)."""
    scale   = 1.0 / math.sqrt(head_dim)
    scores  = (Q @ K.T) * scale          # [seq, seq]
    weights = F.softmax(scores, dim=-1)  # [seq, seq]
    return weights @ V                   # [seq, d]


def validate(seq_len: int, head_dim: int):
    torch.manual_seed(42)
    Q = torch.rand(seq_len, head_dim, dtype=torch.float32)
    K = torch.rand(seq_len, head_dim, dtype=torch.float32)
    V = torch.rand(seq_len, head_dim, dtype=torch.float32)

    ref = reference_attention(Q, K, V, head_dim)

    # Cross-check ref against PyTorch's fused SDPA (should be ~0 diff)
    ref_sdpa = F.scaled_dot_product_attention(
        Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
    ).squeeze(0)
    sdpa_gap = (ref - ref_sdpa).abs().max().item()
    print(f"  ref vs torch SDPA max diff = {sdpa_gap:.2e}  (expected ~0)")

    try:
        O_kernel = run_binary(seq_len, head_dim, Q, K, V)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        n_show = min(head_dim, 8)
        print(f"  ref O[row=0, 0:{n_show}] = {ref[0, :n_show].tolist()}")
        return None

    diff     = (O_kernel - ref).abs()
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    passed   = max_err <= ATOL_FP32

    n_show = min(head_dim, 8)
    print(f"seq_len={seq_len}  head_dim={head_dim}")
    print(f"  ref    O[row=0, 0:{n_show}] = {ref[0, :n_show].tolist()}")
    print(f"  kernel O[row=0, 0:{n_show}] = {O_kernel[0, :n_show].tolist()}")
    print(f"  max absolute error           = {max_err:.2e}  (atol={ATOL_FP32:.0e})")
    print(f"  mean absolute error          = {mean_err:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <seq_len> <head_dim>")
        sys.exit(1)

    seq_len  = int(sys.argv[1])
    head_dim = int(sys.argv[2])
    ok = validate(seq_len, head_dim)
    sys.exit(0 if ok else 1)
