"""
generate_test_data.py — produce a small RMSNorm test case for manual
real-hardware verification (M=N=64 by default), using the exact same
generation pattern as kernels/rmsnorm/validate_rmsnorm.py (seed=42,
torch.rand for X and weight, explicit RMSNorm reference formula), so
results here are directly comparable to what that script already
validates on the CUDA dev machine.

Usage:
    python generate_test_data.py [M] [N]
    (defaults to 64 64 if not given)

Produces, in this directory:
    X.bin              — M*N float32, row-major
    W.bin               — N float32 (per-column scale)
    ref_Y.bin           — M*N float32, row-major (reference output)
    ref_Y_preview.txt   — first 8 reference values of row 0, printed in the
                          same "Y[row=0, 0:8] = ..." format rmsnorm.cu itself
                          prints to stdout, for a quick manual eyeball diff
                          with no tooling required on the deployment machine.

Run rmsnorm on the target machine as:
    ./rmsnorm <M> <N> X.bin W.bin Y_out.bin
and compare its printed "Y[row=0, 0:8] = ..." line against ref_Y_preview.txt
(or diff Y_out.bin against ref_Y.bin for a full check instead of just the
first 8 elements).
"""

import sys
import numpy as np
import torch

# Must match the EPS constant in kernels/rmsnorm/rmsnorm.cu exactly, so the
# kernel and this reference always compare apples to apples.
EPS = 1e-6

M, N = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (64, 64)

torch.manual_seed(42)
X = torch.rand(M, N, dtype=torch.float32)
W = torch.rand(N, dtype=torch.float32)

mean_sq = X.pow(2).mean(dim=-1, keepdim=True)
Y_ref = X / torch.sqrt(mean_sq + EPS) * W

X.numpy().astype(np.float32).tofile("X.bin")
W.numpy().astype(np.float32).tofile("W.bin")
Y_ref.numpy().astype(np.float32).tofile("ref_Y.bin")

n_preview = min(N, 8)
preview_values = Y_ref[0, :n_preview].tolist()
preview_line = "Y[row=0, 0:%d] =" % n_preview + "".join(f" {v:.6f}" for v in preview_values)

with open("ref_Y_preview.txt", "w") as f:
    f.write(f"M={M} N={N}  eps={EPS}\n")
    f.write("Reference (x / sqrt(mean(x^2, dim=-1) + eps) * weight):\n")
    f.write(preview_line + "\n")
    f.write("\n")
    f.write("Run on target machine:\n")
    f.write(f"  ./rmsnorm {M} {N} X.bin W.bin Y_out.bin\n")
    f.write("Compare its stdout line (same \"Y[row=0, 0:N] = ...\" format) against the\n")
    f.write("reference line above. atol=1e-5 per project validation standard\n")
    f.write("(kernel accumulates the sum-of-squares reduction in double per\n")
    f.write("dcu_numerics.md's precision discipline, so the mismatch should be\n")
    f.write("~1e-7, well under atol).\n")

print(f"Wrote X.bin, W.bin, ref_Y.bin, ref_Y_preview.txt for M={M} N={N}")
print(preview_line)
