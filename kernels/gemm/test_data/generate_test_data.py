"""
generate_test_data.py — produce a small GEMM test case for manual real-hardware
verification (M=K=N=64), using the exact same generation pattern as
kernels/gemm/validate_gemm.py (seed=42, torch.rand, FP64 correctly-rounded
reference), so results here are directly comparable to what that script
already validates on the CUDA dev machine.

Usage:
    python generate_test_data.py [M] [K] [N]
    (defaults to 64 64 64 if not given)

Produces, in this directory:
    A.bin            — M*K float32, row-major
    B.bin             — K*N float32, row-major
    ref_C.bin          — M*N float32, row-major (full FP64-accumulated reference)
    ref_C_preview.txt  — first 8 reference values, printed in the same
                         "C[0:8] = ..." format kernels/gemm/gemm.cu itself
                         prints to stdout, for a quick manual eyeball diff
                         with no tooling required on the deployment machine.

Run gemm on the target machine as:
    ./gemm <M> <K> <N> A.bin B.bin [C_out.bin]
and compare its printed "C[0:8] = ..." line against ref_C_preview.txt (or,
if you passed the optional C_out.bin argument, diff that file against
ref_C.bin for a full check instead of just the first 8 elements).
"""

import sys
import numpy as np
import torch

M, K, N = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) == 4 else (64, 64, 64)

torch.manual_seed(42)
A = torch.rand(M, K, dtype=torch.float32)
B = torch.rand(K, N, dtype=torch.float32)

# FP64 accumulation gives the correctly-rounded float32 result; pairs with
# gemm.cu's own double accumulator for a fair, tight (atol=1e-5) comparison.
C_ref = torch.mm(A.double(), B.double()).float()

A.numpy().astype(np.float32).tofile("A.bin")
B.numpy().astype(np.float32).tofile("B.bin")
C_ref.numpy().astype(np.float32).tofile("ref_C.bin")

n_preview = min(M * N, 8)
preview_values = C_ref.flatten()[:n_preview].tolist()
preview_line = "C[0:%d] =" % n_preview + "".join(f" {v:.6f}" for v in preview_values)

with open("ref_C_preview.txt", "w") as f:
    f.write(f"M={M} K={K} N={N}\n")
    f.write("Reference (FP64-accumulated, torch.mm(A.double(), B.double()).float()):\n")
    f.write(preview_line + "\n")
    f.write("\n")
    f.write("Run on target machine:\n")
    f.write(f"  ./gemm {M} {K} {N} A.bin B.bin\n")
    f.write("Compare its stdout line (same \"C[0:N] = ...\" format) against the\n")
    f.write("reference line above. atol=1e-5 per project validation standard\n")
    f.write("(should match near-exactly, or even bit-for-bit, given gemm.cu's\n")
    f.write("double accumulator).\n")

print(f"Wrote A.bin, B.bin, ref_C.bin, ref_C_preview.txt for M={M} K={K} N={N}")
print(preview_line)
