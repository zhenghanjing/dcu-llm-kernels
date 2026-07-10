"""
generate_test_data.py — produce a small softmax test case for manual
real-hardware verification (M=N=64), using the exact same generation
pattern as kernels/softmax/validate_softmax.py (seed=42, torch.rand,
torch.nn.functional.softmax reference), so results here are directly
comparable to what that script already validates on the CUDA dev machine.

Usage:
    python generate_test_data.py [M] [N]
    (defaults to 64 64 if not given)

Produces, in this directory:
    X.bin              — M*N float32, row-major
    ref_Y.bin           — M*N float32, row-major (F.softmax(X, dim=1) reference)
    ref_Y_preview.txt   — first 8 reference values of row 0, printed in the
                          same "Y[row=0, 0:8] = ..." format softmax.cu itself
                          prints to stdout, for a quick manual eyeball diff
                          with no tooling required on the deployment machine.

Run softmax on the target machine as:
    ./softmax <M> <N> X.bin Y_out.bin
and compare its printed "Y[row=0, 0:8] = ..." line against ref_Y_preview.txt
(or diff Y_out.bin against ref_Y.bin for a full check instead of just the
first 8 elements).
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

M, N = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (64, 64)

torch.manual_seed(42)
X = torch.rand(M, N, dtype=torch.float32)

Y_ref = F.softmax(X, dim=1)

X.numpy().astype(np.float32).tofile("X.bin")
Y_ref.numpy().astype(np.float32).tofile("ref_Y.bin")

n_preview = min(N, 8)
preview_values = Y_ref[0, :n_preview].tolist()
preview_line = "Y[row=0, 0:%d] =" % n_preview + "".join(f" {v:.6f}" for v in preview_values)

with open("ref_Y_preview.txt", "w") as f:
    f.write(f"M={M} N={N}\n")
    f.write("Reference (torch.nn.functional.softmax(X, dim=1)):\n")
    f.write(preview_line + "\n")
    f.write("\n")
    f.write("Run on target machine:\n")
    f.write(f"  ./softmax {M} {N} X.bin Y_out.bin\n")
    f.write("Compare its stdout line (same \"Y[row=0, 0:N] = ...\" format) against the\n")
    f.write("reference line above. atol=1e-5 per project validation standard\n")
    f.write("(kernel does the standard row-max-subtracted exp/sum in float32,\n")
    f.write("same numerical approach as this reference).\n")

print(f"Wrote X.bin, ref_Y.bin, ref_Y_preview.txt for M={M} N={N}")
print(preview_line)
