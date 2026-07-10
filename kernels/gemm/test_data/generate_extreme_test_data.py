"""
generate_extreme_test_data.py — produce the M=K=N=4096 GEMM "extreme" test
case for real-hardware correctness verification, without shipping ~128MiB of
fully-random raw data in the submission zip.

Why not plain torch.rand like generate_test_data.py: at 4096x4096, A and B
are 64MiB each (128MiB total) in raw float32. Fully random data doesn't
compress, so a zip built from it would stay ~128MiB -- the real-hardware
portal has a documented history of failing on large/complex uploads (see
.claude/skills/dcu_hip_porting.md #5). Instead, tile a small (64x64) random
block across the full 4096x4096 matrix: the values are still real,
non-trivial random numbers (not a degenerate all-ones/all-zeros case that
would fail to exercise the kernel meaningfully), but the massive byte-level
repetition compresses down to a small zip.

Usage:
    python generate_extreme_test_data.py
    (fixed at M=K=N=4096; run from within kernels/gemm/test_data/extreme/ so
    A.bin/B.bin/ref_C_preview.txt land in that directory, matching the
    relative-path convention used by generate_test_data.py and the other
    test_data/<shape>/ directories)

Produces, in this directory:
    A.bin              — 4096*4096 float32, row-major (tiled 64x64 pattern)
    B.bin              — 4096*4096 float32, row-major (tiled 64x64 pattern)
    ref_C_preview.txt  — first 8 reference values in gemm.cu's own
                         "C[0:8] = ..." stdout format, for a quick manual
                         diff on the deployment machine (no ref_C.bin full
                         dump -- that would itself be another 64MiB file).

Run gemm on the target machine as:
    ./gemm 4096 4096 4096 A.bin B.bin
and compare its printed "C[0:8] = ..." line against ref_C_preview.txt.
"""

import numpy as np
import torch

torch.manual_seed(42)
N_DIM = 4096
TILE = 64  # 64x64 random tile, repeated (4096/64)x(4096/64) = 64x64 times

a_tile = torch.rand(TILE, TILE, dtype=torch.float32)
b_tile = torch.rand(TILE, TILE, dtype=torch.float32)

A = a_tile.repeat(N_DIM // TILE, N_DIM // TILE)  # -> 4096x4096, heavily repeated pattern
B = b_tile.repeat(N_DIM // TILE, N_DIM // TILE)

# FP64 accumulation gives the correctly-rounded float32 result; pairs with
# gemm.cu's own double accumulator for a fair, tight (atol=1e-5) comparison.
# Reference matmul is done on GPU if available -- a 4096^3 double matmul on
# CPU can be very slow.
device = "cuda" if torch.cuda.is_available() else "cpu"
A_ref = A.to(device)
B_ref = B.to(device)
C_ref = torch.mm(A_ref.double(), B_ref.double()).float().cpu()

A.numpy().astype(np.float32).tofile("A.bin")
B.numpy().astype(np.float32).tofile("B.bin")

n_preview = 8
preview_values = C_ref.flatten()[:n_preview].tolist()
preview_line = "C[0:%d] =" % n_preview + "".join(f" {v:.6f}" for v in preview_values)

with open("ref_C_preview.txt", "w") as f:
    f.write(f"M={N_DIM} K={N_DIM} N={N_DIM} (tiled {TILE}x{TILE} random pattern, not fully random)\n")
    f.write("Reference (FP64-accumulated, torch.mm(A.double(), B.double()).float()):\n")
    f.write(preview_line + "\n")
    f.write("\n")
    f.write("Run on target machine:\n")
    f.write(f"  ./gemm {N_DIM} {N_DIM} {N_DIM} A.bin B.bin\n")
    f.write("Compare its stdout line (same \"C[0:N] = ...\" format) against the\n")
    f.write("reference line above. atol=1e-5 per project validation standard\n")
    f.write("(should match near-exactly, or even bit-for-bit, given gemm.cu's\n")
    f.write("double accumulator).\n")

print(f"Wrote A.bin, B.bin, ref_C_preview.txt for M=K=N={N_DIM} (tiled {TILE}x{TILE} pattern)")
print(preview_line)
