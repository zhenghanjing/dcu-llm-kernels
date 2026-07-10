"""
generate_test_data.py — produce small flash_attn test cases for manual
real-hardware verification, using the exact same generation pattern as
kernels/flash_attention/validate_flash_attn.py (seed=42, torch.rand,
explicit scaled-dot-product-attention reference cross-checked against
F.scaled_dot_product_attention), so results here are directly comparable
to what that script already validates on the CUDA dev machine.

Usage:
    python generate_test_data.py [seq_len] [head_dim] [out_dir]
    (defaults to 64 64 "." if not given)

Produces, in <out_dir>:
    Q.bin, K.bin, V.bin  — seq_len*head_dim float32 each, row-major
    ref_O.bin             — seq_len*head_dim float32, row-major (reference)
    ref_O_preview.txt      — first 8 reference values of row 0, printed in
                             the same "O[row=0, 0:8] = ..." format
                             flash_attn.cu itself prints to stdout, for a
                             quick manual eyeball diff with no tooling
                             required on the deployment machine.

Run flash_attn on the target machine as:
    ./flash_attn <seq_len> <head_dim> Q.bin K.bin V.bin O_out.bin
and compare its printed "O[row=0, 0:8] = ..." line against ref_O_preview.txt
(or diff O_out.bin against ref_O.bin for a full check).

Two configs are generated for the real-hardware run (see run_flash_attn.sh
one level up), because flash_attn.cu's dynamic shared memory depends only
on BLOCK_SIZE (fixed at 64) and head_dim, not seq_len:

  - seq_len=64  head_dim=64  : smem = 2*64*64*4   = 32768 B (32 KiB)
    Below the 48 KiB opt-in threshold in configure_smem() -- cudaFuncSetAttribute
    is NOT called for this shape. Baseline correctness check only.

  - seq_len=128 head_dim=128 : smem = 2*64*128*4  = 65536 B (64 KiB)
    Same per-block smem footprint as the "extreme" (4096, 128) shape flagged
    HIGH risk in docs/dcu_portability_review.md (flash_attn.cu:197-205, see
    also "真机验证记录" / "拿到真机后的优先级建议" item 3 in that doc):
    exceeds the NVIDIA-specific 48 KiB threshold, forcing configure_smem()
    to call cudaFuncSetAttribute(..., hipFuncAttributeMaxDynamicSharedMemorySize,
    ...) on the HIP path -- and the review flagged that gfx906's LDS budget
    for this size was unknown and could fail to launch at all. This is the
    exact code path that needs real-hardware verification: watch for a
    non-zero exit code or a printed CUDA/HIP error at this step.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F

if len(sys.argv) == 4:
    seq_len, head_dim, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
elif len(sys.argv) == 1:
    seq_len, head_dim, out_dir = 64, 64, "."
else:
    print("Usage: python generate_test_data.py [seq_len] [head_dim] [out_dir]")
    sys.exit(1)

os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(42)
Q = torch.rand(seq_len, head_dim, dtype=torch.float32)
K = torch.rand(seq_len, head_dim, dtype=torch.float32)
V = torch.rand(seq_len, head_dim, dtype=torch.float32)

scale = 1.0 / math.sqrt(head_dim)
scores = (Q @ K.T) * scale
weights = F.softmax(scores, dim=-1)
O_ref = weights @ V

ref_sdpa = F.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
).squeeze(0)
sdpa_gap = (O_ref - ref_sdpa).abs().max().item()

Q.numpy().astype(np.float32).tofile(os.path.join(out_dir, "Q.bin"))
K.numpy().astype(np.float32).tofile(os.path.join(out_dir, "K.bin"))
V.numpy().astype(np.float32).tofile(os.path.join(out_dir, "V.bin"))
O_ref.numpy().astype(np.float32).tofile(os.path.join(out_dir, "ref_O.bin"))

n_preview = min(head_dim, 8)
preview_values = O_ref[0, :n_preview].tolist()
preview_line = "O[row=0, 0:%d] =" % n_preview + "".join(f" {v:.6f}" for v in preview_values)

smem_bytes = 2 * 64 * head_dim * 4  # BLOCK_SIZE=64 fixed in flash_attn.cu
triggers_attr = smem_bytes > 48 * 1024

with open(os.path.join(out_dir, "ref_O_preview.txt"), "w") as f:
    f.write(f"seq_len={seq_len} head_dim={head_dim}\n")
    f.write("Reference (explicit scaled-dot-product-attention, FP32):\n")
    f.write(preview_line + "\n")
    f.write(f"  (cross-check vs torch SDPA max diff = {sdpa_gap:.2e}, expected ~0)\n")
    f.write("\n")
    f.write(f"Dynamic shared memory required: {smem_bytes} bytes ({smem_bytes/1024:.0f} KiB) -- ")
    f.write("exceeds 48 KiB, cudaFuncSetAttribute WILL be called\n" if triggers_attr
            else "below 48 KiB, cudaFuncSetAttribute NOT called\n")
    f.write("\n")
    f.write("Run on target machine:\n")
    f.write(f"  ./flash_attn {seq_len} {head_dim} Q.bin K.bin V.bin O_out.bin\n")
    f.write("Compare its stdout line (same \"O[row=0, 0:N] = ...\" format) against the\n")
    f.write("reference line above. atol=1e-3 per validate_flash_attn.py (tiled online\n")
    f.write("softmax reorders float ops relative to the naive reference).\n")

print(f"Wrote Q.bin, K.bin, V.bin, ref_O.bin, ref_O_preview.txt to '{out_dir}' "
      f"for seq_len={seq_len} head_dim={head_dim}")
print(preview_line)
print(f"smem_bytes={smem_bytes} "
      f"({'TRIGGERS' if triggers_attr else 'does NOT trigger'} cudaFuncSetAttribute)")
