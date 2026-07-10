#!/bin/sh
# Basic case: seq_len=64 head_dim=64 -> smem=32KiB, below the 48KiB opt-in
# threshold, cudaFuncSetAttribute is NOT called. Baseline correctness only.
echo "=== flash_attn basic (seq_len=64 head_dim=64, smem=32KiB) ==="
./flash_attention/flash_attn 64 64 \
  flash_attention/test_data/basic/Q.bin \
  flash_attention/test_data/basic/K.bin \
  flash_attention/test_data/basic/V.bin \
  flash_attention/test_data/basic/O_out.bin

# Risk case: seq_len=128 head_dim=128 -> smem=64KiB, same per-block footprint
# as the "extreme" shape flagged HIGH risk in docs/dcu_portability_review.md
# (flash_attn.cu:197-205). Exceeds 48KiB, forces configure_smem() to call
# cudaFuncSetAttribute on the HIP path -- watch this step for a launch
# failure or non-zero exit.
echo "=== flash_attn smem_risk (seq_len=128 head_dim=128, smem=64KiB, triggers cudaFuncSetAttribute) ==="
./flash_attention/flash_attn 128 128 \
  flash_attention/test_data/smem_risk/Q.bin \
  flash_attention/test_data/smem_risk/K.bin \
  flash_attention/test_data/smem_risk/V.bin \
  flash_attention/test_data/smem_risk/O_out.bin
