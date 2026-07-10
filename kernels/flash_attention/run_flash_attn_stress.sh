#!/bin/sh
# FlashAttn stress sweep: boundary / non-2^n / large shapes from
# tests/stress_test.py's TESTS list. All three stay well under the 48KiB
# cudaFuncSetAttribute threshold (head_dim=64 -> smem=32KiB); that code path
# is separately covered by run_flash_attn.sh's smem_risk case.
echo "=== flash_attn boundary (seq_len=1 head_dim=64) ==="
./flash_attention/flash_attn 1 64 \
  flash_attention/test_data/boundary/Q.bin \
  flash_attention/test_data/boundary/K.bin \
  flash_attention/test_data/boundary/V.bin \
  flash_attention/test_data/boundary/O_out.bin

echo "=== flash_attn non-2^n (seq_len=100 head_dim=64) ==="
./flash_attention/flash_attn 100 64 \
  flash_attention/test_data/non2n/Q.bin \
  flash_attention/test_data/non2n/K.bin \
  flash_attention/test_data/non2n/V.bin \
  flash_attention/test_data/non2n/O_out.bin

echo "=== flash_attn large (seq_len=512 head_dim=64) ==="
./flash_attention/flash_attn 512 64 \
  flash_attention/test_data/large/Q.bin \
  flash_attention/test_data/large/K.bin \
  flash_attention/test_data/large/V.bin \
  flash_attention/test_data/large/O_out.bin
