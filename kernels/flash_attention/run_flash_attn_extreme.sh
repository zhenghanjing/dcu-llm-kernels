#!/bin/sh
./flash_attention/flash_attn 4096 128 \
  flash_attention/test_data/extreme/Q.bin \
  flash_attention/test_data/extreme/K.bin \
  flash_attention/test_data/extreme/V.bin \
  flash_attention/test_data/extreme/O_out.bin
