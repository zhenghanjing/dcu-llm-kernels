#!/bin/sh
# Softmax stress sweep: boundary / non-2^n / large shapes from
# tests/stress_test.py's TESTS list.
echo "=== softmax boundary (M=1 N=1) ==="
./softmax/softmax 1 1 softmax/test_data/boundary/X.bin softmax/test_data/boundary/Y_out.bin

echo "=== softmax non-2^n (M=100 N=300) ==="
./softmax/softmax 100 300 softmax/test_data/non2n/X.bin softmax/test_data/non2n/Y_out.bin

echo "=== softmax large (M=128 N=1024) ==="
./softmax/softmax 128 1024 softmax/test_data/large/X.bin softmax/test_data/large/Y_out.bin
