#!/bin/sh
# run_miopengemm_probe.sh -- one-shot real-hardware probe of MIOpenGEMM/MIOpen
# on this DTK install, same rationale as run_lib_probe.sh: the real-hardware
# portal gives no interactive shell, so instead of guessing whether
# MIOpenGEMM is present/usable here and finding out later, print everything
# to stdout for a human to read. No kernel is launched and no GPU load is
# generated on purpose -- this only discovers headers/libs/symbols/packages.
echo "=== find miopengemm/miopen headers ==="
find /opt/rocm* \( -iname "*miopengemm*" -o -iname "miopen.h" -o -iname "*miogemm*" \) 2>/dev/null
echo
echo "=== exported symbols in libmiopengemm.so (demangled) ==="
nm -D --defined-only /opt/rocm/lib/libmiopengemm.so 2>&1 | c++filt | head -150
echo
echo "=== ldconfig -p | grep miopen ==="
ldconfig -p 2>&1 | grep -i miopen
echo
echo "=== package info (dpkg or rpm, whichever exists) ==="
(dpkg -l 2>/dev/null | grep -i miopen) || (rpm -qa 2>/dev/null | grep -i miopen)
echo
echo "=== does MIOpen's main public header mention gemm? ==="
find /opt/rocm* -iname "miopen.h" -exec grep -il gemm {} \; 2>/dev/null
