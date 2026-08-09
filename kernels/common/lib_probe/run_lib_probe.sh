#!/bin/sh
# run_lib_probe.sh -- one-shot real-hardware probe of rocBLAS/hipBLAS on this
# DTK 22.10.1 install, same rationale as run_rocm_probe.sh / run_rocprof_probe.sh:
# the real-hardware portal is a one-shot submit/collect-stdout/cleanup batch job
# with no interactive shell, so instead of guessing whether rocBLAS/hipBLAS
# actually compile+link+run here (header presence alone proves nothing) and
# finding out later, run every candidate check once and print everything to
# stdout for a human to read. This answers two questions in one pass:
#   (1) does rocBLAS/hipBLAS really compile, link, and execute on this DTK
#       install -- needed as the DCU same-hardware SOTA comparison baseline;
#   (2) gfx906's real CU count / clock / wavefront size from rocminfo, needed
#       to replace the current guessed FP32 peak (~20 TFLOPS, assumed from an
#       unverified FP64:FP32 = 1:2 ratio) with a derived number.
# No kernel is launched and no GPU load is generated on purpose -- this is
# only discovering the toolchain's interface and the device's static info.
echo "=== hipcc --version ==="
hipcc --version 2>&1
echo
echo "=== rocminfo (full) ==="
rocminfo 2>&1
echo
echo "=== rocm-smi --showhw ==="
rocm-smi --showhw 2>&1
echo
echo "=== rocm-smi --showproductname ==="
rocm-smi --showproductname 2>&1
echo
echo "=== ldconfig -p | grep rocblas/hipblas/miopen ==="
ldconfig -p 2>&1 | grep -iE "rocblas|hipblas|miopen"
echo
echo "=== find headers/libs ==="
find /opt/rocm* \( -iname "*rocblas*" -o -iname "*hipblas*" \) 2>/dev/null
echo
echo "=== minimal hipBLAS link+run test ==="
cat > /tmp/test_hipblas.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
int main() {
    hipblasHandle_t handle;
    hipblasStatus_t st = hipblasCreate(&handle);
    printf("hipblasCreate status=%d\n", (int)st);
    if (st == HIPBLAS_STATUS_SUCCESS) { printf("hipBLAS OK\n"); hipblasDestroy(handle); }
    return 0;
}
EOF
hipcc -O2 /tmp/test_hipblas.cpp -o /tmp/test_hipblas -lhipblas 2>&1
echo "compile exit: $?"
[ -x /tmp/test_hipblas ] && /tmp/test_hipblas 2>&1
echo
echo "=== minimal rocBLAS link+run test ==="
cat > /tmp/test_rocblas.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
int main() {
    rocblas_handle handle;
    rocblas_status st = rocblas_create_handle(&handle);
    printf("rocblas_create_handle status=%d\n", (int)st);
    if (st == rocblas_status_success) { printf("rocBLAS OK\n"); rocblas_destroy_handle(handle); }
    return 0;
}
EOF
hipcc -O2 /tmp/test_rocblas.cpp -o /tmp/test_rocblas -lrocblas 2>&1
echo "compile exit: $?"
[ -x /tmp/test_rocblas ] && /tmp/test_rocblas 2>&1
