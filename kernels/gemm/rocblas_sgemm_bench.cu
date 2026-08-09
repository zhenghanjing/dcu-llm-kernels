// rocblas_sgemm_bench.cu — rocBLAS SOTA baseline for FP32 GEMM on DCU (gfx906)
// A[M,K] * B[K,N] = C[M,N], row-major, matching gemm.cu's problem definition.
// Compile (DCU): hipcc -O2 -o rocblas_sgemm_bench rocblas_sgemm_bench.cu -lrocblas
// Usage:         ./rocblas_sgemm_bench <M> <K> <N> [warmup=10] [iters=50]
// Prints:        BENCH_MS <median>
//
// Timing/median convention (hipEventRecord/hipEventElapsedTime, warmup+iters,
// qsort+median) is deliberately identical to gemm.cu's run_benchmark() so the
// BENCH_MS numbers from both binaries can be placed directly into the same
// comparison table.
//
// rocBLAS is column-major; our A/B/C buffers are row-major. Standard trick:
// row-major C[M,N] = A[M,K]*B[K,N] is the same memory layout as column-major
// C^T[N,M] = B^T[N,K]*A^T[K,M], so we ask rocBLAS for a column-major
// (N,M) = (N,K)*(K,M) product with A and B swapped and M/N swapped — no
// transpose flags needed, both operands stay rocblas_operation_none.
//
// #include path verified compiling on real hardware (dtk-22.10.1, gfx906) —
// see kernels/common/lib_probe/run_lib_probe.sh.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check_hip(hipError_t err, const char* msg)
{
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s: %s\n", msg, hipGetErrorString(err));
        exit(1);
    }
}

static void check_rocblas(rocblas_status st, const char* msg)
{
    if (st != rocblas_status_success) {
        fprintf(stderr, "rocBLAS error at %s: status=%d\n", msg, (int)st);
        exit(1);
    }
}

static int cmp_float(const void* a, const void* b)
{
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <M> <K> <N> [warmup=10] [iters=50]\n", argv[0]);
        return 1;
    }
    int M      = atoi(argv[1]);
    int K      = atoi(argv[2]);
    int N      = atoi(argv[3]);
    int warmup = (argc > 4) ? atoi(argv[4]) : 10;
    int iters  = (argc > 5) ? atoi(argv[5]) : 50;

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float* hA = (float*)malloc(szA);
    float* hB = (float*)malloc(szB);
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; ++i) hA[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; ++i) hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    check_hip(hipMalloc(&dA, szA), "malloc A");
    check_hip(hipMalloc(&dB, szB), "malloc B");
    check_hip(hipMalloc(&dC, szC), "malloc C");
    check_hip(hipMemcpy(dA, hA, szA, hipMemcpyHostToDevice), "copy A");
    check_hip(hipMemcpy(dB, hB, szB, hipMemcpyHostToDevice), "copy B");

    rocblas_handle handle;
    check_rocblas(rocblas_create_handle(&handle), "create handle");
    check_rocblas(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host), "set pointer mode");

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int i = 0; i < warmup; ++i) {
        check_rocblas(
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                          N, M, K, &alpha, dB, N, dA, K, &beta, dC, N),
            "warmup sgemm");
    }
    check_hip(hipDeviceSynchronize(), "warmup sync");

    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "event create start");
    check_hip(hipEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check_hip(hipEventRecord(start), "record start");
        check_rocblas(
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                          N, M, K, &alpha, dB, N, dA, K, &beta, dC, N),
            "timed sgemm");
        check_hip(hipEventRecord(stop), "record stop");
        check_hip(hipEventSynchronize(stop), "sync stop");
        check_hip(hipEventElapsedTime(&times[i], start, stop), "elapsed");
    }

    qsort(times, iters, sizeof(float), cmp_float);
    float median = (iters % 2 == 0)
        ? 0.5f * (times[iters / 2 - 1] + times[iters / 2])
        : times[iters / 2];

    printf("BENCH_MS %.6f\n", median);

    hipEventDestroy(start); hipEventDestroy(stop);
    rocblas_destroy_handle(handle);
    hipFree(dA); hipFree(dB); hipFree(dC);
    free(hA); free(hB); free(times);
    return 0;
}
