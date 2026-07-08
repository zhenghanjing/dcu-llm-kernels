// softmax.cu — row-wise numerically-stable softmax for X[M, N]
// Algorithm: online softmax (subtract row-max before exp)
// Reduction:  shared-memory tree reduction, one CTA per row
//
// Compile (CUDA):  nvcc -O2 -o softmax softmax.cu
// Compile (DCU):   hipcc -O2 -o softmax softmax.cu
// Usage:           ./softmax <M> <N> <file_in> <file_out>
//   file_in  : raw float32 binary, M*N elements, row-major
//   file_out : raw float32 binary, M*N elements, row-major (result)

#include "../common/gpu_compat.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Platform portability
// ---------------------------------------------------------------------------
// __HIPCC__, not __HIP_PLATFORM_HCC__/__HIP_PLATFORM_AMD__: those are
// defined by hip_runtime.h itself, not by the compiler driver, so checking
// them here (before/without including that header) never actually
// triggered on real hardware. See kernels/common/gpu_compat.h.
#ifdef __HIPCC__
  constexpr int WARP_SIZE = 64;
#else
  constexpr int WARP_SIZE = 32;
#endif

// One block per row; must be a power of 2 for the tree reduction.
constexpr int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void softmax_kernel(const float* __restrict__ X,
                                float*       __restrict__ Y,
                                int M, int N)
{
    extern __shared__ float smem[];   // BLOCK_SIZE floats

    int row = blockIdx.x;
    if (row >= M) return;

    const float* x = X + (size_t)row * N;
    float*       y = Y + (size_t)row * N;

    int tid = threadIdx.x;
    int bsz = blockDim.x;            // == BLOCK_SIZE

    // ------------------------------------------------------------------
    // Phase 1: row maximum (for numerical stability)
    // ------------------------------------------------------------------
    float local_max = -FLT_MAX;
    for (int col = tid; col < N; col += bsz)
        local_max = fmaxf(local_max, x[col]);

    smem[tid] = local_max;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: sum of exp(x_i - row_max)
    // ------------------------------------------------------------------
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += bsz)
        local_sum += expf(x[col] - row_max);

    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3: normalise
    // ------------------------------------------------------------------
    for (int col = tid; col < N; col += bsz)
        y[col] = expf(x[col] - row_max) / row_sum;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static float* read_binary(const char* path, size_t n)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open '%s'\n", path); exit(1); }
    float* buf = (float*)malloc(n * sizeof(float));
    size_t got = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (got != n) {
        fprintf(stderr, "Expected %zu floats from '%s', got %zu\n", n, path, got);
        exit(1);
    }
    return buf;
}

static void write_binary(const char* path, const float* buf, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open '%s' for writing\n", path); exit(1); }
    fwrite(buf, sizeof(float), n, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Benchmark mode: GPU-side timing via cudaEvent, warmup + median over N runs.
// Usage: ./softmax --bench <M> <N> [warmup=10] [iters=50]
// Prints: BENCH_MS <median>
// ---------------------------------------------------------------------------
static int cmp_float(const void* a, const void* b)
{
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static int run_benchmark(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s --bench <M> <N> [warmup=10] [iters=50]\n", argv[0]);
        return 1;
    }
    int M      = atoi(argv[2]);
    int N      = atoi(argv[3]);
    int warmup = (argc > 4) ? atoi(argv[4]) : 10;
    int iters  = (argc > 5) ? atoi(argv[5]) : 50;

    size_t sz = (size_t)M * N * sizeof(float);

    float* hX = (float*)malloc(sz);
    srand(42);
    for (size_t i = 0; i < (size_t)M * N; ++i) hX[i] = (float)rand() / RAND_MAX;

    float *dX, *dY;
    check(cudaMalloc(&dX, sz), "malloc X");
    check(cudaMalloc(&dY, sz), "malloc Y");
    check(cudaMemcpy(dX, hX, sz, cudaMemcpyHostToDevice), "copy X");

    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    size_t smem_bytes = BLOCK_SIZE * sizeof(float);

    for (int i = 0; i < warmup; ++i) {
        softmax_kernel<<<grid, block, smem_bytes>>>(dX, dY, M, N);
        check(cudaGetLastError(), "warmup kernel launch");
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(start), "record start");
        softmax_kernel<<<grid, block, smem_bytes>>>(dX, dY, M, N);
        check(cudaGetLastError(), "timed kernel launch");
        check(cudaEventRecord(stop), "record stop");
        check(cudaEventSynchronize(stop), "sync stop");
        check(cudaEventElapsedTime(&times[i], start, stop), "elapsed");
    }

    qsort(times, iters, sizeof(float), cmp_float);
    float median = (iters % 2 == 0)
        ? 0.5f * (times[iters / 2 - 1] + times[iters / 2])
        : times[iters / 2];

    printf("BENCH_MS %.6f\n", median);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dX); cudaFree(dY);
    free(hX); free(times);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0)
        return run_benchmark(argc, argv);

    if (argc != 5) {
        fprintf(stderr,
            "Usage: %s <M> <N> <file_in> <file_out>\n"
            "  file_in : raw float32, M*N elements, row-major\n"
            "  file_out: raw float32, M*N elements, row-major\n",
            argv[0]);
        return 1;
    }
    int   M      = atoi(argv[1]);
    int   N      = atoi(argv[2]);
    const char* file_in  = argv[3];
    const char* file_out = argv[4];

    size_t sz = (size_t)M * N * sizeof(float);

    float* hX = read_binary(file_in, (size_t)M * N);
    float* hY = (float*)calloc(M * N, sizeof(float));

    float *dX, *dY;
    check(cudaMalloc(&dX, sz), "malloc X");
    check(cudaMalloc(&dY, sz), "malloc Y");
    check(cudaMemcpy(dX, hX, sz, cudaMemcpyHostToDevice), "copy X");
    check(cudaMemset(dY, 0, sz), "memset Y");

    // One block per row; shared memory = BLOCK_SIZE floats
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    size_t smem_bytes = BLOCK_SIZE * sizeof(float);
    softmax_kernel<<<grid, block, smem_bytes>>>(dX, dY, M, N);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    check(cudaMemcpy(hY, dY, sz, cudaMemcpyDeviceToHost), "copy Y back");

    write_binary(file_out, hY, (size_t)M * N);

    // Print first 8 elements of row 0 for quick sanity check
    int print_n = (N < 8) ? N : 8;
    printf("Y[row=0, 0:%d] =", print_n);
    for (int i = 0; i < print_n; ++i)
        printf(" %.6f", hY[i]);
    printf("\n");

    cudaFree(dX); cudaFree(dY);
    free(hX); free(hY);
    return 0;
}
