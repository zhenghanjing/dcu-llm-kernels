// rmsnorm.cu — row-wise RMSNorm for X[M, N] with per-column weight W[N]
// Formula: y = x / sqrt(mean(x^2, dim=-1) + eps) * weight
// Reduction:  shared-memory tree reduction, one CTA per row (same layout as
//             kernels/softmax/softmax.cu)
//
// Compile (CUDA):  nvcc -O2 -o rmsnorm rmsnorm.cu
// Compile (DCU):   hipcc -O2 -o rmsnorm rmsnorm.cu
// Usage:           ./rmsnorm <M> <N> <file_X> <file_W> <file_out>
//   file_X   : raw float32 binary, M*N elements, row-major
//   file_W   : raw float32 binary, N elements (per-column scale)
//   file_out : raw float32 binary, M*N elements, row-major (result)

#include "../common/gpu_compat.h"

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

// LLaMA-style default; matched independently in validate_rmsnorm.py so the
// kernel and the PyTorch reference always compare apples to apples.
constexpr float EPS = 1e-6f;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const float* __restrict__ X,
                                const float* __restrict__ W,
                                float*       __restrict__ Y,
                                int M, int N, float eps)
{
    // Reduction accumulator is double, not float (dcu_numerics.md precision
    // discipline: any "sum many values, then output once" reduction should
    // default to a double accumulator rather than wait for a precision bug
    // to show up). Shared memory therefore holds BLOCK_SIZE doubles, not
    // floats, unlike softmax.cu's max/sum-of-exp phases.
    extern __shared__ double smem[];

    int row = blockIdx.x;
    if (row >= M) return;

    const float* x = X + (size_t)row * N;
    float*       y = Y + (size_t)row * N;

    int tid = threadIdx.x;
    int bsz = blockDim.x;            // == BLOCK_SIZE

    // ------------------------------------------------------------------
    // Phase 1: sum of squares, accumulated in double
    // ------------------------------------------------------------------
    double local_sum = 0.0;
    for (int col = tid; col < N; col += bsz) {
        double v = (double)x[col];
        local_sum += v * v;
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    double mean_sq = smem[0] / (double)N;
    double rms     = sqrt(mean_sq + (double)eps);
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: normalise and scale
    // ------------------------------------------------------------------
    for (int col = tid; col < N; col += bsz) {
        float normalized = (float)((double)x[col] / rms);
        y[col] = normalized * W[col];
    }
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
// Usage: ./rmsnorm --bench <M> <N> [warmup=10] [iters=50]
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

    size_t szX = (size_t)M * N * sizeof(float);
    size_t szW = (size_t)N * sizeof(float);

    float* hX = (float*)malloc(szX);
    float* hW = (float*)malloc(szW);
    srand(42);
    for (size_t i = 0; i < (size_t)M * N; ++i) hX[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)N; ++i)      hW[i] = (float)rand() / RAND_MAX;

    float *dX, *dW, *dY;
    check(cudaMalloc(&dX, szX), "malloc X");
    check(cudaMalloc(&dW, szW), "malloc W");
    check(cudaMalloc(&dY, szX), "malloc Y");
    check(cudaMemcpy(dX, hX, szX, cudaMemcpyHostToDevice), "copy X");
    check(cudaMemcpy(dW, hW, szW, cudaMemcpyHostToDevice), "copy W");

    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    size_t smem_bytes = BLOCK_SIZE * sizeof(double);

    for (int i = 0; i < warmup; ++i) {
        rmsnorm_kernel<<<grid, block, smem_bytes>>>(dX, dW, dY, M, N, EPS);
        check(cudaGetLastError(), "warmup kernel launch");
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(start), "record start");
        rmsnorm_kernel<<<grid, block, smem_bytes>>>(dX, dW, dY, M, N, EPS);
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
    cudaFree(dX); cudaFree(dW); cudaFree(dY);
    free(hX); free(hW); free(times);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0)
        return run_benchmark(argc, argv);

    if (argc != 6) {
        fprintf(stderr,
            "Usage: %s <M> <N> <file_X> <file_W> <file_out>\n"
            "  file_X   : raw float32, M*N elements, row-major\n"
            "  file_W   : raw float32, N elements (per-column scale)\n"
            "  file_out : raw float32, M*N elements, row-major\n",
            argv[0]);
        return 1;
    }
    int   M        = atoi(argv[1]);
    int   N        = atoi(argv[2]);
    const char* file_X   = argv[3];
    const char* file_W   = argv[4];
    const char* file_out = argv[5];

    size_t szX = (size_t)M * N * sizeof(float);
    size_t szW = (size_t)N * sizeof(float);

    float* hX = read_binary(file_X, (size_t)M * N);
    float* hW = read_binary(file_W, (size_t)N);
    float* hY = (float*)calloc(M * N, sizeof(float));

    float *dX, *dW, *dY;
    check(cudaMalloc(&dX, szX), "malloc X");
    check(cudaMalloc(&dW, szW), "malloc W");
    check(cudaMalloc(&dY, szX), "malloc Y");
    check(cudaMemcpy(dX, hX, szX, cudaMemcpyHostToDevice), "copy X");
    check(cudaMemcpy(dW, hW, szW, cudaMemcpyHostToDevice), "copy W");
    check(cudaMemset(dY, 0, szX), "memset Y");

    // One block per row; shared memory = BLOCK_SIZE doubles
    dim3 block(BLOCK_SIZE);
    dim3 grid(M);
    size_t smem_bytes = BLOCK_SIZE * sizeof(double);
    rmsnorm_kernel<<<grid, block, smem_bytes>>>(dX, dW, dY, M, N, EPS);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    check(cudaMemcpy(hY, dY, szX, cudaMemcpyDeviceToHost), "copy Y back");

    write_binary(file_out, hY, (size_t)M * N);

    // Print first 8 elements of row 0 for quick sanity check
    int print_n = (N < 8) ? N : 8;
    printf("Y[row=0, 0:%d] =", print_n);
    for (int i = 0; i < print_n; ++i)
        printf(" %.6f", hY[i]);
    printf("\n");

    cudaFree(dX); cudaFree(dW); cudaFree(dY);
    free(hX); free(hW); free(hY);
    return 0;
}
