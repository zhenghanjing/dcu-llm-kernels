// gemm.cu — FP32 GEMM with shared-memory tiling
// A[M,K] * B[K,N] = C[M,N]
// Compile (CUDA):  nvcc -O2 -o gemm gemm.cu
// Compile (DCU):   hipcc -O2 -o gemm gemm.cu
// Usage:           ./gemm <M> <K> <N> <file_A> <file_B> [file_C]
//   file_A : raw binary, M*K float32, row-major
//   file_B : raw binary, K*N float32, row-major
//   file_C : (optional) raw binary output, M*N float32, row-major

#include "../common/gpu_compat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// __HIPCC__, not __HIP_PLATFORM_HCC__/__HIP_PLATFORM_AMD__: those are
// defined by hip_runtime.h itself, not by the compiler driver, so checking
// them here (before/without including that header) never actually
// triggered on real hardware. See kernels/common/gpu_compat.h.
#ifdef __HIPCC__
  constexpr int WARP_SIZE = 64;
#else
  constexpr int WARP_SIZE = 32;
#endif

constexpr int TILE = 16;   // kept as the K-tile depth (BK below)

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
// Register blocking: each thread computes a TMxTN micro-tile of C instead of
// a single element, so each value loaded from shared memory is reused TM (or
// TN) times in FMAs instead of once — this is what actually raises the
// compute:memory-access ratio (float4 loads alone, step 1, left it unchanged).
constexpr int BK = TILE;   // K-dimension tile depth
constexpr int BM = 64;     // output rows per block
#ifndef GEMM_BN
#define GEMM_BN 64
#endif
constexpr int BN = GEMM_BN;   // output cols per block
constexpr int TM = 4;      // output rows per thread
constexpr int TN = 4;      // output cols per thread
constexpr int VEC = 4;     // vector width for float4 loads/stores (== TM == TN)
constexpr int NUM_THREADS = (BM / TM) * (BN / TN);   // 256

// Tile-size tuning notes (BM/BN/BK tried against the benchmark suite):
//   - The B-fragment load below (sB[kk][threadIdx.x*TN]) has a real 2-way
//     shared-memory bank conflict: with blockDim.x = BN/TN = 16 distinct
//     addresses active per warp but only 32/TN = 8 non-aliasing 4-bank slots,
//     groups i and i+8 collide. Shrinking BN to 32 removes it (verified: no
//     collision when BN <= 32) but doubles how often each A-tile column gets
//     reloaded from global memory across N-blocks, which cost more than the
//     conflict did — measured ~5% slower on 1024^3/4096^3 despite ~45% faster
//     on the small non-2^n shape. Kept BN=64 since large/extreme is the target.
//   - Doubling BK to 32 (deeper K-tile, fewer syncthreads/loop iterations)
//     was also tried: strictly worse everywhere (~2-6%), because the doubled
//     shared-memory footprint (16 KiB/block vs 8 KiB) lowers achievable
//     occupancy by more than the loop-overhead savings are worth.
//   - Net: BM=BN=64, BK=16 (this file) benchmarked best for the shapes that
//     matter (1024^3, 4096^3), so the known bank conflict was left in place.
//   - Now buildable with -DGEMM_BN=32 to compile the conflict-free variant
//     for A/B comparison; DCU real-hardware results still pending.
constexpr int A_VEC_SLOTS = (BM * BK) / VEC;
constexpr int B_VEC_SLOTS = (BK * BN) / VEC;

__global__ void gemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int K, int N)
{
    // sA is stored TRANSPOSED as [k][m]: for a fixed k, the TM=4 rows a thread
    // needs are contiguous in shared memory, making the per-k fragment load
    // below a single float4 read instead of 4 scalar ones.
    __shared__ __align__(16) float sA[BK][BM];
    __shared__ __align__(16) float sB[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.y * (BN / TN) + threadIdx.x;   // 0..255

    // Accumulate in FP64: eliminates O(K·ε) rounding drift from FP32 summation.
    // Tiles in shared memory stay float32; only the running sums are promoted.
#ifndef GEMM_ACC_T
#define GEMM_ACC_T double
#endif
    GEMM_ACC_T acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    // A float4 global read/write is only guaranteed 16B-aligned when the row
    // stride (K for A, N for B/C) is itself a multiple of 4 elements.
    // Non-multiple-of-4 K/N (e.g. K=513) fall back to scalar masked paths.
    const bool k_aligned = (K % VEC == 0);
    const bool n_aligned = (N % VEC == 0);

    const int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        const int k0 = t * BK;

        // --- Load A tile [BM x BK] -> sA[BK][BM] (transposed) ---
        // Strided over vector-slots (not a single slot/thread) so this stays
        // correct regardless of how BM*BK/VEC compares to NUM_THREADS.
        const bool a_full = k_aligned &&
            (block_row + BM - 1 < M) && (k0 + BK - 1 < K);
        if (a_full) {
            for (int vidx = tid; vidx < A_VEC_SLOTS; vidx += NUM_THREADS) {
                const int m      = vidx / (BK / VEC);
                const int kgroup = vidx % (BK / VEC);
                const int kk0    = kgroup * VEC;
                float4 v = *reinterpret_cast<const float4*>(
                    &A[(size_t)(block_row + m) * K + (k0 + kk0)]);
                sA[kk0 + 0][m] = v.x;
                sA[kk0 + 1][m] = v.y;
                sA[kk0 + 2][m] = v.z;
                sA[kk0 + 3][m] = v.w;
            }
        } else {
            for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
                const int m  = idx / BK;
                const int kk = idx % BK;
                const int g_row = block_row + m;
                const int g_col = k0 + kk;
                sA[kk][m] = (g_row < M && g_col < K) ? A[(size_t)g_row * K + g_col] : 0.0f;
            }
        }

        // --- Load B tile [BK x BN] -> sB[BK][BN] (direct) ---
        const bool b_full = n_aligned &&
            (k0 + BK - 1 < K) && (block_col + BN - 1 < N);
        if (b_full) {
            for (int vidx = tid; vidx < B_VEC_SLOTS; vidx += NUM_THREADS) {
                const int k      = vidx / (BN / VEC);
                const int ngroup = vidx % (BN / VEC);
                const int n0     = ngroup * VEC;
                *reinterpret_cast<float4*>(&sB[k][n0]) = *reinterpret_cast<const float4*>(
                    &B[(size_t)(k0 + k) * N + (block_col + n0)]);
            }
        } else {
            for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
                const int k = idx / BN;
                const int n = idx % BN;
                const int g_row = k0 + k;
                const int g_col = block_col + n;
                sB[k][n] = (g_row < K && g_col < N) ? B[(size_t)g_row * N + g_col] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM], b_frag[TN];
            *reinterpret_cast<float4*>(a_frag) =
                *reinterpret_cast<float4*>(&sA[kk][threadIdx.y * TM]);
            *reinterpret_cast<float4*>(b_frag) =
                *reinterpret_cast<float4*>(&sB[kk][threadIdx.x * TN]);

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += (GEMM_ACC_T)a_frag[i] * (GEMM_ACC_T)b_frag[j];
        }

        __syncthreads();
    }

    const int row_base = block_row + threadIdx.y * TM;
    const int col_base = block_col + threadIdx.x * TN;
    const bool store_full = n_aligned && (col_base + TN - 1 < N);

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = row_base + i;
        if (row >= M) continue;
        if (store_full) {
            float4 out = make_float4((float)acc[i][0], (float)acc[i][1],
                                      (float)acc[i][2], (float)acc[i][3]);
            *reinterpret_cast<float4*>(&C[(size_t)row * N + col_base]) = out;
        } else {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = col_base + j;
                if (col < N) C[(size_t)row * N + col] = (float)acc[i][j];
            }
        }
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

static float* read_binary(const char* path, size_t n_floats)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open '%s'\n", path);
        exit(1);
    }
    float* buf = (float*)malloc(n_floats * sizeof(float));
    size_t got = fread(buf, sizeof(float), n_floats, f);
    fclose(f);
    if (got != n_floats) {
        fprintf(stderr, "Expected %zu floats from '%s', got %zu\n", n_floats, path, got);
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
// Usage: ./gemm --bench <M> <K> <N> [warmup=10] [iters=50]
// Prints: BENCH_MS <median>
// ---------------------------------------------------------------------------
static int cmp_float(const void* a, const void* b)
{
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static int run_benchmark(int argc, char* argv[])
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s --bench <M> <K> <N> [warmup=10] [iters=50]\n", argv[0]);
        return 1;
    }
    int M      = atoi(argv[2]);
    int K      = atoi(argv[3]);
    int N      = atoi(argv[4]);
    int warmup = (argc > 5) ? atoi(argv[5]) : 10;
    int iters  = (argc > 6) ? atoi(argv[6]) : 50;

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float* hA = (float*)malloc(szA);
    float* hB = (float*)malloc(szB);
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; ++i) hA[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; ++i) hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    check(cudaMalloc(&dA, szA), "malloc A");
    check(cudaMalloc(&dB, szB), "malloc B");
    check(cudaMalloc(&dC, szC), "malloc C");
    check(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice), "copy B");

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup; ++i) {
        gemm_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
        check(cudaGetLastError(), "warmup kernel launch");
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(start), "record start");
        gemm_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
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
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(times);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0)
        return run_benchmark(argc, argv);

    if (argc < 6 || argc > 7) {
        fprintf(stderr, "Usage: %s <M> <K> <N> <file_A> <file_B> [file_C]\n", argv[0]);
        fprintf(stderr, "  file_A: raw float32 binary, M*K elements, row-major\n");
        fprintf(stderr, "  file_B: raw float32 binary, K*N elements, row-major\n");
        fprintf(stderr, "  file_C: (optional) raw float32 binary output, M*N elements\n");
        return 1;
    }
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    const char* file_A = argv[4];
    const char* file_B = argv[5];
    const char* file_C = (argc == 7) ? argv[6] : nullptr;

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float* hA = read_binary(file_A, (size_t)M * K);
    float* hB = read_binary(file_B, (size_t)K * N);
    float* hC = (float*)calloc(M * N, sizeof(float));

    float *dA, *dB, *dC;
    check(cudaMalloc(&dA, szA), "malloc A");
    check(cudaMalloc(&dB, szB), "malloc B");
    check(cudaMalloc(&dC, szC), "malloc C");

    check(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice), "copy B");
    check(cudaMemset(dC, 0, szC), "memset C");

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    check(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost), "copy C back");

    if (file_C)
        write_binary(file_C, hC, (size_t)M * N);

    int print_n = (M * N < 8) ? M * N : 8;
    printf("C[0:%d] =", print_n);
    for (int i = 0; i < print_n; ++i)
        printf(" %.6f", hC[i]);
    printf("\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
