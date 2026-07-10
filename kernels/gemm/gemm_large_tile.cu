// gemm_large_tile.cu — FP32 GEMM with dynamic shared memory, runtime-selectable tile size.
// A[M,K] * B[K,N] = C[M,N]
// Compile (CUDA):  nvcc -O2 -o gemm_large_tile gemm_large_tile.cu
// Compile (DCU):   hipcc -O2 -o gemm_large_tile gemm_large_tile.cu
// Usage:           ./gemm_large_tile [--tile=small|medium|large|xlarge] <M> <K> <N> <file_A> <file_B> [file_C]
//   file_A : raw binary, M*K float32, row-major
//   file_B : raw binary, K*N float32, row-major
//   file_C : (optional) raw binary output, M*N float32, row-major
//
// Extends kernels/gemm/gemm.cu (same overall structure, double-accumulator
// precision discipline -- see .claude/skills/dcu_numerics.md) with one
// change: the A/B shared-memory tiles are dynamic (extern __shared__, size
// picked at kernel-launch time from a runtime-selected tile config) instead
// of gemm.cu's compile-time-fixed static __shared__ arrays. The "xlarge"
// config's dynamic shared-memory request (64 KiB) is deliberately above
// CUDA's 48 KiB static/opt-in threshold, exercising the same
// cudaFuncSetAttribute opt-in + gfx906-LDS-budget-check path already
// validated for kernels/flash_attention/flash_attn.cu (see
// .claude/skills/dcu_hip_porting.md, sections 3-4).

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

constexpr int VEC = 4;   // float4 vector width for shared-mem fragment loads

// ---------------------------------------------------------------------------
// Tile configs: BM/BN/BK is the shared-memory tile shape; TM/TN is the
// per-thread register micro-tile. Each config is a distinct template
// instantiation of the kernel below, so BM/BN/BK/TM/TN are still
// compile-time constants inside the kernel body (required for the
// #pragma unroll loops and the fixed-size a_frag/b_frag arrays) even though
// the choice *among* configs is made at runtime (see parse_tile_config /
// dispatch_gemm).
//
// Only BK (the K-tile depth) grows across configs; BM=BN=64 and TM=TN=4 are
// kept fixed at gemm.cu's validated values. BM/BN/TM/TN together determine
// NUM_THREADS = (BM/TM)*(BN/TN) and therefore per-thread register pressure
// (acc[TM][TN] is TM*TN doubles) -- growing BM/BN instead of BK to reach a
// larger tile was tried first and hit a real limit: BM=BN=128 with TM=TN=4
// needs 1024 threads/block, and the resulting register count per thread
// exceeded the SM's register file for that block size (nvcc/driver reported
// "too many resources requested for launch" at kernel-launch time on the
// RTX 5090 dev box -- not a correctness bug, a hard register-file ceiling).
// Growing BK instead leaves NUM_THREADS and the register footprint
// unchanged across every config (only the shared-memory tile grows), so it
// reaches the required >48 KiB dynamic-shared-memory config without
// re-triggering that limit.
// ---------------------------------------------------------------------------
struct ConfigSmall  { static constexpr int BM = 64, BN = 64, BK = 16,  TM = 4, TN = 4; };
struct ConfigMedium { static constexpr int BM = 64, BN = 64, BK = 48,  TM = 4, TN = 4; };
struct ConfigLarge  { static constexpr int BM = 64, BN = 64, BK = 80,  TM = 4, TN = 4; };
// xlarge: (64*128 + 128*64) floats * 4 bytes = 64 KiB dynamic shared memory
// per block -- above CUDA's 48 KiB static/opt-in threshold (requires the
// cudaFuncSetAttribute opt-in below, or the launch silently no-ops), and
// exactly at gfx906's 64 KiB LDS-per-workgroup ceiling (no headroom -- see
// dcu_hip_porting.md #4, same situation as flash_attn.cu's extreme shape).
struct ConfigXLarge { static constexpr int BM = 64, BN = 64, BK = 128, TM = 4, TN = 4; };

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_large_tile_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float*       __restrict__ C,
                                        int M, int K, int N)
{
    // Dynamic shared memory: sA[BK][BM] (transposed, as in gemm.cu -- for a
    // fixed k, the TM rows a thread needs are contiguous, making the
    // per-k fragment load a single float4 read) followed by sB[BK][BN].
    // Size is decided at launch time (see launch_gemm), not baked into the
    // binary as a fixed-size __shared__ array.
    extern __shared__ __align__(16) float smem[];
    float* sA = smem;               // [BK][BM]
    float* sB = smem + BK * BM;     // [BK][BN]

    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr int A_VEC_SLOTS = (BM * BK) / VEC;
    constexpr int B_VEC_SLOTS = (BK * BN) / VEC;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.y * (BN / TN) + threadIdx.x;

    // Accumulate in FP64 (precision discipline, not an optimizable knob --
    // see .claude/skills/dcu_numerics.md section 4). Tiles in shared memory
    // stay float32; only the running sums are promoted.
    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    const bool k_aligned = (K % VEC == 0);
    const bool n_aligned = (N % VEC == 0);

    const int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        const int k0 = t * BK;

        // --- Load A tile [BM x BK] -> sA[BK][BM] (transposed) ---
        const bool a_full = k_aligned &&
            (block_row + BM - 1 < M) && (k0 + BK - 1 < K);
        if (a_full) {
            for (int vidx = tid; vidx < A_VEC_SLOTS; vidx += NUM_THREADS) {
                const int m      = vidx / (BK / VEC);
                const int kgroup = vidx % (BK / VEC);
                const int kk0    = kgroup * VEC;
                float4 v = *reinterpret_cast<const float4*>(
                    &A[(size_t)(block_row + m) * K + (k0 + kk0)]);
                sA[(kk0 + 0) * BM + m] = v.x;
                sA[(kk0 + 1) * BM + m] = v.y;
                sA[(kk0 + 2) * BM + m] = v.z;
                sA[(kk0 + 3) * BM + m] = v.w;
            }
        } else {
            for (int idx = tid; idx < BM * BK; idx += NUM_THREADS) {
                const int m  = idx / BK;
                const int kk = idx % BK;
                const int g_row = block_row + m;
                const int g_col = k0 + kk;
                sA[kk * BM + m] = (g_row < M && g_col < K) ? A[(size_t)g_row * K + g_col] : 0.0f;
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
                *reinterpret_cast<float4*>(&sB[k * BN + n0]) = *reinterpret_cast<const float4*>(
                    &B[(size_t)(k0 + k) * N + (block_col + n0)]);
            }
        } else {
            for (int idx = tid; idx < BK * BN; idx += NUM_THREADS) {
                const int k = idx / BN;
                const int n = idx % BN;
                const int g_row = k0 + k;
                const int g_col = block_col + n;
                sB[k * BN + n] = (g_row < K && g_col < N) ? B[(size_t)g_row * N + g_col] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM], b_frag[TN];
            *reinterpret_cast<float4*>(a_frag) =
                *reinterpret_cast<float4*>(&sA[kk * BM + threadIdx.y * TM]);
            *reinterpret_cast<float4*>(b_frag) =
                *reinterpret_cast<float4*>(&sB[kk * BN + threadIdx.x * TN]);

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += (double)a_frag[i] * (double)b_frag[j];
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

// Same opt-in pattern as kernels/flash_attention/flash_attn.cu's
// configure_smem(): CUDA's "dynamic shared memory above 48 KiB needs an
// explicit per-kernel opt-in" is a two-tier model with no DCU equivalent
// (gfx906 LDS is a single fixed 64 KiB-per-workgroup budget), so the HIP
// branch always queries the real device limit instead of reusing the
// 48 KiB CUDA threshold. kernel_func takes const void* (not a typed function
// pointer) so this one function serves every tile-config instantiation
// without templating; callers must pass an explicit (const void*) cast --
// hipcc's Clang frontend rejects the implicit function-pointer-to-void*
// conversion that nvcc/MSVC silently allow (see dcu_hip_porting.md #3).
static void configure_smem(const void* kernel_func, size_t smem_bytes)
{
#ifdef __HIPCC__
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "get device properties");
    if (smem_bytes > (size_t)prop.sharedMemPerBlock) {
        fprintf(stderr,
                "gemm_large_tile: requested dynamic shared memory %zu bytes "
                "exceeds device limit %zu bytes (LDS per workgroup on this "
                "DCU) -- pick a smaller --tile config for this shape\n",
                smem_bytes, (size_t)prop.sharedMemPerBlock);
        exit(1);
    }
    check(cudaFuncSetAttribute(kernel_func,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem_bytes),
          "set max dynamic shared memory");
#else
    if (smem_bytes > 48 * 1024) {
        check(cudaFuncSetAttribute(kernel_func,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes),
              "set max dynamic shared memory");
    }
#endif
}

template <typename Cfg>
static void launch_gemm(const float* dA, const float* dB, float* dC,
                         int M, int K, int N)
{
    constexpr int BM = Cfg::BM, BN = Cfg::BN, BK = Cfg::BK;
    constexpr int TM = Cfg::TM, TN = Cfg::TN;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const size_t smem_bytes = (size_t)(BM * BK + BK * BN) * sizeof(float);

    auto kernel = gemm_large_tile_kernel<BM, BN, BK, TM, TN>;
    configure_smem((const void*)kernel, smem_bytes);
    kernel<<<grid, block, smem_bytes>>>(dA, dB, dC, M, K, N);
}

enum class TileConfig { Small, Medium, Large, XLarge };

static TileConfig parse_tile_config(const char* name)
{
    if (strcmp(name, "small") == 0)  return TileConfig::Small;
    if (strcmp(name, "medium") == 0) return TileConfig::Medium;
    if (strcmp(name, "large") == 0)  return TileConfig::Large;
    if (strcmp(name, "xlarge") == 0) return TileConfig::XLarge;
    fprintf(stderr, "Unknown --tile value '%s' (expected small|medium|large|xlarge)\n", name);
    exit(1);
}

static void print_tile_config(TileConfig cfg)
{
    const char* name = "";
    size_t smem_bytes = 0;
    switch (cfg) {
        case TileConfig::Small:
            name = "small";
            smem_bytes = (size_t)(ConfigSmall::BM * ConfigSmall::BK + ConfigSmall::BK * ConfigSmall::BN) * sizeof(float);
            break;
        case TileConfig::Medium:
            name = "medium";
            smem_bytes = (size_t)(ConfigMedium::BM * ConfigMedium::BK + ConfigMedium::BK * ConfigMedium::BN) * sizeof(float);
            break;
        case TileConfig::Large:
            name = "large";
            smem_bytes = (size_t)(ConfigLarge::BM * ConfigLarge::BK + ConfigLarge::BK * ConfigLarge::BN) * sizeof(float);
            break;
        case TileConfig::XLarge:
            name = "xlarge";
            smem_bytes = (size_t)(ConfigXLarge::BM * ConfigXLarge::BK + ConfigXLarge::BK * ConfigXLarge::BN) * sizeof(float);
            break;
    }
    fprintf(stderr, "gemm_large_tile: tile=%s dynamic_smem=%zu bytes (%.1f KiB)\n",
            name, smem_bytes, smem_bytes / 1024.0);
}

static void dispatch_gemm(TileConfig cfg, const float* dA, const float* dB, float* dC,
                           int M, int K, int N)
{
    switch (cfg) {
        case TileConfig::Small:  launch_gemm<ConfigSmall> (dA, dB, dC, M, K, N); break;
        case TileConfig::Medium: launch_gemm<ConfigMedium>(dA, dB, dC, M, K, N); break;
        case TileConfig::Large:  launch_gemm<ConfigLarge> (dA, dB, dC, M, K, N); break;
        case TileConfig::XLarge: launch_gemm<ConfigXLarge>(dA, dB, dC, M, K, N); break;
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
// Usage: ./gemm_large_tile --bench [--tile=...] <M> <K> <N> [warmup=10] [iters=50]
// Prints: BENCH_MS <median>
// ---------------------------------------------------------------------------
static int cmp_float(const void* a, const void* b)
{
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

static int run_benchmark(int argc, char* argv[], TileConfig cfg, int arg0)
{
    if (argc < arg0 + 3) {
        fprintf(stderr, "Usage: %s --bench [--tile=...] <M> <K> <N> [warmup=10] [iters=50]\n", argv[0]);
        return 1;
    }
    int M      = atoi(argv[arg0]);
    int K      = atoi(argv[arg0 + 1]);
    int N      = atoi(argv[arg0 + 2]);
    int warmup = (argc > arg0 + 3) ? atoi(argv[arg0 + 3]) : 10;
    int iters  = (argc > arg0 + 4) ? atoi(argv[arg0 + 4]) : 50;

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

    print_tile_config(cfg);

    for (int i = 0; i < warmup; ++i) {
        dispatch_gemm(cfg, dA, dB, dC, M, K, N);
        check(cudaGetLastError(), "warmup kernel launch");
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(start), "record start");
        dispatch_gemm(cfg, dA, dB, dC, M, K, N);
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
    TileConfig cfg = TileConfig::Large;   // default: 40 KiB, no opt-in needed on either platform
    int arg = 1;
    bool bench = false;

    if (arg < argc && strcmp(argv[arg], "--bench") == 0) {
        bench = true;
        ++arg;
    }
    if (arg < argc && strncmp(argv[arg], "--tile=", 7) == 0) {
        cfg = parse_tile_config(argv[arg] + 7);
        ++arg;
    }

    if (bench)
        return run_benchmark(argc, argv, cfg, arg);

    if (argc - arg < 5 || argc - arg > 6) {
        fprintf(stderr, "Usage: %s [--tile=small|medium|large|xlarge] <M> <K> <N> <file_A> <file_B> [file_C]\n", argv[0]);
        fprintf(stderr, "  file_A: raw float32 binary, M*K elements, row-major\n");
        fprintf(stderr, "  file_B: raw float32 binary, K*N elements, row-major\n");
        fprintf(stderr, "  file_C: (optional) raw float32 binary output, M*N elements\n");
        fprintf(stderr, "  --tile: shared-memory tile config (default: large); xlarge needs 64 KiB dynamic smem\n");
        return 1;
    }
    int M = atoi(argv[arg]);
    int K = atoi(argv[arg + 1]);
    int N = atoi(argv[arg + 2]);
    const char* file_A = argv[arg + 3];
    const char* file_B = argv[arg + 4];
    const char* file_C = (argc - arg == 6) ? argv[arg + 5] : nullptr;

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

    print_tile_config(cfg);

    dispatch_gemm(cfg, dA, dB, dC, M, K, N);
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
