// flash_attn.cu — Flash Attention v1 (single-head, FP32)
// Q[seq_len, d], K[seq_len, d], V[seq_len, d] --> O[seq_len, d]
// Scale: 1 / sqrt(d)
//
// Algorithm: tiled online softmax (Dao et al. 2022, Algorithm 1)
//   For each Q tile (Br rows):
//     - Initialize running max m=-inf, sum l=0, output o=0
//     - Iterate over K/V tiles (Bc cols):
//         1. Load K_tile, V_tile into shared memory
//         2. Compute scores s[k] = scale * dot(q, K_tile[k])
//         3. Update running max m_new = max(m, max(s))
//         4. Rescale: o *= exp(m - m_new),  l *= exp(m - m_new)
//         5. Accumulate: p[k]=exp(s[k]-m_new), o += p[k]*V_tile[k], l += p[k]
//     - Normalize: O[row] = o / l
//
// Grid:  (ceil(seq_len / BLOCK_SIZE),)  — one CTA per Q tile
// Block: (BLOCK_SIZE,)                  — one thread per query row in tile
// Smem:  sK[BLOCK_SIZE * d] + sV[BLOCK_SIZE * d]  (dynamic)
//
// Compile (CUDA): nvcc -O2 -o flash_attn flash_attn.cu
// Compile (DCU):  hipcc -O2 -o flash_attn flash_attn.cu
// Usage:          ./flash_attn <seq_len> <head_dim> <Q.bin> <K.bin> <V.bin> <O.bin>
//   All .bin: raw float32, seq_len*head_dim elements, row-major.

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
  constexpr int WARP_SIZE = 64;   // DCU (gfx906)
#else
  constexpr int WARP_SIZE = 32;   // NVIDIA CUDA
#endif

constexpr int BLOCK_SIZE    = 64;    // Br = Bc; must be a power of 2
constexpr int MAX_HEAD_DIM  = 128;   // runtime head_dim must satisfy d <= MAX_HEAD_DIM

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
constexpr int VEC = 4;   // vector width for float4 tile loads

// Two "amortize the K/V tile load over more query rows" experiments were
// tried here and both measurably REGRESSED performance (see dcu_perf.md
// methodology: test, don't assume — this is the negative-result case):
//
//  1. Register blocking: give each thread TM=4 query rows instead of 1 (the
//     GEMM-style TMxTN micro-tile idea). 4-5x SLOWER everywhere. Unlike GEMM,
//     where a thread's TM/TN registers are what make reuse possible at all,
//     here q[MAX_HEAD_DIM]+o[MAX_HEAD_DIM] per row already spills to local
//     memory (128+128 floats vastly exceeds the ~255-register budget).
//     Quadrupling that already-spilling footprint AND cutting grid.x (the
//     parallelism available to hide the resulting local-memory latency) by
//     the same 4x was a double cost with no compensating win.
//  2. Growing BLOCK_SIZE itself (e.g. 64->128) at runtime based on the
//     device's shared-memory budget: also SLOWER (up to ~3x). Reason this
//     time: growing BLOCK_SIZE shrinks grid.x just like TM did (e.g.
//     seq_len=100 dropped from 2 blocks to 1). For the seq_len values that
//     matter here (100-4096), grid-level parallelism — having enough
//     independent blocks to fill this GPU's ~170 SMs — is already the
//     scarce resource, unlike GEMM's large/extreme shapes where plenty of
//     blocks remained after tiling up. Cutting block count backfires here
//     even with zero added per-thread register pressure.
//
// Net: BLOCK_SIZE=64 (unchanged from the original) plus the float4 tile-load
// vectorization above is the best validated configuration for this kernel.
__global__ void flash_attn_kernel(
    const float* __restrict__ Q,    // [seq_len, d]
    const float* __restrict__ K,    // [seq_len, d]
    const float* __restrict__ V,    // [seq_len, d]
    float*       __restrict__ O,    // [seq_len, d]
    int seq_len, int d, float scale)
{
    // Shared memory layout: [sK | sV], each BLOCK_SIZE * d floats.
    // __align__(16): base must be 16B-aligned for the float4 tile loads below
    // (a plain extern float[] only guarantees 4B alignment). BLOCK_SIZE*d is
    // always a multiple of 4 (BLOCK_SIZE=64), so sV's offset stays aligned too.
    extern __shared__ __align__(16) float smem[];
    float* sK = smem;
    float* sV = smem + BLOCK_SIZE * d;

    const int tid   = threadIdx.x;
    const int q_row = blockIdx.x * BLOCK_SIZE + tid;  // global query row

    // Per-thread accumulators in local memory (L1-cached on all GPUs)
    float q[MAX_HEAD_DIM];   // this thread's query vector
    float o[MAX_HEAD_DIM];   // running unnormalized output
    float m = -FLT_MAX;      // running row-max
    float l = 0.0f;          // running sum of exp weights

    if (q_row < seq_len) {
        for (int i = 0; i < d; ++i) {
            q[i] = Q[q_row * d + i];
            o[i] = 0.0f;
        }
    }

    const int num_kv = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv = 0; kv < num_kv; ++kv) {
        const int kv_start = kv * BLOCK_SIZE;
        const int valid_kv = min(BLOCK_SIZE, seq_len - kv_start);

        // --- Cooperative tile load: all threads load sK and sV ---
        // Fast path: whole tile in-bounds (no row past seq_len) and d is a
        // multiple of 4 -> vectorized float4 loads. Otherwise (last, partial
        // kv-tile, or unaligned d) fall back to the original scalar path;
        // a float4 read of K[r*d+c] is only guaranteed 16B-aligned when the
        // row stride d itself is a multiple of 4 elements.
        const bool d_aligned    = (d % VEC == 0);
        const bool kv_tile_full = (kv_start + BLOCK_SIZE - 1) < seq_len;

        if (d_aligned && kv_tile_full) {
            const int total_vec = (BLOCK_SIZE * d) / VEC;
            for (int v = tid; v < total_vec; v += BLOCK_SIZE) {
                const int row  = (v * VEC) / d;
                const int col0 = (v * VEC) % d;
                const int grow = kv_start + row;
                *reinterpret_cast<float4*>(&sK[row * d + col0]) =
                    *reinterpret_cast<const float4*>(&K[(size_t)grow * d + col0]);
                *reinterpret_cast<float4*>(&sV[row * d + col0]) =
                    *reinterpret_cast<const float4*>(&V[(size_t)grow * d + col0]);
            }
        } else {
            // Access pattern: consecutive threads → consecutive columns → coalesced.
            for (int elem = tid; elem < BLOCK_SIZE * d; elem += BLOCK_SIZE) {
                const int row = elem / d, col = elem % d;
                const int grow = kv_start + row;
                sK[elem] = (grow < seq_len) ? K[grow * d + col] : 0.0f;
                sV[elem] = (grow < seq_len) ? V[grow * d + col] : 0.0f;
            }
        }
        __syncthreads();  // tiles ready

        if (q_row < seq_len) {
            // --- Compute QK^T scores for this tile ---
            float s[BLOCK_SIZE];
            for (int k = 0; k < valid_kv; ++k) {
                float dot = 0.0f;
                for (int i = 0; i < d; ++i)
                    dot += q[i] * sK[k * d + i];
                s[k] = scale * dot;
            }

            // --- Online softmax: find new running max ---
            float m_new = m;
            for (int k = 0; k < valid_kv; ++k)
                m_new = fmaxf(m_new, s[k]);

            // --- Rescale previous accumulator with correction factor ---
            const float alpha = expf(m - m_new);  // in (0, 1]
            float l_new = alpha * l;
            for (int i = 0; i < d; ++i)
                o[i] *= alpha;

            // --- Accumulate current tile: o += sum_k p_k * V_k ---
            for (int k = 0; k < valid_kv; ++k) {
                const float p = expf(s[k] - m_new);
                l_new += p;
                for (int i = 0; i < d; ++i)
                    o[i] += p * sV[k * d + i];
            }

            m = m_new;
            l = l_new;
        }

        __syncthreads();  // guard sK/sV before next tile overwrites them
    }

    // --- Final normalisation and write ---
    if (q_row < seq_len) {
        for (int i = 0; i < d; ++i)
            O[q_row * d + i] = o[i] / l;
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

// Dynamic shared memory above 48 KiB requires an explicit opt-in per kernel
// (true on all compute capability 7.0+ devices); without it, the launch
// silently fails and every thread block does nothing.
//
// DCU (gfx906) has no NVIDIA-style "48KiB default + opt-in" two-tier model --
// LDS is a single fixed budget per workgroup (64 KiB on gfx906). The 48*1024
// threshold below is therefore CUDA-specific and meaningless on the HIP path
// (see docs/dcu_portability_review.md, "建议 2" -- this branch was flagged
// high-risk/untested until now). On HIP we instead always query the real
// device limit via hipGetDeviceProperties().sharedMemPerBlock, fail loudly
// with a clear message if the requested shape doesn't fit (e.g. the extreme
// shape seq_len=4096/head_dim=128 needs exactly 64 KiB -- right at the
// gfx906 ceiling, no headroom), and always call cudaFuncSetAttribute
// (mapped to hipFuncSetAttribute via gpu_compat.h) rather than gating it
// behind the 48KiB CUDA threshold.
static void configure_smem(size_t smem_bytes)
{
#ifdef __HIPCC__
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "get device properties");
    if (smem_bytes > (size_t)prop.sharedMemPerBlock) {
        fprintf(stderr,
                "flash_attn: requested dynamic shared memory %zu bytes "
                "exceeds device limit %zu bytes (LDS per workgroup on this "
                "DCU) -- reduce BLOCK_SIZE or head_dim for this shape\n",
                smem_bytes, prop.sharedMemPerBlock);
        exit(1);
    }
    check(cudaFuncSetAttribute((const void*)flash_attn_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem_bytes),
          "set max dynamic shared memory");
#else
    if (smem_bytes > 48 * 1024) {
        check(cudaFuncSetAttribute((const void*)flash_attn_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes),
              "set max dynamic shared memory");
    }
#endif
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
// Usage: ./flash_attn --bench <seq_len> <head_dim> [warmup=10] [iters=50]
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
        fprintf(stderr, "Usage: %s --bench <seq_len> <head_dim> [warmup=10] [iters=50]\n", argv[0]);
        return 1;
    }
    int seq_len  = atoi(argv[2]);
    int head_dim = atoi(argv[3]);
    int warmup   = (argc > 4) ? atoi(argv[4]) : 10;
    int iters    = (argc > 5) ? atoi(argv[5]) : 50;

    if (head_dim > MAX_HEAD_DIM) {
        fprintf(stderr, "head_dim %d > MAX_HEAD_DIM %d\n", head_dim, MAX_HEAD_DIM);
        return 1;
    }

    const size_t n     = (size_t)seq_len * head_dim;
    const size_t sz    = n * sizeof(float);
    const float  scale = 1.0f / sqrtf((float)head_dim);

    float* hQ = (float*)malloc(sz);
    float* hK = (float*)malloc(sz);
    float* hV = (float*)malloc(sz);
    srand(42);
    for (size_t i = 0; i < n; ++i) hQ[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < n; ++i) hK[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < n; ++i) hV[i] = (float)rand() / RAND_MAX;

    float *dQ, *dK, *dV, *dO;
    check(cudaMalloc(&dQ, sz), "malloc Q");
    check(cudaMalloc(&dK, sz), "malloc K");
    check(cudaMalloc(&dV, sz), "malloc V");
    check(cudaMalloc(&dO, sz), "malloc O");
    check(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice), "copy Q");
    check(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice), "copy K");
    check(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice), "copy V");

    dim3   block(BLOCK_SIZE);
    dim3   grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t smem_bytes = 2ull * BLOCK_SIZE * head_dim * sizeof(float);
    configure_smem(smem_bytes);

    for (int i = 0; i < warmup; ++i) {
        flash_attn_kernel<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, seq_len, head_dim, scale);
        check(cudaGetLastError(), "warmup kernel launch");
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(start), "record start");
        flash_attn_kernel<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, seq_len, head_dim, scale);
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
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(times);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0)
        return run_benchmark(argc, argv);

    if (argc != 7) {
        fprintf(stderr,
            "Usage: %s <seq_len> <head_dim> <Q.bin> <K.bin> <V.bin> <O.bin>\n"
            "  All .bin: raw float32, seq_len*head_dim elements, row-major\n"
            "  head_dim must be <= %d\n",
            argv[0], MAX_HEAD_DIM);
        return 1;
    }
    int seq_len  = atoi(argv[1]);
    int head_dim = atoi(argv[2]);
    const char *fQ = argv[3], *fK = argv[4], *fV = argv[5], *fO = argv[6];

    if (head_dim > MAX_HEAD_DIM) {
        fprintf(stderr, "head_dim %d > MAX_HEAD_DIM %d\n", head_dim, MAX_HEAD_DIM);
        return 1;
    }

    const size_t n     = (size_t)seq_len * head_dim;
    const size_t sz    = n * sizeof(float);
    const float  scale = 1.0f / sqrtf((float)head_dim);

    float* hQ = read_binary(fQ, n);
    float* hK = read_binary(fK, n);
    float* hV = read_binary(fV, n);
    float* hO = (float*)calloc(n, sizeof(float));

    float *dQ, *dK, *dV, *dO;
    check(cudaMalloc(&dQ, sz), "malloc Q");
    check(cudaMalloc(&dK, sz), "malloc K");
    check(cudaMalloc(&dV, sz), "malloc V");
    check(cudaMalloc(&dO, sz), "malloc O");

    check(cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice), "copy Q");
    check(cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice), "copy K");
    check(cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice), "copy V");
    check(cudaMemset(dO, 0, sz), "memset O");

    dim3  block(BLOCK_SIZE);
    dim3  grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t smem_bytes = 2ull * BLOCK_SIZE * head_dim * sizeof(float);
    configure_smem(smem_bytes);

    flash_attn_kernel<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, seq_len, head_dim, scale);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    check(cudaMemcpy(hO, dO, sz, cudaMemcpyDeviceToHost), "copy O back");
    write_binary(fO, hO, n);

    const int pn = (head_dim < 8) ? head_dim : 8;
    printf("O[row=0, 0:%d] =", pn);
    for (int i = 0; i < pn; ++i) printf(" %.6f", hO[i]);
    printf("\n");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO);
    return 0;
}
