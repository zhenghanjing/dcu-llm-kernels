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
//
// BM/BN/TM/TN/double-buffering are now gemm_tiled<> TEMPLATE parameters (see
// below), not fixed file-scope constants — there are two known-good configs
// (the original 64/64/4/4 default and Experiment A's 128/128/8/8, see
// .claude/skills/dcu_perf.md), and the framework to pick between them
// per-call based on grid size still exists (use_large_tile()/launch_gemm()
// further down) but is OFF by default: real DCU hardware measured the
// 128/128/8/8 config as a net loss at the one shape (4096^3) it was
// supposed to win on, opposite of the local RTX 5090 result — see
// GEMM_ENABLE_AUTO_DISPATCH further down for the full story. The default
// (no macros) build always uses 64/64/4/4, unconditionally.
// BK (the K-tile depth) and VEC (float4 width) are NOT part of that choice —
// neither was ever varied by the tile-size experiments — so they stay plain
// file-scope constants.
constexpr int BK = TILE;   // K-dimension tile depth
constexpr int VEC = 4;     // vector width for float4 loads/stores

// Accumulate in FP64 by default: eliminates O(K·ε) rounding drift from FP32
// summation. Tiles in shared memory stay float32; only the running sums are
// promoted.
// GEMM_ACC_T also counts as "user explicitly set something" (below): it's
// not one of the tile-shape macros, but run_gemm_ab_bench.sh's gemm_accf
// binary (-DGEMM_ACC_T=float) is exactly the kind of isolated A/B this
// dispatch-bypass exists to protect -- if it were exempted, gemm_accf would
// silently ALSO switch tile config underneath it at the 4096^3 shape
// (auto-dispatch would pick the large tile there), conflating accumulator
// type and tile size into one measurement instead of isolating accumulator
// type the way that script's whole point is to do.
#ifdef GEMM_ACC_T
  #define GEMM_ACC_T_USER_SET 1
#else
  #define GEMM_ACC_T double
  #define GEMM_ACC_T_USER_SET 0
#endif

// GEMM_BM/GEMM_BN/GEMM_TM/GEMM_TN/GEMM_DOUBLE_BUFFER: kept for isolated A/B
// builds (run_gemm_ab_bench.sh-style — force ONE fixed tile config for the
// whole binary, e.g. -DGEMM_BN=32). Detect whether the caller passed any of
// them explicitly (vs. relying on the #ifndef fallback below) so the host
// code can tell "isolated A/B mode" apart from "dispatch-framework mode"
// (which, per GEMM_ENABLE_AUTO_DISPATCH further down, currently just means
// "always small tile" by default) — see GEMM_ANY_MACRO_SET and
// launch_gemm() further down. Passing none of them (the common case) goes
// through launch_gemm()'s dispatch-framework branch; passing any one of
// them (including GEMM_ACC_T, see above) forces the old fixed-single-kernel
// behavior, unchanged, so existing A/B scripts keep working exactly as
// before.
#ifdef GEMM_BM
  #define GEMM_BM_USER_SET 1
#else
  #define GEMM_BM 64
  #define GEMM_BM_USER_SET 0
#endif
#ifdef GEMM_BN
  #define GEMM_BN_USER_SET 1
#else
  #define GEMM_BN 64
  #define GEMM_BN_USER_SET 0
#endif
#ifdef GEMM_TM
  #define GEMM_TM_USER_SET 1
#else
  #define GEMM_TM 4
  #define GEMM_TM_USER_SET 0
#endif
#ifdef GEMM_TN
  #define GEMM_TN_USER_SET 1
#else
  #define GEMM_TN 4
  #define GEMM_TN_USER_SET 0
#endif
#ifdef GEMM_DOUBLE_BUFFER
  #define GEMM_DOUBLE_BUFFER_USER_SET 1
#else
  #define GEMM_DOUBLE_BUFFER 0
  #define GEMM_DOUBLE_BUFFER_USER_SET 0
#endif
#define GEMM_ANY_MACRO_SET \
    (GEMM_BM_USER_SET || GEMM_BN_USER_SET || GEMM_TM_USER_SET || \
     GEMM_TN_USER_SET || GEMM_DOUBLE_BUFFER_USER_SET || GEMM_ACC_T_USER_SET)

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
//   - Net: BM=BN=64, BK=16 (the "small" variant, and the permanent default
//     — see GEMM_ENABLE_AUTO_DISPATCH below) benchmarked best for mid-size
//     shapes (1024^3) on both platforms. Experiment A's BM=BN=128, TM=TN=8
//     ("large" variant) benchmarked ~14% faster at 4096^3 on the local
//     RTX 5090 but 1.6-3.2x SLOWER at 1024^3/small shapes there (grid too
//     thin at 170 SMs) — AND on real DCU hardware, the 4096^3 number itself
//     flipped sign: 12.4% SLOWER (43.141ms -> 48.487ms), not faster. Same
//     "5090 and DCU disagree on direction" pattern as the BN=64-vs-32 A/B
//     directly above. Net effect: the large-tile variant has no shape,
//     on either platform, where it's an unambiguous win — hence
//     GEMM_ENABLE_AUTO_DISPATCH defaulting off.
//   - Buildable with -DGEMM_BN=32 (isolated A/B mode) for the conflict-free
//     variant; DCU real-hardware results still pending.

// Per-(BM,BN,TM,TN) compile-time derived constants, shared by every helper
// below so NUM_THREADS/A_VEC_SLOTS/etc. are computed identically everywhere
// instead of being re-derived ad hoc per function.
template<int BM, int BN, int TM, int TN>
struct GemmTileCfg {
    static constexpr int NUM_THREADS        = (BM / TM) * (BN / TN);
    static constexpr int A_VEC_SLOTS        = (BM * BK) / VEC;
    static constexpr int B_VEC_SLOTS        = (BK * BN) / VEC;
    // Per-thread register-array size for the double-buffer prefetch path:
    // how many float4 slots the strided tid/tid+NUM_THREADS/... loop visits
    // per thread, worst case.
    static constexpr int A_SLOTS_PER_THREAD = (A_VEC_SLOTS + NUM_THREADS - 1) / NUM_THREADS;
    static constexpr int B_SLOTS_PER_THREAD = (B_VEC_SLOTS + NUM_THREADS - 1) / NUM_THREADS;
};

// ---------------------------------------------------------------------------
// Tile load/compute helpers, shared by both the single-buffer and
// double-buffer code paths in gemm_tiled() below.
// ---------------------------------------------------------------------------

// Load one [BM x BK] A-tile into sA[BK][BM] (transposed) and one [BK x BN]
// B-tile into sB[BK][BN]: vectorized float4 path when the tile is fully
// in-bounds/aligned, scalar masked path otherwise. Used for the initial
// load and for the (rare) non-vectorizable tile-transition fallback in the
// double-buffer path, and for every tile in the single-buffer path.
template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void load_A_tile(float sA[BK][BM],
                                             const float* __restrict__ A,
                                             int block_row, int k0, int tid,
                                             int M, int K, bool a_full)
{
    using Cfg = GemmTileCfg<BM, BN, TM, TN>;
    if (a_full) {
        for (int vidx = tid; vidx < Cfg::A_VEC_SLOTS; vidx += Cfg::NUM_THREADS) {
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
        for (int idx = tid; idx < BM * BK; idx += Cfg::NUM_THREADS) {
            const int m  = idx / BK;
            const int kk = idx % BK;
            const int g_row = block_row + m;
            const int g_col = k0 + kk;
            sA[kk][m] = (g_row < M && g_col < K) ? A[(size_t)g_row * K + g_col] : 0.0f;
        }
    }
}

template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void load_B_tile(float sB[BK][BN],
                                             const float* __restrict__ B,
                                             int block_col, int k0, int tid,
                                             int N, int K, bool b_full)
{
    using Cfg = GemmTileCfg<BM, BN, TM, TN>;
    if (b_full) {
        for (int vidx = tid; vidx < Cfg::B_VEC_SLOTS; vidx += Cfg::NUM_THREADS) {
            const int k      = vidx / (BN / VEC);
            const int ngroup = vidx % (BN / VEC);
            const int n0     = ngroup * VEC;
            *reinterpret_cast<float4*>(&sB[k][n0]) = *reinterpret_cast<const float4*>(
                &B[(size_t)(k0 + k) * N + (block_col + n0)]);
        }
    } else {
        for (int idx = tid; idx < BK * BN; idx += Cfg::NUM_THREADS) {
            const int k = idx / BN;
            const int n = idx % BN;
            const int g_row = k0 + k;
            const int g_col = block_col + n;
            sB[k][n] = (g_row < K && g_col < N) ? B[(size_t)g_row * N + g_col] : 0.0f;
        }
    }
}

// Consume one resident [BK][BM]/[BK][BN] tile pair: for each of the BK
// reduction steps, load this thread's TMxTN fragment (VEC-wide chunks) and
// FMA it into acc. Identical in both code paths — only which shared buffer
// is passed in differs (single buffer vs. the current ping-pong slot).
// acc is a pointer-to-array-of-TN parameter (array-to-pointer decay of the
// caller's GEMM_ACC_T[TM][TN]), not a reference-to-2D-array -- see the
// prefetch/store helpers below for why reference-to-template-sized-array
// parameters are avoided throughout this file.
template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void compute_tile(GEMM_ACC_T acc[][TN],
                                              float sA[BK][BM], float sB[BK][BN])
{
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        float a_frag[TM], b_frag[TN];
        #pragma unroll
        for (int iv = 0; iv < TM / VEC; ++iv)
            *reinterpret_cast<float4*>(&a_frag[iv * VEC]) =
                *reinterpret_cast<float4*>(&sA[kk][threadIdx.y * TM + iv * VEC]);
        #pragma unroll
        for (int jv = 0; jv < TN / VEC; ++jv)
            *reinterpret_cast<float4*>(&b_frag[jv * VEC]) =
                *reinterpret_cast<float4*>(&sB[kk][threadIdx.x * TN + jv * VEC]);

        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] += (GEMM_ACC_T)a_frag[i] * (GEMM_ACC_T)b_frag[j];
    }
}

// Prefetch-to-registers versions of the vectorized load paths above: read
// from global memory into a per-thread register array WITHOUT touching
// shared memory yet, so the (high-latency) global load is issued while the
// caller goes on to run compute_tile() on the tile already in shared memory
// — the store_*_prefetch() calls write the fetched data into the *other*
// ping-pong buffer only after that compute finishes. Only ever called when
// the tile is confirmed fully vectorizable (see gemm_tiled()); the masked
// tail case reuses load_A_tile/load_B_tile directly instead, same as the
// single-buffer path. Register prefetch (not cp.async) is used because
// gfx906 has no async global->shared copy path, so this has to work with
// plain loads on both nvcc and hipcc.
//
// reg is a plain pointer, not a reference-to-template-sized-array
// (float4 (&reg)[Cfg::A_SLOTS_PER_THREAD]) -- nvcc accepts that form, but
// real-hardware hipcc/clang-14 (DTK 22.10.1) fails to parse a reference
// parameter whose array bound is a dependent nested-type member access
// (GemmTileCfg<...>::A_SLOTS_PER_THREAD), erroring on the reg parameter
// itself ("variable has incomplete type 'void'" / "use of undeclared
// identifier 'reg'"). A raw pointer parameter has identical codegen here
// (nothing in these functions relied on sizeof(reg) or the type-encoded
// bound; the loop bound is always Cfg::A_SLOTS_PER_THREAD, independently)
// and the caller's fixed-size array argument decays to it automatically --
// the only thing lost is the compiler catching an accidental
// wrong-size-array-passed-in at the call site, which nothing here does.
template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void prefetch_A_tile(
    float4* __restrict__ reg,
    const float* __restrict__ A, int block_row, int k0, int tid, int K)
{
    using Cfg = GemmTileCfg<BM, BN, TM, TN>;
    #pragma unroll
    for (int i = 0; i < Cfg::A_SLOTS_PER_THREAD; ++i) {
        const int vidx = tid + i * Cfg::NUM_THREADS;
        if (vidx < Cfg::A_VEC_SLOTS) {
            const int m      = vidx / (BK / VEC);
            const int kgroup = vidx % (BK / VEC);
            const int kk0    = kgroup * VEC;
            reg[i] = *reinterpret_cast<const float4*>(
                &A[(size_t)(block_row + m) * K + (k0 + kk0)]);
        }
    }
}

template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void store_A_prefetch(
    float sA[BK][BM],
    float4* __restrict__ reg,
    int tid)
{
    using Cfg = GemmTileCfg<BM, BN, TM, TN>;
    #pragma unroll
    for (int i = 0; i < Cfg::A_SLOTS_PER_THREAD; ++i) {
        const int vidx = tid + i * Cfg::NUM_THREADS;
        if (vidx < Cfg::A_VEC_SLOTS) {
            const int m      = vidx / (BK / VEC);
            const int kgroup = vidx % (BK / VEC);
            const int kk0    = kgroup * VEC;
            sA[kk0 + 0][m] = reg[i].x;
            sA[kk0 + 1][m] = reg[i].y;
            sA[kk0 + 2][m] = reg[i].z;
            sA[kk0 + 3][m] = reg[i].w;
        }
    }
}

template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void prefetch_B_tile(
    float4* __restrict__ reg,
    const float* __restrict__ B, int block_col, int k0, int tid, int N)
{
    using Cfg = GemmTileCfg<BM, BN, TM, TN>;
    #pragma unroll
    for (int i = 0; i < Cfg::B_SLOTS_PER_THREAD; ++i) {
        const int vidx = tid + i * Cfg::NUM_THREADS;
        if (vidx < Cfg::B_VEC_SLOTS) {
            const int k      = vidx / (BN / VEC);
            const int ngroup = vidx % (BN / VEC);
            const int n0     = ngroup * VEC;
            reg[i] = *reinterpret_cast<const float4*>(
                &B[(size_t)(k0 + k) * N + (block_col + n0)]);
        }
    }
}

template<int BM, int BN, int TM, int TN>
__device__ __forceinline__ void store_B_prefetch(
    float sB[BK][BN],
    float4* __restrict__ reg,
    int tid)
{
    using Cfg = GemmTileCfg<BM, BN, TM, TN>;
    #pragma unroll
    for (int i = 0; i < Cfg::B_SLOTS_PER_THREAD; ++i) {
        const int vidx = tid + i * Cfg::NUM_THREADS;
        if (vidx < Cfg::B_VEC_SLOTS) {
            const int k      = vidx / (BN / VEC);
            const int ngroup = vidx % (BN / VEC);
            const int n0     = ngroup * VEC;
            *reinterpret_cast<float4*>(&sB[k][n0]) = reg[i];
        }
    }
}

// BM/BN/TM/TN/DoubleBuffer are now template parameters, not fixed file-scope
// constants: gemm_tiled<64,64,4,4,false> is the original default kernel,
// bit-for-bit; gemm_tiled<128,128,8,8,false> is Experiment A's large tile;
// other instantiations (e.g. via GEMM_ANY_MACRO_SET's isolated A/B mode)
// are compiled on demand from whatever GEMM_BM/BN/TM/TN/GEMM_DOUBLE_BUFFER
// were passed. See launch_gemm() for which instantiation actually gets
// called for a given problem size.
template<int BM, int BN, int TM, int TN, bool DoubleBuffer>
__global__ void gemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int K, int N)
{
    // float4 fragment loads/stores above need TM/TN to split evenly into
    // VEC-wide chunks -- true for every config used so far (VEC=4, TM/TN in
    // {4,8}) but not guaranteed for an arbitrary override, so make the
    // requirement explicit instead of failing silently on a misaligned read.
    static_assert(TM % VEC == 0 && TN % VEC == 0,
                  "GEMM_TM/GEMM_TN must be multiples of VEC(4) for vectorized fragment loads");

    // sA is stored TRANSPOSED as [k][m]: for a fixed k, the TM rows a thread
    // needs are contiguous in shared memory, making the per-k fragment load
    // below VEC-wide float4 reads instead of scalar ones.
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.y * (BN / TN) + threadIdx.x;   // 0..NUM_THREADS-1

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

    if (DoubleBuffer) {
        // Ping-pong the two shared-memory tiles: while compute_tile()
        // consumes sA[cur]/sB[cur], the next tile's global load is already
        // in flight (prefetched to registers before the compute call,
        // written into sA[nxt]/sB[nxt] after) instead of being fully
        // load->sync'd before any compute can start, the way the
        // single-buffer path below does.
        __shared__ __align__(16) float sA[2][BK][BM];
        __shared__ __align__(16) float sB[2][BK][BN];

        int cur = 0;
        {
            const bool a_full0 = k_aligned && (block_row + BM - 1 < M) && (BK - 1 < K);
            const bool b_full0 = n_aligned && (BK - 1 < K) && (block_col + BN - 1 < N);
            load_A_tile<BM, BN, TM, TN>(sA[cur], A, block_row, 0, tid, M, K, a_full0);
            load_B_tile<BM, BN, TM, TN>(sB[cur], B, block_col, 0, tid, N, K, b_full0);
        }
        __syncthreads();

        for (int t = 0; t < num_tiles; ++t) {
            const bool has_next = (t + 1 < num_tiles);
            const int nxt = cur ^ 1;

            float4 a_pref[GemmTileCfg<BM, BN, TM, TN>::A_SLOTS_PER_THREAD];
            float4 b_pref[GemmTileCfg<BM, BN, TM, TN>::B_SLOTS_PER_THREAD];
            bool do_prefetch = false, a_full_next = false, b_full_next = false;
            int k0_next = 0;
            if (has_next) {
                k0_next = (t + 1) * BK;
                a_full_next = k_aligned && (block_row + BM - 1 < M) && (k0_next + BK - 1 < K);
                b_full_next = n_aligned && (k0_next + BK - 1 < K) && (block_col + BN - 1 < N);
                // Only pipeline the fully-vectorizable case; the (rare)
                // masked tail tile falls back to a synchronous load below,
                // same as the single-buffer path -- keeps the prefetch
                // register footprint small and reuses already-correct
                // masked-path logic verbatim.
                do_prefetch = a_full_next && b_full_next;
                if (do_prefetch) {
                    prefetch_A_tile<BM, BN, TM, TN>(a_pref, A, block_row, k0_next, tid, K);
                    prefetch_B_tile<BM, BN, TM, TN>(b_pref, B, block_col, k0_next, tid, N);
                }
            }

            compute_tile<BM, BN, TM, TN>(acc, sA[cur], sB[cur]);

            if (has_next) {
                if (do_prefetch) {
                    store_A_prefetch<BM, BN, TM, TN>(sA[nxt], a_pref, tid);
                    store_B_prefetch<BM, BN, TM, TN>(sB[nxt], b_pref, tid);
                } else {
                    load_A_tile<BM, BN, TM, TN>(sA[nxt], A, block_row, k0_next, tid, M, K, a_full_next);
                    load_B_tile<BM, BN, TM, TN>(sB[nxt], B, block_col, k0_next, tid, N, K, b_full_next);
                }
                __syncthreads();
            }
            cur = nxt;
        }
    } else {
        __shared__ __align__(16) float sA[BK][BM];
        __shared__ __align__(16) float sB[BK][BN];

        for (int t = 0; t < num_tiles; ++t) {
            const int k0 = t * BK;
            const bool a_full = k_aligned && (block_row + BM - 1 < M) && (k0 + BK - 1 < K);
            const bool b_full = n_aligned && (k0 + BK - 1 < K) && (block_col + BN - 1 < N);
            load_A_tile<BM, BN, TM, TN>(sA, A, block_row, k0, tid, M, K, a_full);
            load_B_tile<BM, BN, TM, TN>(sB, B, block_col, k0, tid, N, K, b_full);

            __syncthreads();
            compute_tile<BM, BN, TM, TN>(acc, sA, sB);
            __syncthreads();
        }
    }

    const int row_base = block_row + threadIdx.y * TM;
    const int col_base = block_col + threadIdx.x * TN;
    const bool store_full = n_aligned && (col_base + TN - 1 < N);

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = row_base + i;
        if (row >= M) continue;
        if (store_full) {
            #pragma unroll
            for (int jv = 0; jv < TN / VEC; ++jv) {
                float4 out = make_float4((float)acc[i][jv * VEC + 0], (float)acc[i][jv * VEC + 1],
                                          (float)acc[i][jv * VEC + 2], (float)acc[i][jv * VEC + 3]);
                *reinterpret_cast<float4*>(&C[(size_t)row * N + col_base + jv * VEC]) = out;
            }
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
// Runtime tile-variant dispatch FRAMEWORK (only reachable when
// GEMM_ANY_MACRO_SET is false — no GEMM_BM/BN/TM/TN/GEMM_DOUBLE_BUFFER macro
// was passed at compile time). When also enabled via GEMM_ENABLE_AUTO_
// DISPATCH (see below — off by default), picks between the two compiled-in
// variants, gemm_tiled<64,64,4,4,false> ("small", the original default) and
// gemm_tiled<128,128,8,8,false> ("large", Experiment A), based on how much
// parallelism the large tile's grid would actually have for this M/N —
// never on a hardcoded shape. With GEMM_ENABLE_AUTO_DISPATCH left off (the
// default), use_large_tile() always returns false and this threshold logic
// is compiled but never actually evaluated at runtime — see the rationale
// by GEMM_ENABLE_AUTO_DISPATCH's definition below.
//
// Threshold derivation: the 128-tile kernel's grid is
// ceil(M/128)*ceil(N/128) blocks. That block count needs enough margin over
// the SM/CU count on BOTH target platforms — the local dev GPU (RTX 5090,
// 170 SMs) and the deployment target (DCU gfx906, 64 CUs) — to actually
// keep the GPU busy; a grid barely bigger than the SM count leaves most SMs
// idle once the first (and only) wave of blocks starts finishing at
// slightly different times. Dividing by the LARGER SM count (170) is the
// conservative choice: whatever margin that computes is a LOWER BOUND on
// the margin the same block count would give on the 64-CU DCU (fewer CUs
// to fill => the same block count divides into a bigger number there), so
// "margin computed against 170 is enough" already implies "margin against
// 64 is also enough" — one number covers both platforms.
//
// The multiple itself: .claude/skills/dcu_perf.md #5 calls ~1.5x grid/SM
// "marginal/insufficient" and 6x-24x "comfortable" headroom (measured, post
// hoc, for kernels that performed well). 4x sits clearly above the
// documented-bad 1.5x and conservatively below the documented-good 6x-24x
// range — chosen so multiple blocks can still be resident per SM (hiding
// memory/sync latency across blocks) without requiring the full
// comfortable-range margin those particular measured kernels happened to
// have.
constexpr int    DISPATCH_SM_COUNT   = 170;   // larger of {RTX5090:170, gfx906:64}
constexpr double DISPATCH_MIN_MARGIN = 4.0;   // see derivation above

// Auto-dispatch was validated only on the local RTX 5090 dev box before
// this macro was added -- real DCU (gfx906) hardware data then showed the
// OPPOSITE result: the 128-tile variant that measured 12.4% FASTER than
// the small tile on 5090 at 4096^3 measured 12.4% SLOWER on DCU
// (43.141ms -> 48.487ms). Same "5090 and DCU disagree on direction"
// pattern already seen once for the BN=64-vs-32 bank-conflict A/B
// (.claude/skills/dcu_perf.md #8) -- not a fluke, apparently a recurring
// risk of tuning on the dev GPU and assuming it transfers to the
// deployment target. DCU is the actual deployment target, so this is a
// real regression there, and auto-dispatch defaults OFF: use_large_tile()
// always returns false unless GEMM_ENABLE_AUTO_DISPATCH is explicitly
// defined, which keeps the default (no-macros) build permanently on the
// small-tile kernel -- the one actually validated on real DCU hardware
// (dcu_perf.md §7/§8, plus the boundary/non2n/large tiers of the
// dispatch-verify submission). The dispatch machinery itself (the
// gemm_tiled<128,128,8,8,false> instantiation, the grid-margin threshold
// logic above) is kept, not deleted, so a future DCU-specific tile-tuning
// study (register-pressure/occupancy analysis on gfx906, not just
// re-running the 5090-derived config) can build on this dispatch
// framework instead of starting over -- GEMM_ENABLE_AUTO_DISPATCH is
// exactly the switch that study would flip on once it has its own
// DCU-validated threshold/config to dispatch to.
#ifndef GEMM_ENABLE_AUTO_DISPATCH
#define GEMM_ENABLE_AUTO_DISPATCH 0
#endif

static bool use_large_tile(int M, int N)
{
#if GEMM_ENABLE_AUTO_DISPATCH
    const long long large_blocks =
        (long long)((M + 127) / 128) * (long long)((N + 127) / 128);
    return (double)large_blocks / DISPATCH_SM_COUNT >= DISPATCH_MIN_MARGIN;
#else
    (void)M; (void)N;
    return false;
#endif
}

// Single launch site used by both run_benchmark() and main(): in
// auto-dispatch mode, picks small vs. large tile per use_large_tile(); in
// isolated-A/B mode (GEMM_ANY_MACRO_SET), always launches the one
// GEMM_BM/BN/TM/TN/GEMM_DOUBLE_BUFFER-configured variant, unchanged from
// before this refactor.
static void launch_gemm(const float* dA, const float* dB, float* dC, int M, int K, int N)
{
#if GEMM_ANY_MACRO_SET
    dim3 block(GEMM_BN / GEMM_TN, GEMM_BM / GEMM_TM);
    dim3 grid((N + GEMM_BN - 1) / GEMM_BN, (M + GEMM_BM - 1) / GEMM_BM);
    gemm_tiled<GEMM_BM, GEMM_BN, GEMM_TM, GEMM_TN, (bool)GEMM_DOUBLE_BUFFER>
        <<<grid, block>>>(dA, dB, dC, M, K, N);
#else
    if (use_large_tile(M, N)) {
        dim3 block(128 / 8, 128 / 8);
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        gemm_tiled<128, 128, 8, 8, false><<<grid, block>>>(dA, dB, dC, M, K, N);
    } else {
        dim3 block(64 / 4, 64 / 4);
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        gemm_tiled<64, 64, 4, 4, false><<<grid, block>>>(dA, dB, dC, M, K, N);
    }
#endif
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

    for (int i = 0; i < warmup; ++i) {
        launch_gemm(dA, dB, dC, M, K, N);
        check(cudaGetLastError(), "warmup kernel launch");
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    float* times = (float*)malloc(iters * sizeof(float));
    for (int i = 0; i < iters; ++i) {
        check(cudaEventRecord(start), "record start");
        launch_gemm(dA, dB, dC, M, K, N);
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

    launch_gemm(dA, dB, dC, M, K, N);
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
