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

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Platform portability
// ---------------------------------------------------------------------------
#ifdef __HIP_PLATFORM_HCC__
  constexpr int WARP_SIZE = 64;   // DCU (gfx906)
#else
  constexpr int WARP_SIZE = 32;   // NVIDIA CUDA
#endif

constexpr int BLOCK_SIZE   = 64;   // Br = Bc; must be a power of 2
constexpr int MAX_HEAD_DIM = 128;  // runtime head_dim must satisfy d <= MAX_HEAD_DIM

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void flash_attn_kernel(
    const float* __restrict__ Q,    // [seq_len, d]
    const float* __restrict__ K,    // [seq_len, d]
    const float* __restrict__ V,    // [seq_len, d]
    float*       __restrict__ O,    // [seq_len, d]
    int seq_len, int d, float scale)
{
    // Shared memory layout: [sK | sV], each BLOCK_SIZE * d floats
    extern __shared__ float smem[];
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
        // Access pattern: consecutive threads → consecutive columns → coalesced.
        for (int elem = tid; elem < BLOCK_SIZE * d; elem += BLOCK_SIZE) {
            const int row = elem / d, col = elem % d;
            const int grow = kv_start + row;
            sK[elem] = (grow < seq_len) ? K[grow * d + col] : 0.0f;
            sV[elem] = (grow < seq_len) ? V[grow * d + col] : 0.0f;
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
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
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
    size_t smem_bytes = 2 * BLOCK_SIZE * head_dim * sizeof(float);

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
