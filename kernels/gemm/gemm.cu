// gemm.cu — FP32 GEMM with shared-memory tiling
// A[M,K] * B[K,N] = C[M,N]
// Compile (CUDA):  nvcc -O2 -o gemm gemm.cu
// Compile (DCU):   hipcc -O2 -o gemm gemm.cu
// Usage:           ./gemm <M> <K> <N> <file_A> <file_B> [file_C]
//   file_A : raw binary, M*K float32, row-major
//   file_B : raw binary, K*N float32, row-major
//   file_C : (optional) raw binary output, M*N float32, row-major

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __HIP_PLATFORM_HCC__
  constexpr int WARP_SIZE = 64;
#else
  constexpr int WARP_SIZE = 32;
#endif

constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void gemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Accumulate in FP64: eliminates O(K·ε) rounding drift from FP32 summation.
    // Tiles in shared memory stay float32; only the running sum is promoted.
    double acc = 0.0;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += (double)sA[threadIdx.y][k] * (double)sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = (float)acc;
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
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
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
