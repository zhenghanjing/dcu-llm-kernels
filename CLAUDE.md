# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dcu-llm-kernels` is a collection of high-performance compute kernels for LLM workloads targeting DCU (Hygon Data Center Unit) accelerators. Kernels are written in HIP C++ using the DCU Toolkit (DTK), which is ROCm-compatible.

## Repository Layout

```
kernels/
  gemm/           # General Matrix Multiply kernels (core of linear layers)
  softmax/        # Softmax kernels (attention score normalization)
  flash_attention/ # Fused Flash Attention kernels (memory-efficient attention)
tests/            # Correctness tests comparing kernel output to reference
validate/         # Numerical accuracy and performance benchmarks
```

## Toolchain

- Compiler: `hipcc` (from DTK/ROCm) for DCU; `nvcc` (CUDA 12.8) for current development
- Current dev hardware: NVIDIA RTX 5090 (CUDA 12.8, `nvcc`)
- Final target hardware: DCU (Hygon gfx906)
- Build system: to be established (CMake or Makefile expected)

## Platform Portability

`warpSize` differs between platforms and **must always be handled via conditional compilation**:

```cpp
#ifdef __HIP_PLATFORM_HCC__
  constexpr int WARP_SIZE = 64;  // DCU (gfx906)
#else
  constexpr int WARP_SIZE = 32;  // NVIDIA CUDA
#endif
```

Never hard-code `32` or `64` for warp-level operations. Use `WARP_SIZE` throughout.

## Kernel Development Conventions

Each kernel directory under `kernels/` should contain:
- The HIP/CUDA kernel implementation (`.cpp` / `.hip` / `.cu`)
- A kernel header (`.h`)
- A `CMakeLists.txt` or `Makefile` for that kernel
- A Python validation script (see below)

Tests in `tests/` validate correctness against a CPU or cuBLAS/rocBLAS reference. `validate/` is for performance benchmarking and numerical precision analysis.

## Validation Standards

Every operator must ship with a Python validation script that uses PyTorch as the reference baseline.

Tolerance thresholds:

| Precision | `atol` |
|-----------|--------|
| FP32      | 1e-5   |
| FP16      | 1e-2   |

Validation scripts live alongside the kernel (e.g., `kernels/gemm/validate_gemm.py`) and must:
1. Run the custom kernel output against `torch` reference ops.
2. Assert `torch.allclose(out, ref, atol=ATOL)` with the appropriate threshold.
3. Print a clear PASS / FAIL result.
