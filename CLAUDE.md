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

## Performance Benchmarking Standard (SOTA comparison)

Project goal (as of 2026-08-09): DCU kernels should aim to match or exceed
the performance of vendor/framework SOTA implementations — but "SOTA" and
"match/exceed" must be reported on two separate tracks, never conflated:

1. **Absolute cross-hardware numbers** (TFLOPS / latency), directly
   comparing DCU against RTX 5090. RTX 5090 FP32 peak is ~104.8 TFLOPS;
   the real DCU card behind this project's portal (gfx906, `Device 66a1`,
   SKU D160A1) has a measured FP32 peak of **~13.9 TFLOPS**
   (64 CU × 64 FP32 ALU/CU × 2 FLOP/cycle × 1.7 GHz max clock, from real
   `rocminfo` output — see `docs/dcu_portability_review.md`). That's roughly
   a 7.5x hardware gap, so absolute-number parity/superiority is generally
   **not achievable** regardless of kernel quality — do not treat it as a
   realistic target without flagging the hardware ceiling.
2. **Hardware utilization efficiency** (achieved TFLOPS ÷ own platform's
   theoretical peak). This is the fair, achievable target: a DCU kernel
   should be judged against what DCU's *own* best available implementation
   can do on the same silicon.

**SOTA reference baselines** (confirmed on real hardware, DTK 22.10.1):
- **5090 side**: `torch.mm` with `torch.backends.cuda.matmul.allow_tf32 =
  False` set explicitly (do not rely on the default — it may silently route
  through the TF32 tensor-core path, which is a different, lower-precision
  hardware path than FP32 CUDA cores and would invalidate the peak-%
  comparison).
- **DCU side**: `rocBLAS` (`rocblas_sgemm`) — confirmed present, linkable,
  and runnable on the real machine (`/opt/rocm/include/rocblas.h`,
  `-lrocblas`). `hipBLAS` is **not** a separate implementation: per ROCm
  docs it's a marshalling layer that dispatches to rocBLAS on this
  platform, so benchmarking it separately is redundant. `MIOpenGEMM`
  (`libmiopengemm.so`) exists on this machine but is a legacy OpenCL-based
  autotuning library pinned to ROCm 2.9 (2019) with no simple callable
  GEMM entry point (requires building `Geometry`/`HyPas`/`Constraints`
  objects and running a `find1()` autotuning search) — not used as a SOTA
  baseline.

Any new performance-comparison work should report both tracks explicitly
and cite which one a given percentage/number refers to.
