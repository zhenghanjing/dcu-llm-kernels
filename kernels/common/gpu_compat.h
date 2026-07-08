// kernels/common/gpu_compat.h — CUDA/HIP symbol-compatibility shim.
//
// Verified on real hardware (dtk-22.10.1, gfx906 — see test_min/test_min.cu
// and docs/dcu_portability_review.md, "最高优先级发现" / "建议 1"):
//   1. hipcc does NOT implicitly inject any header for .cu files the way
//      nvcc does — <hip/hip_runtime.h> must be included explicitly, or
//      nothing (not even float4) resolves.
//   2. Even with that include, hipcc does NOT map cuda*-named symbols to
//      their hip* equivalents — cudaError_t/cudaMalloc/cudaSuccess/
//      cudaFree/cudaGetErrorString were all rejected as "undeclared" on
//      real hardware, with the compiler suggesting the hip*-prefixed name
//      instead.
//   3. float4 needs no mapping — it's a native HIP vector type too, and
//      resolves fine as soon as hip_runtime.h is included.
//
// This header covers every cuda* symbol actually referenced by
// kernels/gemm/gemm.cu, kernels/softmax/softmax.cu, and
// kernels/flash_attention/flash_attn.cu (found by grepping those three
// files, not copied from a hypothetical/complete CUDA API list). Only the
// first 5 entries below (cudaError_t/cudaMalloc/cudaSuccess/cudaFree/
// cudaGetErrorString) have been individually confirmed to work via
// test_min.cu on real hardware; the rest follow the exact same confirmed
// pattern (hip* is the literal cuda*->hip* rename) but have NOT been
// individually re-verified yet — that's what running stress_test.py /
// benchmark.py on DCU after this header lands is for.
//
// Usage: #include this at the top of each kernel .cu file, before any
// cuda*/hip* symbol is used.

#pragma once

// Route on __HIPCC__, not __HIP_PLATFORM_HCC__/__HIP_PLATFORM_AMD__.
// Real-hardware root-cause finding (dtk-22.10.1): those two platform macros
// are defined BY hip_runtime.h itself, not by the hipcc compiler driver up
// front — using them to decide whether to include hip_runtime.h is a
// chicken-and-egg check that can never trigger. test_min.cu only worked
// because it included hip_runtime.h unconditionally, with no #ifdef gating
// it at all. __HIPCC__ is the compiler-driver-level macro (parallel to
// nvcc's __CUDACC__): hipcc predefines it before any header is processed,
// regardless of backend, so it has no such ordering problem.
#ifdef __HIPCC__
  #include <hip/hip_runtime.h>

  // --- confirmed via test_min.cu on real hardware ---
  #define cudaError_t                                  hipError_t
  #define cudaSuccess                                  hipSuccess
  #define cudaGetErrorString                            hipGetErrorString
  #define cudaMalloc                                    hipMalloc
  #define cudaFree                                      hipFree

  // --- same pattern, not yet individually re-verified on real hardware ---
  #define cudaGetLastError                              hipGetLastError
  #define cudaMemcpy                                     hipMemcpy
  #define cudaMemset                                     hipMemset
  #define cudaMemcpyHostToDevice                         hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost                         hipMemcpyDeviceToHost
  #define cudaDeviceSynchronize                          hipDeviceSynchronize
  #define cudaEvent_t                                    hipEvent_t
  #define cudaEventCreate                                hipEventCreate
  #define cudaEventRecord                                hipEventRecord
  #define cudaEventSynchronize                           hipEventSynchronize
  #define cudaEventElapsedTime                           hipEventElapsedTime
  #define cudaEventDestroy                               hipEventDestroy
  #define cudaFuncSetAttribute                           hipFuncSetAttribute
  #define cudaFuncAttributeMaxDynamicSharedMemorySize    hipFuncAttributeMaxDynamicSharedMemorySize
#else
  #include <cuda_runtime.h>
#endif
