# DCU 加速卡深度学习算子智能生成与验证流水线

基于 [Claude Code](https://claude.com/claude-code) 构建的深度学习底层算子（C++）智能开发系统，目标平台为 DCU（海光 Hygon 数据中心加速卡）。系统利用大模型编程智能体自动生成、编译、验证并优化 HIP C++ 算子代码。

## 项目简介

DCU 采用 ROCm/HIP 生态，与 CUDA 编程模型相近但存在关键差异（`warpSize=64`、LDS 内存模型、`hipcc` 编译器等）。本项目探索利用编程智能体自动化算子开发流程。

由于 HIP 的跨平台特性，项目在 **NVIDIA RTX 5090** 平台上完成开发与功能验证，代码通过条件编译保证可迁移至 **DCU（gfx906）**。HIP 代码可同时编译运行于 NVIDIA（CUDA 后端）与 DCU（ROCm 后端），二者核心差异仅在 `warpSize`（32 vs 64）与 LDS 容量，已通过宏定义统一处理。

## 当前进度

三个核心算子均已成功生成、编译、在 GPU 上运行，并通过 PyTorch 基线的数值精度验证：

| 算子 | 实现要点 | 编译 | 运行 | 精度验证 | 最大绝对误差 |
|------|----------|------|------|----------|--------------|
| GEMM | 16×16 shared memory 分块 | ✅ | ✅ | PASS | 0.00e+00 |
| Softmax | 三阶段 online softmax | ✅ | ✅ | PASS | 9.31e-10 |
| Flash Attention | 分块 + online softmax | ✅ | ✅ | PASS | 6.56e-07 |

> GEMM 与 Softmax 采用 FP32 容差 `atol=1e-5`；Flash Attention 因分块改变浮点累加顺序，采用 `atol=1e-3`。

## 目录结构

```
dcu-llm-kernels/
├── CLAUDE.md                          # 智能体约束文件（项目方法论核心）
├── kernels/
│   ├── gemm/
│   │   ├── gemm.cu                    # GEMM 算子 CUDA C++ 实现
│   │   └── validate_gemm.py           # PyTorch 基线验证脚本
│   ├── softmax/
│   │   ├── softmax.cu
│   │   └── validate_softmax.py
│   └── flash_attention/
│       ├── flash_attn.cu
│       └── validate_flash_attn.py
└── tests/
    └── stress_test.py                 # 10 用例自动化压力测试框架
```

## 环境要求

| 组件 | 版本 / 说明 |
|------|-------------|
| CUDA Toolkit | 13.3 |
| C++ 编译器 | Visual Studio 2022 BuildTools (MSVC 14.44) |
| PyTorch | 2.11.0 + cu128 |
| GPU | NVIDIA RTX 5090（Blackwell, sm_120） |
| 最终目标平台 | DCU (Hygon gfx906)，通过 HIP 条件编译迁移 |

## 快速开始

以 GEMM 为例：

```bash
# 1. 编译
nvcc -O2 -arch=sm_120 -o kernels/gemm/gemm kernels/gemm/gemm.cu

# 2. 精度验证（自动用 PyTorch 计算基线并逐元素对比）
python kernels/gemm/validate_gemm.py 64 128 64

# 3. 运行全部压力测试
python tests/stress_test.py
```

Softmax 与 Flash Attention 同理：

```bash
nvcc -O2 -arch=sm_120 -o kernels/softmax/softmax kernels/softmax/softmax.cu
python kernels/softmax/validate_softmax.py 128 512

nvcc -O2 -arch=sm_120 -o kernels/flash_attention/flash_attn kernels/flash_attention/flash_attn.cu
python kernels/flash_attention/validate_flash_attn.py 128 64
```

验证脚本会自动用 PyTorch 计算基线，与算子输出逐元素对比，并打印 `PASS/FAIL` 及最大绝对误差。

## 技术路线

项目采用三阶段递进式路线：

1. **阶段一（生成与验证）** — 用智能体直接生成三类算子的 C++ 实现，编译运行后与 PyTorch 库结果做精度对比。**（已完成）**
2. **阶段二（优化）** — 通过 Skills 机制注入 DCU 领域知识，提升生成代码的精度与性能。**（进行中）**
3. **阶段三（缺陷分析与重构）** — 系统测试各类编程场景的缺陷，重构验证 harness 为自动化测试框架。**（待开展）**

## 已知问题

压力测试（10 用例，7 PASS / 3 FAIL）定位出 GEMM 算子在大 K 维度下的精度劣化：误差随 K 单调增大（K=200 时 9.16e-05，K=4096 时 5.07e-03）。根因为 FP32 累加精度损失，与 PyTorch 不同累加顺序产生偏差，并非算法逻辑错误。该问题已列为第二阶段优化目标（可通过更高精度累加或 Kahan 补偿求和解决）。

## 关于 DCU 硬件

当前在 NVIDIA 平台完成的工作约占项目整体 75%。剩余约 25% 为 DCU 硬件专属内容，包括 `hipcc` 针对 gfx906 的编译、`rocprof` 性能剖析、`warpSize=64` 的实机行为验证。由于代码已通过 HIP 条件编译保证可移植性，这部分工作在获得 DCU 机器后可快速补全。
