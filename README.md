# DCU 加速卡深度学习算子智能生成与验证流水线

基于 [Claude Code](https://claude.com/claude-code) 构建的深度学习底层算子（C++）智能开发系统，目标平台为 DCU（海光 Hygon 数据中心加速卡）。系统利用大模型编程智能体自动生成、编译、验证并优化 HIP C++ 算子代码。

## 项目简介

DCU 采用 ROCm/HIP 生态，与 CUDA 编程模型相近但存在关键差异（`warpSize=64`、LDS 内存模型、`hipcc` 编译器等）。本项目探索利用编程智能体自动化算子开发流程。

由于 HIP 的跨平台特性，项目在 **NVIDIA RTX 5090** 平台上完成开发与功能验证，代码通过条件编译保证可迁移至 **DCU（gfx906）**。HIP 代码可同时编译运行于 NVIDIA（CUDA 后端）与 DCU（ROCm 后端），二者核心差异仅在 `warpSize`（32 vs 64）与 LDS 容量，已通过宏定义统一处理。

## 当前进度

三个核心算子均已成功生成、编译、在 GPU 上运行，并通过 PyTorch 基线的数值精度验证（NVIDIA RTX 5090 / CUDA）：

| 算子 | 实现要点 | 编译 | 运行 | 精度验证 | 最大绝对误差 |
|------|----------|------|------|----------|--------------|
| GEMM | 16×16 shared memory 分块 | ✅ | ✅ | PASS | 0.00e+00 |
| Softmax | 三阶段 online softmax | ✅ | ✅ | PASS | 9.31e-10 |
| Flash Attention | 分块 + online softmax | ✅ | ✅ | PASS | 6.56e-07 |

> GEMM 与 Softmax 采用 FP32 容差 `atol=1e-5`；Flash Attention 因分块改变浮点累加顺序，采用 `atol=1e-3`。

三个算子均已进一步在海光 **DCU 真机**（gfx906/gfx926，DTK 22.10.1，`hipcc` 编译）验证编译与运行正确性：

| 算子 | 真机编译 | 真机运行 | 输出对比本地 CUDA |
|------|----------|----------|--------------------|
| GEMM | ✅ | ✅ exit 0 | **bit-for-bit 一致** |
| Softmax | ✅ | ✅ exit 0 | 逐位一致 |
| Flash Attention | ✅（含 smem_risk 用例，64KiB 动态共享内存，触发 `hipFuncSetAttribute`） | ✅ exit 0，两用例均无 launch failure | 一致（1e-6 量级浮点舍入内） |

真机验证过程中发现并修复了两个真实的可移植性问题（而非"直接跑通"）：`cuda*` 符号在 hipcc 下不会隐式可用（需要显式 include + 手写符号映射），以及 `cudaFuncSetAttribute`/`hipFuncSetAttribute` 的核函数指针参数在 hipcc 的 Clang 前端下不能隐式转换为 `const void*`（需要显式 cast，nvcc/MSVC 此前放行了这个非标准写法）。详见 `docs/dcu_portability_review.md`。

新增的 `kernels/gemm/gemm_large_tile.cu`（GEMM 的动态共享内存变体，运行时可选 tile 配置，最大档需要 64KiB 动态共享内存）进一步在真机验证了上述两个可移植性修复方案能否被复用到一个全新写的 kernel：**首次 `hipcc` 编译即一次性成功**，4 个 tile 配置 × 3 个 shape 共 12 组测试全部 bit-for-bit 通过，证明 `.claude/skills/dcu_hip_porting.md` 里沉淀的经验具备跨 kernel 复用能力，而不只对 `flash_attn.cu` 这一个文件有效。详见 `docs/dcu_portability_review.md`"真机验证记录"一节。

真机这一轮只验证了**正确性**，尚未产出 DCU 版本的性能数字，也还没跑完整的 `stress_test.py` 压力测试套件（目前是手工验证单个/两个 shape）；bank conflict、`double` 累加器开销等 `dcu_perf.md` 里基于 RTX 5090 标定的性能结论也都还没有在真机上重新验证。

**后续更新**：bank conflict、`double` 累加器开销均已在真机上补齐隔离 A/B 实测（`.claude/skills/dcu_perf.md` 第 8 节），结论都是保留当前默认实现。

## 性能对标 SOTA：GEMM 双口径基准（新增）

项目目标进一步升级为对标 5090 上 `torch` 算子的 SOTA 性能。由于 5090 与这台真机 DCU(gfx906) 的硬件算力量级不同（5090 FP32 峰值 104.8 TFLOPS，DCU 真机实测峰值仅 ~13.9 TFLOPS，相差约 7.5 倍），绝对数字直接对比对 DCU 天然不利，因此采用**双口径**方法论：跨硬件绝对数字 + 各自硬件利用率百分比分开汇报。

GEMM 已产出四条线数据（DCU 自定义 kernel、DCU rocBLAS、5090 自定义 kernel、5090 `torch.mm`(显式关闭 TF32)）：

| shape | DCU 自定义 | DCU rocBLAS | 5090 自定义 | 5090 torch SOTA |
|---|---|---|---|---|
| (1024³) 利用率 | 19.55% | 68.00% | 0.80% | 36.58% |
| (4096³) 利用率 | 22.87% | 73.90% | 1.02% | 63.26% |

**结论**：绝对吞吐上 DCU 就算用自己最好的库(rocBLAS)也比 5090 SOTA 慢 4~6.4 倍，这是硬件峰值差距决定的，不现实作为目标；但自定义 kernel 在 DCU 上的效率(19~23%)离 DCU 自己的 rocBLAS(68~74%) 还有约 3 倍空间，这是纯软件问题，是当前明确可行的优化目标。完整数据、方法论、rocBLAS/hipBLAS/MIOpenGEMM 基准库选择依据见 `.claude/skills/dcu_perf.md` 第 9 节。

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

已获得海光 DCU 真机访问（gfx906/gfx926，DTK 22.10.1），三个算子的 `hipcc` 编译与运行正确性均已验证通过（见上方"当前进度"表格与 `docs/dcu_portability_review.md`"真机验证记录"）。

尚未完成的 DCU 硬件专属工作：`rocprof` 性能剖析、`warpSize=64` 下 bank conflict 等性能结论的实机重新验证、完整 `stress_test.py`/`benchmark.py` 套件在真机上的运行，以及 flash_attn.cu 真正的 extreme 档（seq_len=4096, head_dim=128）压力测试。
