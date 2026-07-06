---
name: dcu-numerics
description: DCU/CUDA 归约类算子（GEMM/softmax/reduction等）的 FP32 累加精度问题、double 累加器解法，以及性能优化时的精度纪律约束。编写或优化此类算子时应参考本文档。
---

# DCU/CUDA 数值精度经验

本文档记录本仓库在 GEMM 等归约类算子上遇到的 FP32 精度问题、已验证的解决方案，
以及后续做性能优化时必须遵守的精度纪律。适用范围覆盖 `kernels/` 下所有
"多次累加再输出"的算子：GEMM 内积、softmax 分母求和、其他 reduction。

## 1. 问题：FP32 累加精度随累加次数增大而劣化

**现象**：矩阵乘等归约类算子的误差随累加次数（例如 GEMM 的 K 维度）增大而增大，
典型规律接近 `误差 ~ O(√K)`。

**根因**：FP32 求和的舍入漂移会随加数个数近似按平方根规律累积。tile 内每次
`sum += a*b` 都会引入 O(ε) 的舍入误差，K 越大，误差累积越明显。

**本仓库实测数据**（优化前，纯 FP32 累加器）：

| K    | max error |
|------|-----------|
| 200  | 9e-5      |
| 4096 | 5e-3      |
| 8192 | 5e-3 量级 |

均超过 `CLAUDE.md` 规定的 FP32 容差 `atol=1e-5`。

## 2. 解决方案：double 精度累加器

**做法**：内积/归约的累加器声明为 `double`，参与乘加的输入 tile 仍保持
`float32`（不额外占用带宽），只在写出结果时转换回目标 dtype。

```cpp
// kernels/gemm/gemm.cu 的核心改动
double acc = 0.0;                       // 累加器提升到 double
...
acc += (double)a_val * (double)b_val;   // 输入仍是 float，乘加提升到 double
...
C[...] = (float)acc;                    // 最终写出时转回 float
```

**验证结果**：修复后，K=200~8192 范围内 max error 从 9e-5~5e-3 降到
**0.00e+00**（对照 `torch.mm(A.double(), B.double()).float()` 的正确舍入结果）。
验证脚本：`kernels/gemm/validate_gemm.py`、`tests/stress_test.py`。

**适用范围**：不止 GEMM，任何"多次累加再输出一次"的算子都适用同样的模式，例如
softmax 的分母求和、其他 reduction 类算子。新写此类 kernel 时应默认采用 double
累加器，而不是等出现精度问题后再补救。

## 3. 性能优化时的精度纪律（硬约束）

在对已验证正确的算子做性能优化（float4 向量化、寄存器分块、tile size 调整等）
时，必须遵守以下两条：

1. **每次性能改动后都必须重跑精度验证脚本**（`validate_gemm.py` /
   `stress_test.py` 等），确认误差没有劣化，而不能只看 `benchmark.py` 的耗时数字。
2. **累加器精度类型是不可优化项**。即使为了性能想用 `float4` 向量化
   load/store、增大寄存器分块（每线程计算多个输出）、调整 shared memory tile
   size，累加器本身（如 `double acc`）的精度类型**不能降级**为 float 或更低精度
   ——这三类改动都只影响"怎么搬数据、怎么复用寄存器"，不应该也不需要触碰
   "累加发生在什么精度下"这件事。

**理由**：正确性（精度）和性能是两条独立的验收线，性能优化不能拿精度做交换。
以 GEMM 为例，`kernels/gemm/gemm.cu` 经过三轮性能优化——
① float4 向量化 global load/store　② 寄存器分块（4×4 micro-tile）
③ shared memory tile size 调整与 bank conflict 排查——
每一轮都保持 `double acc` 不变，每一轮改完都重跑
`tests/stress_test.py` + `tests/benchmark.py`，确认 10/10 用例仍然 PASS
（多数 shape 误差为 `0.00e+00`）之后才进入下一轮。

## 参考实现

- 精度修复：[kernels/gemm/gemm.cu](../../kernels/gemm/gemm.cu)
- 精度验证：[kernels/gemm/validate_gemm.py](../../kernels/gemm/validate_gemm.py)、
  [tests/stress_test.py](../../tests/stress_test.py)
- 性能基准：[tests/benchmark.py](../../tests/benchmark.py)
