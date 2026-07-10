---
name: dcu-numerics
description: DCU/CUDA 归约类算子（GEMM/softmax/reduction等）的 FP32 累加精度问题、double 累加器解法、何时不需要 double 累加器的判断依据，以及性能优化时的精度纪律约束。编写或优化此类算子时应参考本文档。
---

# DCU/CUDA 数值精度经验

本文档记录本仓库在 GEMM 等归约类算子上遇到的 FP32 精度问题、已验证的解决方案，
以及后续做性能优化时必须遵守的精度纪律。**注意**：double 累加器不是对所有
"多次累加再输出"的算子都必要——第 3 节记录了一个用实测数据证伪的反例
（softmax 分母求和），先判断输出是否会被同量级的除法/归一化稀释误差，
再决定要不要上 double，不要看到"归约"两个字就默认套用。

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

**适用范围（有条件，见第 3 节反例）**：GEMM 这类"输出量级不随累加次数变化、
误差没有被稀释机制"的归约，应该默认采用 double 累加器，而不是等出现精度
问题后再补救。但"归约类算子"不是自动套用 double 的充分条件——是否需要，
取决于输出是否存在把误差和数值一起稀释掉的归一化步骤，见下一节的实测反例。

## 3. 反例：softmax 分母求和不需要 double 累加器

**背景**：`kernels/softmax/softmax.cu` 的分母求和一直用 `float` 累加器，
写在本文档第 1、2 节总结出来之前，一度被认为是历史遗留的"精度债"，怀疑
会重复 GEMM 那种"累加次数越大、误差越大"的问题。

**验证方法**：不凭直觉判断，用 `validate_softmax.py` 固定 M=64，扫一遍
N=512/1024/2048/4096/8192（对应 GEMM 当初扫 K 的方法），实测数据：

| N    | max error | mean error |
|------|-----------|------------|
| 512  | 9.31e-10  | 1.31e-10   |
| 1024 | 5.82e-10  | 7.43e-11   |
| 2048 | 2.33e-10  | 3.79e-11   |
| 4096 | 1.46e-10  | 2.15e-11   |
| 8192 | 8.73e-11  | 1.32e-11   |

全部远低于 `atol=1e-5`，且趋势与 GEMM **相反**——误差随 N 增大**单调下降**，
N=8192 比 N=512 还小了一个数量级，完全没有向阈值靠近的迹象。

**根因（为什么和 GEMM 的规律相反）**：GEMM 的每个输出是"K 个同量级项的内积"，
输出本身的量级不随 K 变化，所以累积的舍入误差没有被稀释的渠道，只会越滚越大。
Softmax 的输出 `y = exp(x - max) / row_sum` 多了一次除以 `row_sum` 的操作——
`row_sum` 本身近似正比于 N（N 个同量级项之和），这一除同时把**数值**和**绝对
误差**都按约 `1/N` 的比例压低，误差不但不会累积，反而会被稀释。float 累加器
在这里不是精度瓶颈。

**结论**：**不改** `softmax.cu` 的累加器精度。这不是"偷懒不修"，是有实测数据
支撑的判断——第 2 节"适用范围"那句话必须加条件，不能看到"归约"就默认套用
double。

**方法论（比这一个结论更重要）**：任何"要不要上 double 累加器"的判断，都应该
用 `validate_*.py` 实测扫一遍代表性规模（不是只测一个默认 shape），画出
误差-规模曲线，再决定要不要改代码；不要仅凭"这是个归约类算子"这种表面相似性
就类比套用其他算子已验证过的修复方案。判断的关键点是：**这个归约的输出，
后续有没有被除以一个同样随规模增长的量**——有（如 softmax），大概率不需要
double；没有（如 GEMM 内积），大概率需要。

**局限**：以上数据只覆盖 `torch.rand`（均匀分布、数值范围收敛）的输入。没有
测试极端/病态输入（比如 logits 里有远大于其余值的离群点）下 float 累加器
是否仍然安全——如果以后要验证更极端的输入分布，应该补一组针对性测试，不能
直接沿用这份数据的结论。

## 4. 性能优化时的精度纪律（硬约束）

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

- 精度修复（GEMM，正例）：[kernels/gemm/gemm.cu](../../kernels/gemm/gemm.cu)
- 精度验证（GEMM）：[kernels/gemm/validate_gemm.py](../../kernels/gemm/validate_gemm.py)、
  [tests/stress_test.py](../../tests/stress_test.py)
- 精度反例（softmax，未改动，保持 float 累加器）：
  [kernels/softmax/softmax.cu](../../kernels/softmax/softmax.cu)、
  [kernels/softmax/validate_softmax.py](../../kernels/softmax/validate_softmax.py)
- 性能基准：[tests/benchmark.py](../../tests/benchmark.py)
