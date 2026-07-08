---
name: dcu-perf
description: DCU/CUDA kernel 性能优化方法论 —— 如何判断瓶颈优先级、寄存器分块、bank conflict 的实测验证、grid 并行度是否受限的判断方法、已知的规模相关 trade-off，以及每次改动后的验证纪律。优化 GEMM/softmax/flash attention 等算子前应参考本文档。
---

# DCU/CUDA 性能优化经验

本文档记录本仓库对 `kernels/gemm/gemm.cu` 做三轮性能优化（float4 向量化 →
寄存器分块 → tile size 调整）过程中总结出的方法论。适用于本仓库内任何
naive tiled kernel（GEMM、softmax、flash attention）的后续性能优化工作。

## 1. 优化优先级判断：先用简单改动定位瓶颈

**方法**：不要凭直觉直接上重手段（寄存器分块、tile 重构），先做一个成本最低
的改动（如 global memory load/store 的 float4 向量化），跑一遍 benchmark，
用它的收益大小反推瓶颈在哪里。

- 如果向量化后收益明显（例如 >20%~30%），说明瓶颈确实在**访存带宽**，
  继续在这个方向深挖（更宽的向量化、更好的合并访存）是对的。
- 如果收益很小（**<5%**），说明瓶颈**不在带宽**，而在**计算访存比**
  （compute-to-memory-access ratio）过低——也就是说，从 shared memory
  读一次数据只用来做一次 FMA，寄存器复用不够。这种情况下继续优化访存模式
  收效有限，应该转向寄存器分块。

**本仓库实测**：对 `gemm.cu`（TILE=16，每线程算 1 个输出）做 float4 向量化
global load 后：

| shape                | 优化前     | 优化后     | 提升   |
|----------------------|-----------|-----------|--------|
| GEMM (1024,1024,1024)| 4.33 ms   | 4.18 ms   | ~3%    |
| GEMM (4096,4096,4096)| 266 ms    | 263 ms    | ~1%    |

收益远低于 5%，说明当时瓶颈根本不在 global memory 带宽（TILE=16 时 shared
memory 只用了 2KB/block，访存本身已经是合并访存），而在于每个线程只产出 1
个输出、复用率太低。这个判断直接指向了下一步该做寄存器分块，而不是继续在
向量化上打磨。

## 2. 寄存器分块（register blocking）：提升计算访存比的核心手段

**做法**：让每个线程计算 `TM x TN` 个输出元素（一个寄存器级的 micro-tile），
而不是 1 个。每次从 shared memory 读入的 `a_frag[TM]` / `b_frag[TN]` 通过
外积（outer product）方式复用 `TM*TN` 次乘加，而不是 1 次。

```cpp
// kernels/gemm/gemm.cu 的核心改动：每线程 4x4=16 个输出
double acc[TM][TN];   // TM=TN=4，累加器仍是 double（精度纪律见 dcu_numerics.md）
...
for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
        acc[i][j] += (double)a_frag[i] * (double)b_frag[j];
```

这一步把"计算:shared memory 读取"的指令比从原来的 `1 FMA : 2 次读`
提升到 `16 FMA : 8 次读`（即 2:1），提升了 4 倍。

**本仓库实测**（BM=BN=64, BK=16, TM=TN=4，相对 float4-only 版本）：

| shape                 | float4-only | 寄存器分块后 | 提速   |
|-----------------------|-------------|-------------|--------|
| GEMM (1024,1024,1024) | 4.18 ms     | 2.55 ms     | ~1.6x  |
| GEMM (4096,4096,4096) | 263 ms      | 129 ms      | ~2.0x  |

大尺寸下约 2 倍提速，是三轮优化里收益最大的一步——印证了"计算访存比不足"
才是真正瓶颈的判断（对应第 1 条）。

## 3. Bank conflict 分析必须实测验证，不能只靠理论下结论

**教训**：理论分析发现 bank conflict 是必要的，但**"理论上存在冲突"不等于
"修复后一定更快"**——消除冲突往往有代价（更小的 tile、更多的 block、更多
重复访存），必须拿实际 benchmark 数据说话，针对目标场景（本仓库以 large/
extreme 档位为准）做判断。

**本仓库案例**：分析发现寄存器分块后的 B-fragment shared memory 读
（`sB[kk][threadIdx.x*TN]`）存在理论上的 2-way bank conflict（`BN=64` 时一个
warp 内有 16 个不同地址，但 32 个 bank 只能提供 8 个不冲突的 4-bank 槽位）。

对此做了实测而非直接"修复"：

| 改动                        | non-2^n (100,300,200) | large (1024³) | extreme (4096³) |
|-----------------------------|------------------------|---------------|------------------|
| 保留 conflict（BN=64，基线）| 0.38 ms                | 2.55 ms       | 129 ms           |
| 消除 conflict（BN=32）      | 0.21 ms（**快 45%**）  | 2.68 ms（慢5%）| 133 ms（慢3%）   |

结论：消除 bank conflict 对小尺寸有明显收益，但对大/超大尺寸反而更慢——因为
`BN` 减半导致每个 A-tile 列在 N 方向被重复加载的次数翻倍，这个代价超过了
消除 conflict 省下的开销。由于本仓库的优化目标是大尺寸场景（GEMM 原本在大
尺寸下比 PyTorch 慢 77~129 倍），最终**保留了有理论 bank conflict 的
`BN=64` 版本**，因为它在目标场景上实测更快。

**方法论**：任何 bank conflict 的"修复"都应该同时跑一遍目标场景（通常是
large/extreme）和一个反例场景（通常是 non-2^n/小尺寸）的 benchmark，两边
都要看，再决定要不要采用。

## 4. 已知 trade-off：大寄存器 tile 在小规模问题上可能变慢

**现象**：寄存器分块把 block 的输出 tile 从 16x16 放大到 64x64 后，一个
100x200 的小规模输出只需要 `ceil(200/64) x ceil(100/64) = 4x2 = 8` 个
block，而之前 TILE=16 时需要 91 个 block。GPU 有上百个 SM
（如本机 RTX 5090 约 170 个），8 个 block 远不足以打满并行度。

**本仓库实测**（GEMM (100,300,200)，float4-only → 寄存器分块）：

| 阶段            | 耗时      | 相对 PyTorch |
|-----------------|-----------|--------------|
| float4-only     | 0.054 ms  | 2.51x        |
| 寄存器分块后    | 0.38 ms   | 17.29x       |

小尺寸变慢了约 7 倍。**这是大寄存器 tile 的常见、预期内的规模相关
trade-off**，不是 bug——真实的 GEMM 库（如 cuBLAS）正是因为这个原因，会
针对不同问题规模派发不同的 kernel 变体（tile size 越大，对大问题越有利，
对小问题并行度越不足）。

**处理原则**：遇到这种 trade-off 时如实报告数字和原因，不要为了让某个
benchmark 好看而隐藏另一个场景的回归。如果需要同时兼顾大小规模，应该作为
独立需求（大小规模自适应派发多个 kernel 变体）明确提出，而不是默默塞进
"单一改动"的优化步骤里。

## 5. grid 并行度 vs 每线程复用：如何判断哪个更稀缺

**教训**：寄存器分块（第 2 条）的本质是"用更多每线程工作量换取每次访存的
复用次数"，但这笔交易只有在 **block 数量本来就远超 GPU 的 SM 数**时才划算
——如果 block 数量本来就不多，让每个线程做更多工作只会进一步压缩 block
数量，把"并行度受限"的问题变得更严重，而不是缓解"复用受限"的问题。这两种
瓶颈需要相反的对策，用错方向会让优化反而变成劣化。

**GEMM 场景**：大尺寸问题下 block 数天然充足，可以放心用寄存器分块换复用。
`kernels/gemm/gemm.cu` 用 BM=BN=64 分块后，grid 大小：

| shape                 | grid 大小        | vs 本机 170 个 SM |
|------------------------|------------------|-------------------|
| GEMM (1024,1024,1024) | 16×16 = 256 block | ~1.5x（略有富余） |
| GEMM (4096,4096,4096) | 64×64 = 4096 block| ~24x（明显富余）  |

寄存器分块后大尺寸提速约 2 倍（见第 2 条），代价是小尺寸 (100,300,200) 的
grid 只有 4×2=8 block，远小于 170——这正是第 4 条记录的、GEMM 自己在小
尺寸上也会踩的同一个坑，只是大尺寸场景下 block 数够多，能把这笔交易的
代价"稀释"到可以忽略。

**FlashAttn 场景**：即使是"large / extreme"档位，`seq_len` 对应的
`grid.x` 也可能远小于 SM 数，这时增加每线程工作量（无论是让每个线程多算
几行，还是把 tile/BLOCK_SIZE 调大）都会因为并行度不足而适得其反。
`kernels/flash_attention/flash_attn.cu` 用 `grid.x = ceil(seq_len/64)`：

| shape (seq_len,d)     | grid.x         | vs 本机 170 个 SM        |
|------------------------|----------------|---------------------------|
| FlashAttn (100,64)    | ceil(100/64)=2 | ~85x **不足**             |
| FlashAttn (512,64)    | ceil(512/64)=8 | ~21x **不足**             |
| FlashAttn (4096,128)  | ceil(4096/64)=64| ~2.7x **不足**（extreme 档也不够！）|

三档全部远低于 SM 数——注意即使是"extreme"档 (4096,128)，64 个 block 也
比 170 个 SM 少。这个仓库在这个 kernel 上做了两次寄存器分块方向的尝试
（每线程算 4 行、把 BLOCK_SIZE 从 64 提到 128/256），两次都让 grid.x 进一步
减半到 1/4，实测双双变慢 3~5 倍——不是实现细节的问题，而是从一开始就选错
了优化方向：这个 kernel 是"并行度受限"，不是"复用受限"。详见
`flash_attn.cu` 里保留的失败实验注释。

**判断方法**：优化前先算一次 `grid.x (× grid.y × grid.z)` vs GPU 的 SM 数
（`torch.cuda.get_device_properties(0).multi_processor_count`，本机
RTX 5090 为 170），不需要跑 benchmark 就能提前判断方向：

- 如果 grid 大小**远大于** SM 数（如 GEMM 大尺寸的 24x 富余）：这个 kernel
  是"复用受限"，可以放心用寄存器分块（第 2 条）换计算访存比。
- 如果 grid 大小**接近或小于** SM 数（如 FlashAttn 全部三档）：这个 kernel
  是"并行度受限"，寄存器分块只会让 block 数进一步变少，方向错误。应该优先
  考虑**增加 block 数量**的思路，例如按 kv/reduction 维度切分出更多独立
  block（每个 block 处理一部分 kv-tile，配合 atomic 或两阶段规约合并部分
  结果），而不是让每个线程做更多事。这个仓库目前还没有实现这个方向，留作
  后续优化 FlashAttn 的候选方案。

**方法论**：grid 大小 vs SM 数的核算应该在写代码之前做，不要等 benchmark
跑出回归了才倒推原因——本仓库在 FlashAttn 上就是先实现两版寄存器分块变体、
测出回归后才做这个核算，如果提前算过 grid.x=2/8/64 vs 170 个 SM，两次尝试
都可以省下来。

## 6. 验证纪律：每次改动后必须同时跑两个脚本

**硬性要求**：每做一次性能改动（哪怕只改一个 tile 参数），必须同时跑：

1. `python tests/stress_test.py --skip-compile` —— 验证**正确性**没有退步
   （10 个用例全部 PASS，且大部分 shape 误差应为 `0.00e+00`，见
   [[dcu_numerics]] 的 double 累加器纪律）。
2. `python tests/benchmark.py --skip-compile` —— 验证**性能**确实有提升，
   同时观察是否有其他 shape 出现回归（如第 3、4 条里的 trade-off）。

**不能只看其中一个**：只跑 benchmark 会让精度回归悄悄溜过去；只跑
stress_test 会让"看起来没退步但其实没有变快甚至变慢"的改动被误当成优化。
本仓库三轮优化（float4 向量化、寄存器分块、tile size 调整）都遵循了
"改一处 → 编译 → stress_test → benchmark → 再决定下一步"的顺序，包括对
被放弃的实验性改动（`BN=32`、`BK=32`）也是先跑完两个脚本拿到数据，再决定
不采用。

## 参考实现

- GEMM 三轮优化后的最终实现（复用受限案例）：
  [kernels/gemm/gemm.cu](../../kernels/gemm/gemm.cu)
- FlashAttn 优化实现，含两次失败的寄存器分块实验注释（并行度受限案例）：
  [kernels/flash_attention/flash_attn.cu](../../kernels/flash_attention/flash_attn.cu)
- 正确性验证：[tests/stress_test.py](../../tests/stress_test.py)、
  [kernels/gemm/validate_gemm.py](../../kernels/gemm/validate_gemm.py)
- 性能基准（含 GPU-side cudaEvent 计时、PyTorch 对照）：
  [tests/benchmark.py](../../tests/benchmark.py)
- 精度纪律（与本文档配套）：[dcu_numerics.md](dcu_numerics.md)
