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

## 7. 真机 (DCU) 性能实测：三个算子的真实数字，以及哪些旧结论还没被验证

**背景**：本文档第 1-6 条全部是在 RTX 5090 (nvcc) 上测出的结论。DCU 计算
节点没有 torch，也没有 `rocprof`（探测过，`command not found`，DTK
22.10.1 这个安装里没带），所以没法照搬 `tests/benchmark.py` 的方案，也
拿不到 occupancy/显存带宽这类 kernel 级别的细粒度 profiling 数据。这一轮
用的是每个 kernel 自带的 `--bench` 模式（`hipEventElapsedTime` 计时，
warmup+iters 取中位数——GEMM/Softmax/FlashAttn/RMSNorm 四个文件都已经内置
了这个模式，不需要额外写代码），配合新增的
`kernels/common/dcu_monitor.sh`（后台采样 `rocm-smi`，确认 DCU 核心真的在
跑，不是"进程退出码 0"这种弱证据——用法和踩过的坑见
`docs/dcu_portability_review.md`"资源利用率监控"一节）。

**GEMM**（shape 与本文档第 1-5 条的 RTX 5090 基线完全对齐）：

| shape | RTX 5090 (nvcc) | DCU 真机 | 倍率 |
|---|---|---|---|
| (100,300,200) | 0.38 ms | 0.114 ms | 快 ~3.3x |
| (1024,1024,1024) | 2.55 ms | 0.789 ms | 快 ~3.2x |
| (4096,4096,4096) | 129 ms | 43.15 ms | 快 ~3.0x |

三个 shape 一致地快约 3 倍，extreme 档 `dcu_monitor.sh` 确认 `card1`
利用率 avg 71.1%、max 100%，不是"跑了个寂寞"。

**Softmax**（本文档之前没有为 Softmax 建过 RTX 5090 的 ms 基线，这里只有
DCU 侧的绝对数字，没有倍率可比）：

| shape | DCU 真机 |
|---|---|
| (100,300) | 0.017 ms |
| (128,1024) | 0.019 ms |

单次 kernel 耗时在微秒级，60 次 warmup+iters 总共只有约 1ms，`rocm-smi`
每次调用本身的进程开销（约 1 秒级）根本来不及采到一个样本——
`dcu_monitor.sh` 如实报了"0 samples captured"的警告，而不是印一个引起误解
的 `0%`，这是符合设计的正确行为，不是新 bug。

**FlashAttn**（shape 同样对齐本文档第 5 条的 grid 分析表；同样没有 RTX
5090 的 ms 基线可比）：

| shape (seq_len,head_dim) | DCU 真机 |
|---|---|
| (100,64) | 3.87 ms |
| (512,64) | 18.07 ms |
| (4096,128) | 276.97 ms |

**(4096,128) 是这次的重点**：这是 `docs/dcu_portability_review.md` 里标注
的真正 extreme 档，此前只验证过它 64KiB 共享内存这一个边界点（用
`smem_risk` 这个更小的替代 shape (128,128)），网格规模、显存访问量这些
维度从没有在真机上完整跑过。这次是第一次端到端跑通——编译、执行、计时
全部成功，`dcu_monitor.sh` 全程 36 个采样点 `avg_use=100.0%`、
`max_use=100%`，没有一个样本掉下来，说明这个 shape 在真机上不存在网格
规模不足或显存问题，`docs/dcu_portability_review.md`"仍待验证"清单里这一
条现在可以标记为已验证。

**这一轮遗留了两条"必须重测，不能沿用 RTX5090 结论"的待办——`double` 累加器
的相对开销、bank conflict 在 wavefront=64 下的实际表现——都已在后续一轮
真机隔离 A/B 实验中补齐，不再是间接证据。完整数据、方法和结论见下面
第 8 节。**

## 8. Bank conflict / 累加器隔离 A/B：真机 (DCU) 完整数据

**背景**：第 7 节末尾记录的两个待办——bank conflict 在 wavefront=64 下的
实际表现、`double` 累加器的相对性能开销——这一轮补齐，用真机上独立编译的
对照二进制做隔离 A/B，而不是像第 7 节那样只看"整体 kernel 更快"这种
间接证据。

**参数化做法**：`gemm.cu` 新增两个编译期宏，默认值与重构前完全一致：

- `GEMM_BN`（默认 64）：控制 `BN`（tile 列宽），`-DGEMM_BN=32` 编译出
  bank-conflict-free 对照版本。
- `GEMM_ACC_T`（默认 double）：控制内积累加器类型，`-DGEMM_ACC_T=float`
  编译出 float 累加器对照版本。

不加任何 `-D` 编译出的默认二进制与重构前逐位一致——已用
`tests/stress_test.py --skip-compile` 验证 10/10 PASS，GEMM 各档误差仍为
`0.00e+00`，确认参数化本身没有引入任何行为变化。

**测量方法**：对 `gemm`（BN=64, double，基线）、`gemm_bn32`（BN=32, double）、
`gemm_accf`（BN=64, float）三个二进制，在 (100,300,200)/(1024³)/(4096³)
三档 shape 上，各自独立调用 15 次 `--bench`（15 个分开的进程，不是把
`--bench` 自身的 iters 调大），记录 15 个 `BENCH_MS` 原始值后再算均值和
标准差——避免"单次运行"的偶然误差冒充成结论。

**Bank conflict（BN=64 baseline vs BN=32 conflict-free）**：

| shape | BN=64 (ms, mean±std) | BN=32 (ms, mean±std) | 差异 |
|---|---|---|---|
| (100,300,200) | 0.1138±0.0001 | 0.1300±0.0001 | 慢 14.2% |
| (1024,1024,1024) | 0.7886±0.0004 | 1.0323±0.0009 | 慢 30.9% |
| (4096,4096,4096) | 43.141±0.007 | 45.441±0.018 | 慢 5.3% |

**累加器（double baseline vs float）**：

| shape | double (ms, mean±std) | float (ms, mean±std) | 差异 |
|---|---|---|---|
| (100,300,200) | 0.1138±0.0001 | 0.0790±0.0004 | float 快 1.44x |
| (1024,1024,1024) | 0.7886±0.0004 | 0.4593±0.0038 | float 快 1.72x |
| (4096,4096,4096) | 43.141±0.007 | 17.904±0.012 | float 快 2.41x |

**结论**：

- **bank conflict**：DCU 上的方向和 RTX5090 相反——RTX5090 上消除冲突
  （BN=32）只在小尺寸更快（快 45%），大/超大尺寸反而变慢；DCU 上消除
  冲突在**全部三档**都更慢（14.2%/30.9%/5.3%），没有任何尺寸受益。当前
  默认 `BN=64`（保留理论 bank conflict）在 DCU 上依然是正确选择，不需要改。
- **累加器**：float 累加器的性能优势已量化，且随规模增大而扩大（1.44x →
  1.72x → 2.41x，extreme 档差距最大）。但 float 累加器本身在本地 RTX5090
  上已经测出不满足 `atol=1e-5`（K=8192 时 max error 1.12e-02，见
  `dcu_numerics.md`）——这是算法层面的精度问题，和平台无关，DCU 上不会
  变好。因此这 1.44~2.41x 的开销是保证正确性必须付出的代价，不采纳切换
  成 float。

**`dcu_monitor.sh` 复测**：extreme 档三个变体（`gemm`/`gemm_bn32`/
`gemm_accf`）`max_use` 全部 100%，确认三次都是真实计算负载，不是"进程
退出码 0"这种弱证据。`gemm_accf` 的 `avg_use`（33.0%）明显低于另外两个
变体，但这不代表利用率真的低——是因为它总耗时短（~18ms vs baseline
~43ms/BN32 ~45ms），`rocm-smi` 采样点集中在进程启动/收尾的空档，和第 7
节记录的 Softmax"耗时太短、采样被稀释"是同一类现象，不是新问题。

**参考实现**：
- 参数化实现：[kernels/gemm/gemm.cu](../../kernels/gemm/gemm.cu)
  （`GEMM_BN`/`GEMM_ACC_T` 宏）
- A/B 数据采集脚本：
  [kernels/gemm/run_gemm_ab_bench.sh](../../kernels/gemm/run_gemm_ab_bench.sh)

## 9. GEMM vs 厂商库/PyTorch SOTA 基准对比：双口径方法论与真机数据

**背景**：项目目标从"DCU 上跑通、比自己之前的版本快"升级为"对标 SOTA"——
即 5090 上 `torch` 算子的性能。但 5090 FP32 峰值(104.8 TFLOPS)和这台真机
DCU(gfx906, `Device 66a1`, SKU D160A1)的 FP32 峰值不在一个量级，直接比
绝对数字对 DCU 天然不利，所以确定了**双口径分开汇报**的方法论：一条看
跨硬件绝对数字，一条看各自硬件的利用率百分比，两条线各自成立、不能混着
下结论。

**DCU 真实理论峰值（真机 `rocminfo` 实测，不是估算）**：

```
Compute Unit: 64, SIMDs per CU: 4, Wavefront: 64, Max Clock: 1700MHz
FP32 峰值 = 64 CU × 64 ALU/CU(4×16) × 2(FMA) × 1.7GHz ≈ 13.9 TFLOPS
FP64 峰值(gfx906 1:2 比例) ≈ 6.96 TFLOPS
FP16(packed) 峰值 ≈ 27.8 TFLOPS
```

用同架构 MI60 的公开数据（64CU, 1.8GHz, 官方标称 FP32=14.7 TFLOPS）反推
同一公式验证过，算法可信。此前凭 FP64≈10TFLOPS 倒推的"~20 TFLOPS"是错的
偏高估计，已作废。5090 FP32 峰值 104.8 TFLOPS 来自公开规格。两者相差约
**7.5 倍**，这是硬件天花板决定的，不是软件能弥补的差距。

**SOTA 基准库选择（真机探测过，不是随便选的）**：

- **5090 侧**：`torch.mm`，且**显式**设 `torch.backends.cuda.matmul.allow_tf32
  = False`——5090 有张量核心，`torch.mm` 对 FP32 输入默认可能走 TF32
  路径（更快但精度类型不同），不显式关闭会导致这条基准线的峰值分母选错，
  出现利用率>100%这种荒谬结果。
- **DCU 侧**：`rocBLAS`（`rocblas_sgemm`）。探测确认头文件/库/链接+运行
  测试全部通过（`/opt/rocm/include/rocblas.h`，`-lrocblas` 直接可链接）。
  **`hipBLAS` 不是独立实现**——查过 ROCm 官方文档，它是一个 marshalling
  层，在这类 AMD/ROCm 平台上本身就是转发调用 rocBLAS，单独测它是重复劳动。
  **`MIOpenGEMM`(`libmiopengemm.so`) 探测后确认不适合**：绑定 ROCm 2.9
  (2019年)的遗留 OpenCL 自动调优库(符号表里是 `_cl_event`/
  `_cl_command_queue` 这类 OpenCL 类型)，没有一行调用式的简单 GEMM
  接口，得先建 `Geometry`/`HyPas`/`Constraints` 对象再跑 `find1()`
  自动调优搜索，投入产出比不划算，不纳入基准候选。

**四条线真机+本地实测数据**（shape 与本文档第 1-8 条完全对齐，
`(100,300,200)`/`(1024³)`/`(4096³)`，FLOPs 按 `2×M×N×K` 算）：

| shape | DCU 自定义kernel | DCU rocBLAS | 5090 自定义kernel | 5090 torch SOTA(TF32关) |
|---|---|---|---|---|
| (100,300,200) | 0.105 TFLOPS | 0.514 TFLOPS | 0.032 TFLOPS | 0.473 TFLOPS |
| (1024³) | 2.723 TFLOPS | 9.472 TFLOPS | 0.842 TFLOPS | 38.337 TFLOPS |
| (4096³) | 3.186 TFLOPS | 10.296 TFLOPS | 1.065 TFLOPS | 66.297 TFLOPS |

换算成各自硬件利用率百分比（DCU 两列 ÷13.9TFLOPS，5090 两列 ÷104.8TFLOPS）：

| shape | DCU 自定义 | DCU rocBLAS | 5090 自定义 | 5090 torch SOTA |
|---|---|---|---|---|
| (100,300,200) | 0.76% | 3.69% | 0.03% | 0.45% |
| (1024³) | 19.55% | 68.00% | 0.80% | 36.58% |
| (4096³) | 22.87% | 73.90% | 1.02% | 63.26% |

原始 ms 数据：DCU 自定义kernel 取自第 8 节的 BN=64/double 基线
（0.1138/0.7886/43.141 ms）；DCU rocBLAS 真机实测
0.023360/0.226725/13.349571 ms；5090 自定义kernel 取自第 7 节基线
（0.38/2.55/129 ms）；5090 torch SOTA 本地实测
0.025392/0.056016/2.073072 ms。

**结论，两条口径分开说**：

- **绝对数字**：即使用 DCU 自己最好的库(rocBLAS)去对比 5090 SOTA，大
  尺寸下仍慢 4~6.4 倍(1024³ 慢 4.05x，4096³ 慢 6.44x)，和硬件峰值差距
  (7.5倍)基本吻合——这是硬件决定的差距，不是"还没优化好"，不应作为
  "达到/超过"的目标口径。
- **利用率百分比**：DCU 自己的 rocBLAS 能打到自身峰值的 68~74%，而当前
  自定义 kernel 只有 19~23%——同一块硬件上有约 3 倍的软件效率空间可挖，
  这个差距纯粹是软件问题，是可以通过继续优化去缩小的、现实可行的目标。
  另外 rocBLAS 在 DCU 上的效率(74%)比 cuBLAS/torch 在 5090 上关闭 TF32
  后的效率(63%)还高，说明"打到本平台 SOTA 级效率"这件事本身并不是不可能
  达到的高标准。
- **可执行的下一步目标**：把自定义 kernel 在 DCU 上的效率从 ~23% 推向
  接近 rocBLAS 的 ~74%，即"DCU 上的软件层面 SOTA"，而不是妄想在绝对数字
  上超过 5090。

## 10. 缩小自定义 kernel 效率差距的一次尝试：5090 调优不能直接搬 DCU

**背景**：第 9 节确认自定义 kernel 在 DCU 上只打到峰值 19~23%，rocBLAS 能到
68~74%，尝试用"寄存器分块再加大一档"这个第 2 条已验证有效的杠杆去缩小
差距。

**本地(5090) 实验**：把寄存器分块从 4×4 加大到 8×8，同时 BM/BN 从 64 加大
到 128(NUM_THREADS 保持 256 不变)。15 次独立进程实测(mean±std)：

| shape | baseline(64/4×4) | 128/8×8 | 变化 |
|---|---|---|---|
| (100,300,200) | 0.3919±0.0003 ms | 1.2417±0.0784 ms | 慢 3.17x |
| (1024³) | 2.6896±0.0615 ms | 4.4095±0.0403 ms | 慢 1.64x |
| (4096³) | 141.18±0.876 ms | 121.23±4.41 ms | **快 14.1%** |

原因：128-tile 下 grid block 数骤减(1024³ 只剩 8×8=64 block，170 个 SM 都
喂不满，从"复用受限"翻成"并行度受限"，第 4 条的陷阱在更大规模复现)；
4096³ grid 仍有 1024 block，两边硬件都有富余，收益是真的。又加了一层
共享内存双缓冲(software pipelining)，实测收益在噪声量级(<3.5%)，说明
256 线程/block 时 occupancy 已经够好，warp 调度自己就把大部分延迟藏住了。

**自适应派发**：写了运行时 dispatch(`use_large_tile()`)，按 grid 富余度
(相对 170 这个两平台里较大的 SM/CU 数，取 ≥4x 保守阈值)在两个 kernel
变体间选择，本地验证 1024³/小 shape 打平、4096³ 保留 12.1% 提升、
threshold 边界(margin=4.0 精确边界)逐一验证过。

**真机(DCU) 验证——发现负优化，和 5090 方向相反**：正确性四档全部逐位
一致(两条 dispatch 分支都算对了)，但性能：

| shape | DCU 真机(小tile，不变) | DCU 真机(128-tile 触发) | 变化 |
|---|---|---|---|
| (100,300,200) | 0.1141 ms | (不触发) | 打平 |
| (1024³) | 0.7847 ms | (不触发) | 打平 |
| (4096³) | 43.141 ms(旧基线) | **48.487 ms** | **慢 12.4%** |

4096³ 正是 dispatch 切到 128-tile 的 shape——5090 上快 12.4%，DCU 上反而
慢 12.4%，几乎对称反转。换算利用率：这个 shape 从 22.87% **倒退到
20.35%**。和第 8 节 bank conflict 那次"两平台方向相反"是同一类现象，但
这次更直接：5090 的调优结果在 DCU 上不是"无效"，是"负优化"。推测原因
是 gfx906 的寄存器文件/occupancy 特性与 5090 不同——TM=TN=8 的累加器
让每线程寄存器占用翻 4 倍，5090 上换来的计算复用收益，在 DCU 上可能被
occupancy 下降或 wavefront=64 下的新 bank conflict 抵消并反超，具体机制
还没有专门分析过。

**处理**：默认(不传任何宏)行为已改回**永远走小 tile**，即原始 kernel
行为，128-tile 相关代码和 dispatch 框架保留但需要显式加
`GEMM_ENABLE_AUTO_DISPATCH=1` 才会生效，不会意外进入真机上的负优化路径。
本地验证 4096³ 已回到 141.9±0.76 ms(原始基线量级)，其余 shape 不变。
这次改动不需要重新提交真机——恢复的是已经在真机上验证过很多次的小
tile 行为，等价性可从已有真机数据传递而来，不用重新烧一次门户。

**结论**：Task 目标(把自定义 kernel 效率从 ~23% 推向接近 rocBLAS 的
~74%) **这条路径没有走通**——DCU 效率差距依然是 19~23%，没有变化(也
没有变差，默认行为已回退)。这次尝试的价值在于验证了一个更普遍的
教训：**5090 上验证有效的调优参数不能假设直接搬到 DCU，必须真机重新
验证**，本仓库现在有两个独立案例支持这个结论(bank conflict 方向相反、
寄存器分块方向相反)。要真正缩小 DCU 效率差距，需要针对 gfx906 自己的
寄存器压力/occupancy/bank conflict 特性专门分析调优参数，而不是把 5090
调优结果当作起点——这是后续待办，本轮未完成。

**参考实现**：
- rocBLAS 对照基准：`kernels/gemm/rocblas_sgemm_bench.cu`、
  `kernels/gemm/run_gemm_rocblas_bench.sh`
- 5090 torch SOTA 基准：`kernels/gemm/benchmark_torch_gemm_sota.py`
- 自适应派发框架(默认关闭)：`kernels/gemm/gemm.cu` 的
  `GEMM_ENABLE_AUTO_DISPATCH`/`use_large_tile()`
- 真机硬件参数/库探测记录：`docs/dcu_portability_review.md`
  "性能对标基准探测记录"一节

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
