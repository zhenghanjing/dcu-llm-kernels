# DCU 大模型算子智能体项目 —— 现状总结（供新对话延续用）

## 1. 项目目的

做一个针对 DCU（海光）加速卡的大模型底层代码（C++）编程智能体。要求的路线：

1. 直接用 ClaudeCode（cc）实现几类深度学习算子（矩阵乘 GEMM、softmax、flash attention），观察生成代码的编译/运行情况，与 GPU/CPU 上 PyTorch 库的结果做对比。
2. 通过 skills 或其它方式，用 cc 优化生成结果、提高正确率；正确性有保证后，再考虑优化运行性能。
3. 测试不同编程场景下的缺点，做对应优化。

快速启动阶段直接基于 cc + skills 改；如果这条路走通了，再考虑重构 harness 做进一步优化（去掉人工编排环节，变成真正自动化的智能体）。

## 2. 当前工作流（人工编排，尚非自动化智能体）

- **Claude/Cowork（我）**：诊断 ClaudeCode 或真机门户返回的结果，写下一步给 ClaudeCode 的自包含提示词，直接编辑部分基础设施文件和文档。
- **ClaudeCode**：用户在本地 VSCode 里运行，有真实的本地编译工具链（CUDA 12.8 / nvcc，因为本地开发机是 RTX 5090；最终目标硬件是 DCU gfx906，用 hipcc/DTK 编译）。
- **真机门户**：一个网页，需要手动上传 zip + 填编译命令 + 填运行文件 + 填超时时间，才能在真正的 DCU 上跑。

**这一点很重要**：目前"skills 复用"这件事是靠人（你）手动触发、我手动编排 prompt 实现的，还不是 cc 自己调用 skills、自己决定下一步、自己提交真机验证的自动化闭环。要求里提到的"重构 harness"就是补上这一段。

## 3. 按三步路线图的完成度评估

**第一步（实现+对比 pytorch）—— 完全满足。**
GEMM/Softmax/FlashAttention 三个算子均已用 ClaudeCode 实现（HIP/CUDA C++），各自配 `validate_*.py`，用 torch 对应算子（`torch.mm`/`F.softmax`/`F.scaled_dot_product_attention`）做参考基准，FP32 atol=1e-5、FP16 atol=1e-2。

**第二步（skills 提高正确率 → 优化性能）—— 正确率部分完全验证，性能优化已有实质进展。**
把 flash_attn.cu 踩坑修复的知识点（`const void*` 强转、LDS 预算查询）沉淀进 `docs/dcu_hip_porting.md` 后，专门新写了一个从未验证过的 kernel（`kernels/gemm/gemm_large_tile.cu`）去测试这些知识能否被自主复用到新场景——真机 12/12 组合 bit-for-bit 通过，证明"skills 积累的知识可迁移"这个假设成立，不是针对某文件的一次性 patch。
性能优化：bank conflict、double 累加器开销这两项已经做完真机隔离 A/B 实测（不是理论分析，是独立编译的对照二进制各自单独测），结论都是"保留当前默认实现"（BN=64 在 DCU 上全部三档都更快；double 累加器精度必须保留，float 会掉精度）,详见 `.claude/skills/dcu_perf.md` 第 8 节。在此基础上，项目目标进一步升级为"对标 5090 上 torch 的 SOTA 性能"，GEMM 已经完成了跨硬件绝对数字 + 各自硬件利用率百分比的双口径基准对比（含 DCU 上 rocBLAS 同硬件对照组），发现自定义 kernel 在 DCU 上只打到峰值的 19~23%，而 rocBLAS 能打到 68~74%——这是当前最新、最具体的性能优化目标，详见 `.claude/skills/dcu_perf.md` 第 9 节和下方第 6 节待办。

**第三步（测试场景缺点 → 对应优化）—— 场景测试充分，正确性类缺点已修复，性能类缺点未优化。**
三个算子各测了 boundary/non-2^n/large/smem_risk/extreme 五类 shape。发现的缺点里，**阻塞编译/运行的**（cuda/hip 符号耦合、48KiB 硬编码阈值、函数指针隐式转换在 hipcc 下不合法等）**已修复**；**只影响性能的**（bank conflict、累加器精度选择）**只是被记录，尚未优化**，与第二步的性能缺口是同一个缺口。

## 4. 已验证完成的具体成果

- **三个算子在真机 (gfx906, DTK 22.10.1) 上编译+运行成功**，且数值正确性全部达标：
  - GEMM：boundary / non-2^n / large / **extreme (4096×4096×4096)** 四档，全部 bit-for-bit 一致。
  - Softmax：多档验证通过。
  - FlashAttn：boundary / non-2^n / large / smem_risk / **extreme (seq_len=4096, head_dim=128)** 五档，全部逐位一致或 atol=1e-3 内（差异量级 1e-6，符合分块+在线 softmax 改变浮点累加顺序的预期）。
- **gemm_large_tile.cu**：全新大 tile GEMM 变体，专门用来验证 skills 可复用性，12/12 组合真机 bit-for-bit 通过。
- **性能 benchmark**：三个算子在真机上都测过，GEMM 三个 shape 比 RTX 5090 快约 3 倍。
- **资源利用率监控**：新增 `kernels/common/gpu_monitor.sh`（本地 nvidia-smi）和 `kernels/common/dcu_monitor.sh`（真机 rocm-smi），确认计算确实吃满了 DCU 核心（不是凑巧算对）。
- **方法论沉淀**（可复用的经验，已写入文档/skills）：
  - 真机门户 zip 结构规律：单顶层目录会自动 cd 进去；多顶层目录（如需要 `common/gpu_compat.h`）则工作目录留在原地，路径要带前缀。
  - 大数据量测试用小 tile 平铺技巧（如 64×64 随机 tile 平铺满 4096×4096）压缩 zip 体积（128MiB→2.46MiB）同时保留真实随机性。
  - 跨平台数据一致性：本地生成二进制文件后原样传输到真机，不依赖不同平台 PRNG 序列一致。
  - rocprof 在当前 DTK 22.10.1 环境不可用，性能验证只能靠 kernel 内置 `--bench` 模式 + `dcu_monitor.sh`。
- **SOTA 对标基准（新增，2026-08-09）**：真机 `rocminfo` 实测算出 DCU 真实 FP32 理论峰值 ≈13.9 TFLOPS（此前"~20 TFLOPS"是估算错误，已作废）；确认 rocBLAS/hipBLAS 在真机上真实可链接可运行，hipBLAS 经查证是 marshalling 层、本质转发到 rocBLAS，MIOpenGEMM 是绑定 2019 年 ROCm 2.9 的遗留 OpenCL 自动调优库、无简单调用接口，两者均不纳入基准候选，只用 rocBLAS 作 DCU 侧 SOTA。GEMM 已产出四条线完整数据（DCU自定义/DCU rocBLAS/5090自定义/5090 torch，torch 侧显式关闭 TF32），双口径结论：绝对数字上 DCU 就算用 rocBLAS 也比 5090 SOTA 慢 4~6.4 倍（硬件峰值差距 7.5 倍决定的，改不动）；但利用率百分比上自定义 kernel(19~23%) 离 DCU 自己的 rocBLAS(68~74%) 还有约 3 倍效率空间，这是可行的下一步优化目标。完整数据和方法论见 `.claude/skills/dcu_perf.md` 第 9 节，探测过程见 `docs/dcu_portability_review.md` "性能对标基准探测记录"一节。

## 5. 关键文件位置

- `docs/dcu_hip_porting.md`：HIP 移植知识点（const void* 转换、LDS 预算查询、WARP_SIZE 处理等）。
- `docs/dcu_portability_review.md`：完整验证记录，包括每个算子每档 shape 的真机结果、资源利用率监控章节。
- `README.md`：项目总览 + gemm_large_tile.cu 跨 kernel 复用验证记录。
- `.claude/skills/dcu_perf.md`：真机性能实测数据（第 7 节），含 GEMM/Softmax/FlashAttn 对比表。
- `kernels/{gemm,softmax,flash_attention}/`：三个算子源码 + validate 脚本 + 真机运行脚本。
- `kernels/common/gpu_compat.h`、`gpu_monitor.sh`、`dcu_monitor.sh`：跨算子共享的移植层和监控脚本。

## 6. 未完成的具体待办（按优先级）

1. ~~bank conflict A/B 对比~~ —— ✅ 已完成，真机隔离实测，结论保留 BN=64。见 dcu_perf.md 第 8 节。
2. ~~double vs float 累加器 A/B 对比~~ —— ✅ 已完成，真机隔离实测，结论保留 double（float 会掉精度）。见 dcu_perf.md 第 8 节。
3. **缩小自定义 GEMM kernel 与 rocBLAS 的 DCU 效率差距**（尝试过一次，未走通，仍是待办）：自定义 kernel 在 DCU 上大 shape 只打到峰值 19~23%，rocBLAS 能打到 68~74%。试过把寄存器分块从 4×4 加大到 8×8(本地 5090 上 4096³ 快 14%)，但真机 DCU 上反而慢 12.4%——和 bank conflict 那次一样，5090 调优方向不能直接搬 DCU，默认行为已回退到原始小 tile 实现（效率差距原样保留在 19~23%，没有变差）。要真正缩小这个差距，需要专门针对 gfx906 的寄存器压力/occupancy/bank conflict 特性分析，不是移植 5090 结论。完整过程见 dcu_perf.md 第 10 节。
4. **把 SOTA 对标方法论扩展到 Softmax / FlashAttention**：目前双口径基准对比只做了 GEMM 一个算子，Softmax/FlashAttn 还没有对应的 rocBLAS 类比库基准（没有直接等价的厂商库，需要先确定用什么做"DCU 侧 SOTA"参照，比如 MIOpen 的对应算子实现）和 torch SOTA 基准。
5. **技术报告整理**：把以上内容整理成给老师/项目汇报的正式技术报告。
6. **（更大的结构性目标）harness 重构**：把当前"人工在 Cowork/ClaudeCode/真机门户三者间搬运上下文"的流程，往自动化闭环方向推进——这是要求里"如果基础路线跑通了，再考虑优化"对应的下一阶段工作，目前还未开始规划。

## 7. 建议汇报时的表述

方法论已验证成立（skills 可复用、正确性可保证），但工程闭环（性能优化 + harness 自动化）尚未完成——建议汇报时把"研究假设已验证"和"智能体工程目标未完成"分开说，避免让人以为项目已经是一个能自主跑的智能体。

**"对标 SOTA"这个目标汇报时也要拆成两句话，不要合并成一句"追上/超过 5090"**：
1. 绝对吞吐/延迟数字上，DCU 追平或超过 5090 基本不现实——5090 FP32 理论峰值(104.8 TFLOPS)是这台 DCU 真机实测峰值(13.9 TFLOPS)的约 7.5 倍，这是硬件天花板决定的，GEMM 实测也验证了即使用 DCU 自己最好的库(rocBLAS)也仍慢 4~6.4 倍，不是"还没优化到位"。
2. 硬件利用率百分比(达到自身理论峰值的比例)上，DCU 自定义 kernel(19~23%)离 DCU 自己的 rocBLAS(68~74%) 还有约 3 倍空间，这部分差距纯粹是软件问题，是明确可行的优化目标，也是目前项目在推进的方向。
汇报时把这两条分开陈述，不要用一个笼统的"性能对比"数字掩盖"哪些是硬件决定、哪些是软件可以改进"这个区别。
