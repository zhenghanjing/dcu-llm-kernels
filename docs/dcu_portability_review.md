# CUDA → HIP 可移植性审查报告

**审查范围**：`kernels/gemm/gemm.cu`、`kernels/softmax/softmax.cu`、
`kernels/flash_attention/flash_attn.cu`，以及围绕它们的编译脚本
（`tests/stress_test.py`、`tests/benchmark.py`、`kernels/*/validate_*.py`）。

**背景**：项目即将把 `kernels/` 部署到真实的海光 DCU 硬件上用 `hipcc` 编译。
此前所有开发和测试都在 NVIDIA RTX 5090 上用 `nvcc` 完成，HIP 代码路径
（`#ifdef __HIP_PLATFORM_HCC__` 分支）从未被 `hipcc` 实际编译过。

**审查方式**：静态代码审查。**当前环境没有真实 hipcc/DTK 工具链可用于验证**
——所有标注"需真机验证"的结论，请在拿到 DCU 硬件后优先确认，不要直接当作
定论采纳。

---

## 一个比清单里任何一项都更根本的发现

在按检查项展开之前，有一个所有三个文件共享、优先级高于其他所有项的问题：
**三个文件没有一处显式 `#include <hip/hip_runtime.h>` 或任何 HIP 头文件，
却在宿主代码里直接写字面量的 `cuda*` API 名字（`cudaMalloc`、`cudaMemcpy`、
`cudaError_t`、`cudaEventCreate` 等），完全依赖 nvcc 对 `.cu` 文件的隐式头
文件注入。**

这在 nvcc 下能编译，是因为 nvcc 会自动注入 `cuda_runtime.h`；但 hipcc 面向
AMD/HCC 后端时，是否会自动让 `cuda*`（而不是 `hip*`）这些名字可用，**审查
时无法验证**。如果不能，三个文件会在 DCU 上**编译直接失败**，比下面任何一
条 warpSize/bank conflict 问题都致命。这条风险贯穿检查项 5、6，作为独立发
现列在表格最前面。

> **[真机验证更新]** 已在 DTK 22.10.1 / gfx906 上确认，分三点：
>
> 1. 确实需要显式 `#include <hip/hip_runtime.h>`——不加的话，包括
>    `float4` 在内的所有符号都无法解析。hipcc 不会像 nvcc 那样为 `.cu`
>    文件隐式注入任何头文件。
> 2. 即使加了这个 include，`cuda*` 符号也**不会**被自动映射成 `hip*`——
>    `cudaError_t`/`cudaMalloc`/`cudaSuccess`/`cudaFree`/
>    `cudaGetErrorString` 等在真机编译时全部报"未声明"，编译器提示应改用
>    对应的 `hip*` 名字，需要手写符号映射。
> 3. 用来判断"该不该 include hip_runtime.h"的条件编译宏，最初尝试的
>    `__HIP_PLATFORM_HCC__`（以及后来怀疑的新名字 `__HIP_PLATFORM_AMD__`）
>    **都不能用**——这两个宏是 `hip_runtime.h` **自己在被 include 之后才
>    定义出来的**，用它们来判断"要不要 include 这个头文件"是先有鸡还是
>    先有蛋的死循环，永远不会触发，会静默落到 `#else`（CUDA）分支，导致
>    尝试 include 一个 HIP 环境下不存在的 `cuda_runtime.h`。正确做法是用
>    **`__HIPCC__`**——这是 hipcc **编译器驱动本身**预定义的宏（类似 nvcc
>    的 `__CUDACC__`），在任何头文件被处理之前就已经确定，不存在这个先后
>    顺序问题。
>
> 以上三点已在 `kernels/common/gpu_compat.h` 中落地，并在真机验证通过
> （见下方"真机验证记录"一节）。

---

## 审查结果汇总表（按风险等级降序）

| 文件:行号 | 问题描述 | 风险等级 | 当前处理方式 | 建议动作 |
|---|---|---|---|---|
| **gemm.cu 全文件**（如 182,249-253,260-275,285-287,322-336,347）<br>**softmax.cu 全文件**（如 93-99,152-177,219-231）<br>**flash_attn.cu 全文件**（如 186-192,269-311,349-367,377） | 所有 `cuda*` API（Malloc/Memcpy/Free/Event*/GetLastError/DeviceSynchronize/Memset 等）都是字面量直写。三个文件各自维护一份重复的 `check()` 辅助函数（gemm.cu:182、softmax.cu:93、flash_attn.cu:186），均以 `cudaError_t`/`cudaSuccess`/`cudaGetErrorString` 为参数。 | **已解决** | 三个文件顶部各加一行 `#include "../common/gpu_compat.h"`；该头文件用 `#ifdef __HIPCC__` 路由，在 HIP 路径下 `#include <hip/hip_runtime.h>` 并把三个文件实际用到的全部 19 个 `cuda*` 符号 `#define` 成对应的 `hip*` 名字，CUDA 路径下改成显式 `#include <cuda_runtime.h>`（原来隐式依赖 nvcc 注入）。方案细节见下方"建议 1"（已更新为真机验证过的最终版本） | GEMM 已在真机 (gfx906, DTK 22.10.1) 验证**编译 + 运行成功**，double 累加器精度结果与本地 CUDA bit-for-bit 一致，见下方"真机验证记录"。softmax.cu / flash_attn.cu 引用同一份 `gpu_compat.h`，理论上应同样生效，但**尚未在真机上单独跑通**，建议后续补验证 |
| flash_attn.cu:197-205 | `configure_smem()`：`cudaFuncSetAttribute(flash_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`，且触发阈值硬编码为 NVIDIA Volta+ 专属的 `48*1024` 字节 | **高** | 只有 CUDA 分支，无 HIP 分支 | 见下方"建议 2"；另外这里有个**容量风险**：extreme 档 (seq_len=4096, head_dim=128) 需要 `2×64×128×4=65536` 字节（恰好 64 KiB），如果 gfx906 每个 workgroup 的 LDS 总预算就是 64 KiB（无 NVIDIA 式的"48KiB默认+opt-in"分层），这个 shape 在 DCU 上可能连启动都启动不了，需要真机验证后决定是否要降低 BLOCK_SIZE |
| tests/stress_test.py:126<br>tests/benchmark.py:111 | `cmd = ["nvcc", "-O2", "-arch=sm_120", "-o", out_base, SRC[op]]`，硬编码 nvcc 可执行名和 CUDA 专属架构标志 `-arch=sm_120`（本机 RTX 5090 的 compute capability，对其他 CUDA 卡也不通用，更不用说 DCU） | **高** | 只识别 nvcc，无 hipcc 分支/检测逻辑 | 见下方"建议 3" |
| gemm.cu:37-51（注释） | Bank conflict 分析（B-fragment `sB[kk][threadIdx.x*TN]` 的 2-way 冲突）是按 **32 个 bank + 32 线程一个 warp** 推导的。AMD gfx906 的 LDS 同样是 32 bank（这点历史上和 NVIDIA 一致），但 **wavefront 是 64 而不是 32**——同一条 shared memory 指令里竞争这 8 个不冲突槽位的线程数会翻倍（16 个不同地址 × 2 份 wavefront-内重复，而不是 warp 情形下的 1 份），结论很可能从"2-way 冲突"恶化成"4-way 冲突"，而不是保持不变 | **中**（不影响正确性，但 dcu_perf.md 里记录的"保留冲突更快"这个结论**只在 CUDA 平台验证过**，不能直接套用到 DCU） | 结论写在注释/skill文档里，没有平台标注 | 在 `gemm.cu:37-51` 的注释末尾加一句"以上结论基于 warp=32 的 CUDA 实测，DCU (wavefront=64) 需要重新验证"；到真机后重新跑一遍 dcu_perf.md §3 那种 A/B 实测，不要假设结论照搬 |
| flash_attn.cu:118-128 | K/V tile 的 float4 向量化写入（`sK[row*d+col0]`/`sV[row*d+col0]`）存在和 GEMM B-load 结构相同的 bank conflict（每个 kv-tile 触发一次），但**这个仓库里从未把它写成文字分析**（此前只分析了 GEMM 的 bank conflict，flash_attn 这处是这次审查新发现的） | **中** | 完全没有分析/注释，也没有 fallback | 到真机后补一次和 GEMM 同样方法论的分析（wavefront=64 版本），并用 benchmark.py 实测是否值得修（大概率不值得，这个 load 不在热循环里，参考 dcu_perf.md §3 的方法论） |
| gemm.cu:63-64,99,124,142,144,166,168<br>flash_attn.cu:83,124-127 | `float4`/`make_float4`/`__align__(16)`/`reinterpret_cast<float4*>` 的使用。当前"4元素对齐检查 + fallback到scalar"逻辑本身是纯 C++ 运行时布尔判断（`k_aligned`/`n_aligned`/`d_aligned`/`kv_tile_full`），**没有任何 `#ifdef`包裹，两个编译路径共用同一份逻辑**——这点是好事，不是"只在CUDA分支生效"。风险点在于 `float4`/`__align__`/`make_float4` 这些**类型和宏本身**是否在 HIP 编译路径下可用、且保持与 CUDA 相同的 16 字节自然对齐语义，这依赖同一个"是否有 HIP 头文件被正确注入"的问题（见开头的最高优先级发现） | **中** | fallback 逻辑平台无关（好），但底层类型/宏可用性未验证 | 与"建议 1"一起解决（确保 HIP 头文件被正确 include 后，这些类型/宏应自动可用，因为 HIP 的 vector_types 是刻意镜像 CUDA 设计的）；到真机后单独编译一个只含 `float4 v; v.x=1;` 的最小用例验证 |
| gemm.cu:70-77,151 | `double acc[TM][TN]` 累加器（dcu_numerics.md 记录的精度纪律）。DCU 上 double 类型本身**正确性没有风险**（IEEE 754 标准类型），但 dcu_perf.md 里记录的"GEMM 比 PyTorch 慢 47-66x"这类**具体倍数**是在 RTX 5090（消费级 Blackwell，FP64:FP32 吞吐通常在 1:32~1:64 量级）上测的。Hygon DCU 脱胎于 AMD Vega20/gfx906 架构，这代芯片公开资料显示 FP64:FP32 比例可以达到 1:2（面向 HPC 设计），如果 Hygon 这颗具体型号延续了类似比例，double 累加器在 DCU 上的相对开销可能远小于 RTX 5090 上测到的，反过来 PyTorch/rocBLAS 参照基线本身在 DCU 上的 FP64 表现也可能不同——**两边都要重测，不能沿用当前倍数** | **中**（不影响正确性，只影响"该不该继续用 double 累加器"这个性能判断的可信度） | 无任何平台相关处理（也不需要，double 是标准类型） | 到真机后用 dcu_perf.md 同样的方法（stress_test.py + benchmark.py）重新测一遍 GEMM 的 double 累加器开销，不要复用 dcu_perf.md 里 RTX5090 测出的具体倍数去做 DCU 上的决策；如果 DCU FP64 确实快很多，说明 double 累加器"几乎免费"，是好消息 |
| gemm.cu:14-18<br>softmax.cu:20-24<br>flash_attn.cu:34-38 | `WARP_SIZE` 已经用 `#ifdef __HIP_PLATFORM_HCC__` 正确区分 32/64，**但三个文件里这个常量都没有被实际使用**（此前编译时 nvcc 一直报 "declared but never referenced" 警告）。搜索了 `__shfl_sync`、warp 级 reduce/broadcast、以 32/64 为边界的循环或位运算掩码（`&31`、`>>5` 等）——**三个 kernel 都不存在**：softmax 的树规约用的是全 block `__syncthreads()`（softmax.cu:57,75，每步都同步，不依赖 warp-synchronous 假设），GEMM/FlashAttn 也没有任何 warp 级原语 | **低** | 已有 `#ifdef` 分支，只是常量当前用不上 | 无需改动；如果未来给 softmax/flash_attn 加 warp-level reduction（常见优化方向）要用这个常量而不是字面量 32/64 |
| kernels/gemm/validate_gemm.py:34<br>kernels/softmax/validate_softmax.py:35<br>kernels/flash_attention/validate_flash_attn.py:46 | 三个 validate 脚本本身**不编译任何东西**，只在二进制缺失时打印一条提示："Compile first: nvcc -O2 -o {_base} {_base}.cu"，字面量写死了 nvcc | **低** | 只是提示文案，不影响实际编译/运行 | 提示文案改成根据当前平台建议 `hipcc` 或 `nvcc`（纯文案改动，不涉及功能） |

---

## 高风险项的建议写法（仅供参考，未落地代码）

### 建议 1：统一的 CUDA/HIP 符号兼容层 —— ✅ 已实现并真机验证通过

在三个文件共用一个小的兼容头 `kernels/common/gpu_compat.h`（三个 `.cu`
文件顶部各 `#include "../common/gpu_compat.h"` 一次），思路是**保留现有
代码里的 `cuda*` 名字不变**，让宏在 HIP 编译路径下把它们映射到 `hip*`。

以下是真机验证通过的最终版本（与仓库里 `kernels/common/gpu_compat.h` 保持
同步；关键点是用 `__HIPCC__` 路由，原因见开头"最高优先级发现"的更新）：

```cpp
// kernels/common/gpu_compat.h
#pragma once

// Route on __HIPCC__ (hipcc 编译器驱动预定义的宏), 不要用
// __HIP_PLATFORM_HCC__/__HIP_PLATFORM_AMD__ —— 那两个宏是 hip_runtime.h
// 自己在被 include 之后才定义出来的，用来判断"要不要 include 它"是
// 死循环，真机上永远不会触发。
#ifdef __HIPCC__
  #include <hip/hip_runtime.h>

  // --- 已通过 test_min.cu 在真机上逐个确认 ---
  #define cudaError_t                                  hipError_t
  #define cudaSuccess                                  hipSuccess
  #define cudaGetErrorString                            hipGetErrorString
  #define cudaMalloc                                    hipMalloc
  #define cudaFree                                      hipFree

  // --- 同样的映射模式，随 GEMM 一起在真机验证通过（见"真机验证记录"），
  //     softmax.cu / flash_attn.cu 尚未单独逐条确认 ---
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
```

真机验证确认：DTK 22.10.1 的 `hip_runtime.h` **没有**内置 `cuda*→hip*` 的
兼容宏（真机编译报 `cudaMalloc` 等符号"未声明"），所以上面这份手写映射
是必须的，不存在宏重复定义冲突的问题。列表覆盖三个 kernel 文件里实际用到
的全部 19 个 `cuda*` 符号（grep 出来的，不是假设的完整 CUDA API 列表）。

### 建议 2：`configure_smem` 的平台分支

```cpp
static void configure_smem(size_t smem_bytes)
{
#ifdef __HIP_PLATFORM_HCC__
    // AMD/DCU 的 LDS 是单一预算，没有 NVIDIA 那种"默认48KiB+opt-in"两级模型；
    // 48*1024 这个阈值在这里没有意义。改成查询设备真实上限。
    // TODO 真机验证: hipFuncAttributeMaxDynamicSharedMemorySize 是否存在、
    // 语义是否和 CUDA 一致；sharedMemPerBlock 是否就是 gfx906 的 64KiB LDS。
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "get device properties");
    if (smem_bytes > (size_t)prop.sharedMemPerBlock) {
        fprintf(stderr, "requested shared memory %zu bytes exceeds device limit %d bytes\n",
                smem_bytes, prop.sharedMemPerBlock);
        exit(1);
    }
    check(cudaFuncSetAttribute(flash_attn_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem_bytes),
          "set max dynamic shared memory");
#else
    if (smem_bytes > 48 * 1024) {
        check(cudaFuncSetAttribute(flash_attn_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes),
              "set max dynamic shared memory");
    }
#endif
}
```
（这里假设建议 1 的兼容层已经生效，所以两个分支里都还是写 `cudaXxx`；如果
不采用建议 1，HIP 分支需要手动换成 `hipXxx`。）

### 建议 3：编译脚本识别 hipcc

`tests/stress_test.py` 和 `tests/benchmark.py` 里各自的 `compile_kernels()`
（当前独立重复实现，两处都要改）：

```python
import shutil

def _pick_compiler():
    if shutil.which("hipcc"):
        # --offload-arch=gfx906 具体标志名需要用真实 DTK 版本核对
        return ["hipcc", "-O2"]
    if shutil.which("nvcc"):
        return ["nvcc", "-O2", "-arch=sm_120"]  # sm_120 仅对本机 RTX5090 成立
    return None

compiler_cmd = _pick_compiler()
...
cmd = [*compiler_cmd, "-o", out_base, SRC[op]]
```

`-arch=sm_120` 本身也不该硬编码（对其他 CUDA 卡不通用），建议长期改成从
`torch.cuda.get_device_capability()` 动态生成，但这是次要问题，不阻塞 DCU
部署。

---

## 真机验证记录

**环境**：海光 DCU，gfx906/gfx926，DTK 22.10.1，`hipcc` 编译。

**验证对象**：`kernels/gemm/gemm.cu`（通过 `kernels/common/gpu_compat.h`），
测试数据 M=K=N=64，随机矩阵（`torch.manual_seed(42)` 生成，与
`validate_gemm.py`/`kernels/gemm/test_data/generate_test_data.py` 同一套
生成逻辑）。

**结果**：

| 步骤 | 结果 |
|---|---|
| `hipcc` 编译 `gemm.cu`（含 `gpu_compat.h`） | ✅ 成功，无需额外修改 |
| 运行（`run_gemm.sh` → `./gemm/gemm 64 64 64 ...`） | ✅ exit 0 |
| 输出精度 vs. 本地 CUDA (nvcc) 结果 | ✅ **bit-for-bit 一致** |

精度结果 bit-for-bit 一致，说明：
- `double` 累加器在 gfx906 上数值行为与 NVIDIA 平台一致（预期之内，
  IEEE 754 标准类型，不依赖厂商实现）。
- `float4` 向量化 load/store（含对齐 fallback 逻辑）在 HIP 路径下产出的
  结果与 CUDA 路径完全一致，前一版报告里"类型/宏可用性未验证"这一条对
  GEMM 而言可以视为已解决。

**这一轮只验证了正确性，以下几项仍然是"待验证"，结论不变**（未重新测量，
不要误读为已解决）：

- gemm.cu:37-51 的 bank conflict 分析（wavefront=64 下是否从 2-way 恶化
  成 4-way）—— 本轮未测。
- flash_attn.cu 的 K/V tile bank conflict —— 本轮未测（flash_attn.cu 也
  未在真机上单独跑过）。
- `double` 累加器在 DCU 上的**相对性能开销**（dcu_perf.md 里 RTX5090 测
  出的"GEMM 比 PyTorch 慢 47-66x"这类具体倍数）—— 本轮只验证了正确性，
  没有跑 benchmark.py，性能倍数需要单独重测，不能因为正确性通过了就假设
  性能结论也成立。
- `softmax.cu` / `flash_attn.cu` 尚未在真机上单独编译运行过，虽然共用同
  一份 `gpu_compat.h`，理论上应该同样能编译通过，但没有实测确认。
- `flash_attn.cu` 的 `configure_smem()`（`cudaFuncSetAttribute` 的 HIP
  分支，即"建议 2"）尚未实现/验证 —— extreme 档 (4096,128) 需要 64 KiB
  共享内存这件事，在真机上是否会启动失败，仍然未知。

---

## 拿到真机后的优先级建议

1. 先编译一个只有 `cudaMalloc`+`check()` 的最小 `.cu` 文件（不含 kernel），
   确认"cuda* 符号能否在 hipcc/HCC 后端直接解析"这个最根本的问题——这决定
   了要不要采用建议 1。
2. 确认后，三个文件套用建议 1 的兼容层，先让**编译**通过。
3. 单独跑 `flash_attn.cu` 的 extreme 档 (4096,128)，确认 64 KiB 共享内存
   请求在真实 LDS 预算下不会启动失败。
4. 全部通过 `stress_test.py` 正确性验证后，再跑 `benchmark.py`，重新产出
   DCU 版本的性能数字——**不要复用 dcu_perf.md 里 RTX5090 的具体倍数**，
   尤其是 bank conflict 结论和 double 累加器开销这两项。
