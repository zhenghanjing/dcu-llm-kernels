---
name: dcu-hip-porting
description: CUDA (nvcc) kernel 移植到海光 DCU (hipcc/HIP, gfx906) 时踩过的坑——符号兼容层怎么写、平台判断宏该用哪个、核函数指针传给 xxxFuncSetAttribute 的隐式转换陷阱、DCU 共享内存/LDS 预算模型跟 CUDA 的差异、以及没有 torch 环境的真机验证方法论。把 kernel 移植到 DCU、或新写要同时跑在 CUDA/HIP 两个平台的 kernel 时应参考本文档。
---

# CUDA → HIP/DCU 移植经验

本文档记录本仓库把 `kernels/gemm/gemm.cu`、`kernels/softmax/softmax.cu`、
`kernels/flash_attention/flash_attn.cu` 从 NVIDIA/nvcc 移植到海光 DCU
(gfx906/gfx926, DTK 22.10.1, hipcc) 过程中遇到的编译期/运行期问题，已验证
的解决方案，以及真机验证的方法论。适用范围：任何要同时兼容 CUDA 和 HIP
两条编译路径的 kernel（新写的，或从 CUDA 移植过来的）。

## 1. `cuda*` 符号在 hipcc 下不会隐式可用

**现象**：三个 kernel 文件此前从未显式 `#include` 任何 HIP/CUDA 头文件，
直接在宿主代码里写字面量的 `cuda*` API 名字（`cudaMalloc`、`cudaMemcpy`、
`cudaError_t`、`cudaEventCreate` 等）。这在 nvcc 下能编译，因为 nvcc 会给
`.cu` 文件隐式注入 `cuda_runtime.h`。真机第一次用 hipcc 编译时：即使显式
加了 `#include <hip/hip_runtime.h>`，`cudaError_t`/`cudaMalloc`/
`cudaSuccess`/`cudaFree`/`cudaGetErrorString` 等仍然报"未声明"，编译器
提示应改用对应的 `hip*` 名字。

**根因**：两层问题叠加——① hipcc 不会像 nvcc 那样为 `.cu` 文件隐式注入
任何头文件，不加 `hip_runtime.h` 的话连 `float4` 这种向量类型都无法解析；
② 即使加了这个 include，DTK 22.10.1 的 HIP 运行时库本身**不提供**
`cuda*→hip*` 的兼容宏，两边是完全独立的符号集合，需要手写映射。

**解决方案**：建一个 `kernels/common/gpu_compat.h`，三个 `.cu` 文件顶部
各 `#include "../common/gpu_compat.h"` 一次。映射表**不是**照抄一份假设的
完整 CUDA API 列表，而是对三个 kernel 文件 `grep cuda` 找出实际引用到的
符号——最终是 19 个，多一个不多、少一个不少：

```cpp
// kernels/common/gpu_compat.h
#ifdef __HIPCC__
  #include <hip/hip_runtime.h>

  // --- 已通过 test_min.cu 在真机上逐个确认 ---
  #define cudaError_t                                  hipError_t
  #define cudaSuccess                                  hipSuccess
  #define cudaGetErrorString                            hipGetErrorString
  #define cudaMalloc                                    hipMalloc
  #define cudaFree                                      hipFree

  // --- 同样的映射模式，随三个 kernel 一起在真机验证通过 ---
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
  #define cudaGetDevice                                  hipGetDevice
  #define cudaDeviceProp                                 hipDeviceProp_t
  #define cudaGetDeviceProperties                        hipGetDeviceProperties
#else
  #include <cuda_runtime.h>
#endif
```

好处是宿主代码（`check()`、`cudaMalloc`/`cudaMemcpy` 调用点等）在两个平台
下完全不用改字面量，只需要保证映射表跟着新用到的符号增量更新——比如后来
`flash_attn.cu` 的 `configure_smem()` 用到 `cudaGetDevice`/`cudaDeviceProp`/
`cudaGetDeviceProperties` 时（见第 4 条），是在这个头文件里补三行，而不是
一次性预判所有可能用到的 CUDA API。

**验证结果**：`gemm.cu`/`softmax.cu`/`flash_attn.cu` 三个文件均在真机
(gfx906/gfx926, DTK 22.10.1) 编译成功，输出与本地 CUDA (nvcc) 结果
bit-for-bit 一致（GEMM/Softmax）或差异在 1e-6 量级浮点舍入范围内
（FlashAttn，符合分块 + 在线 softmax 改变累加顺序的预期）。

## 2. 平台判断宏必须用 `__HIPCC__`，不能用 `__HIP_PLATFORM_HCC__`/`__HIP_PLATFORM_AMD__`

**现象**：最初判断"该不该走 HIP 分支/该不该 include `hip_runtime.h`"用的
是 `#ifdef __HIP_PLATFORM_HCC__`（以及后来怀疑的新名字
`__HIP_PLATFORM_AMD__`）。这两个宏在真机上**永远不会触发**，代码会静默
落到 `#else`（CUDA）分支，尝试 `#include <cuda_runtime.h>`——而这个头文件
在 HIP 环境下根本不存在，导致的失败现象和"没做平台判断"看起来一样，很
容易被误诊成别的问题。

**根因（这条必须讲清楚原理，而不是只记结论——最容易被抄错）**：
`__HIP_PLATFORM_HCC__`/`__HIP_PLATFORM_AMD__` 是 **`hip_runtime.h` 自己在
被 include 之后才定义出来的宏**，不是编译器驱动预先设置的。用它们来判断
"要不要 include `hip_runtime.h`"是一个先有鸡还是先有蛋的死循环：如果这个
头文件还没被 include，这两个宏根本不存在，条件求值永远是假，于是头文件
永远不会被 include，宏也就永远不会被定义——这个分支在设计上就不可能
触发，不是运气不好或版本差异，是判断逻辑本身有循环依赖。

**解决方案**：改用 `__HIPCC__`。这是 hipcc **编译器驱动本身**预先定义的
宏（跟 nvcc 的 `__CUDACC__` 是同一类东西），在任何头文件被处理之前就已经
确定，不存在"先 include 才能判断该不该 include"的顺序问题：

```cpp
// 错误：hip_runtime.h 自己定义的宏，用来判断要不要 include 它是死循环
#ifdef __HIP_PLATFORM_HCC__
  #include <hip/hip_runtime.h>
#else
  #include <cuda_runtime.h>
#endif

// 正确：__HIPCC__ 是编译器驱动预定义的宏，不依赖任何头文件先被 include
#ifdef __HIPCC__
  #include <hip/hip_runtime.h>
#else
  #include <cuda_runtime.h>
#endif
```

**验证结果**：改用 `__HIPCC__` 路由后，`hip_runtime.h` 在真机被正确
include，三个 kernel 文件的 `WARP_SIZE`（64 vs 32）等平台相关分支也随之
正确生效，真机编译/运行全部通过。

## 3. 核函数指针传给 `cudaFuncSetAttribute`/`hipFuncSetAttribute` 必须显式 `(const void*)` 转换

**现象**：这是这一轮真机验证**新发现、此前完全没预料到**的坑。
`flash_attn.cu` 的 `configure_smem()` 里两处 `cudaFuncSetAttribute` 调用都
直接传裸核函数符号 `flash_attn_kernel`（不带任何转换），本地用 nvcc/MSVC
编译完全没有问题。但真机第一次用 hipcc 编译直接报错：

```
error: no matching function for call to 'hipFuncSetAttribute'
note: no known conversion from 'void (...)' to 'const void *' for 1st argument
```

**根因**：`cudaFuncSetAttribute`/`hipFuncSetAttribute` 的第一个参数类型
**两边都声明为 `const void*`**，不是 API 语义差异。问题在调用点：把核函数
符号（类型是核函数签名本身，例如 `void (const float*, const float*, ...)`)
直接传给一个要 `const void*` 的形参，依赖的是"函数指针隐式转 `void*`"这个
**非标准 C++ 扩展**——nvcc/MSVC 默许放行，但 hipcc 的 Clang 前端按标准
C++ 严格拒绝这种隐式转换，直接编译失败。

**解决方案**：在两个平台分支的调用点都显式转换：

```cpp
// 修复前（依赖非标准的函数指针→void*隐式转换，hipcc 拒绝）
check(cudaFuncSetAttribute(flash_attn_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem_bytes),
      "set max dynamic shared memory");

// 修复后（两个分支都要改，不要只改 HIP 分支——两条路径写法保持一致，
// 不再依赖编译器的非标准放行）
check(cudaFuncSetAttribute((const void*)flash_attn_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)smem_bytes),
      "set max dynamic shared memory");
```

**验证结果**：加上 `(const void*)` 转换后两个平台都编译通过。本地 nvcc
重新编译 + `validate_flash_attn.py` 64×64 和 128×128 两个 shape 都仍然
PASS（这个改动对 CUDA 路径的数值行为没有任何影响，纯粹是类型转换）；真机
hipcc 编译成功，`smem_risk` 用例 (seq_len=128, head_dim=128, 64 KiB 动态
共享内存) 成功触发 `hipFuncSetAttribute` 并正常执行，输出与本地 CUDA
逐位一致。**方法论**：任何"核函数符号"作为参数传给期望 `void*`/
`const void*` 的 API（不止 `xxxFuncSetAttribute`，`cudaLaunchKernel` 等同样
接受函数指针的 API 都是同类风险点）时，默认显式转换，不要依赖隐式转换在
两个编译器上都放行。

## 4. DCU 的共享内存(LDS)是单一预算模型，不能套用 CUDA 的"48KiB 默认 + opt-in"两级模型

**现象**：`configure_smem()` 原本只在 `smem_bytes > 48 * 1024` 时才调用
`cudaFuncSetAttribute` 做动态共享内存的 opt-in——这是 NVIDIA Volta+ 专属的
两级模型（每个 kernel 默认最多用 48 KiB 共享内存，超过需要显式 opt-in 到
更大的上限）。这个 48 KiB 阈值在 AMD/DCU 上没有对应的语义。

**根因**：AMD gfx906 的 LDS（Local Data Share，等价于 CUDA 的 shared
memory）是**单一固定预算**（每 workgroup 64 KiB），没有 NVIDIA 那种
"默认更小 + 显式申请更大"的两级模型。照搬 48 KiB 阈值意味着：当请求的
动态共享内存超过 64 KiB 这个真实上限时，HIP 路径完全不会报错——而是要么
静默按旧阈值判断走错分支，要么直到 kernel 启动时才失败，且失败信息不会
说明是共享内存超限。

**解决方案**：HIP 分支不再套用 CUDA 的 48 KiB 阈值判断，而是**总是**先用
`hipGetDeviceProperties().sharedMemPerBlock`（通过 `gpu_compat.h` 映射为
`cudaGetDeviceProperties`）查询设备真实上限，超限就提前 `fprintf` 报错并
`exit(1)`，不超限再调用 `cudaFuncSetAttribute`；CUDA 分支保留原来的 48 KiB
阈值判断不变：

```cpp
static void configure_smem(size_t smem_bytes)
{
#ifdef __HIPCC__
    // AMD/DCU 的 LDS 是单一预算，没有 NVIDIA 那种"默认48KiB+opt-in"两级
    // 模型；48*1024 这个阈值在这里没有意义，改成查询设备真实上限。
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "get device properties");
    if (smem_bytes > (size_t)prop.sharedMemPerBlock) {
        fprintf(stderr,
                "flash_attn: requested dynamic shared memory %zu bytes "
                "exceeds device limit %zu bytes (LDS per workgroup on this "
                "DCU) -- reduce BLOCK_SIZE or head_dim for this shape\n",
                smem_bytes, (size_t)prop.sharedMemPerBlock);
        exit(1);
    }
    check(cudaFuncSetAttribute((const void*)flash_attn_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem_bytes),
          "set max dynamic shared memory");
#else
    if (smem_bytes > 48 * 1024) {
        check(cudaFuncSetAttribute((const void*)flash_attn_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_bytes),
              "set max dynamic shared memory");
    }
#endif
}
```

**验证结果**：`smem_risk` 用例 (seq_len=128, head_dim=128) 请求 64 KiB
动态共享内存——与审查报告标记高风险的 extreme 档 (seq_len=4096,
head_dim=128) 单块共享内存占用完全相同（共享内存只取决于 `BLOCK_SIZE` 和
`head_dim`，与 `seq_len` 无关）——在真机上成功触发 `hipFuncSetAttribute`
并正常执行，未出现启动失败，输出与本地 CUDA 逐位一致。**注意**：这只验证
了 64 KiB 这个共享内存边界点，没有验证 (4096,128) 这个完整 shape 在网格
规模/显存维度上是否也没问题，两者是独立的待验证项。

## 5. 真机验证方法论：无 torch 环境下的手工比对 + 门户按模块拆分提交

**现象**：DCU 计算节点上大概率没有装 torch/完整 Python 环境，而
`tests/stress_test.py`/`tests/benchmark.py` 这类脚本是"生成随机数据 →
用 torch 算参考值 → 跑 kernel 二进制 → 用 torch 比较误差"全部在同一个
Python 进程里完成的，没法直接搬到真机跑。

**解决方案**：把"生成数据/参考值"和"跑 kernel/比对"这两半拆开，参考值
比对方式从"脚本自动 assert"降级为"打印一行 preview 人工/回传比对"：

1. 本地（有 torch 的开发机）用与线上 validate 脚本相同的
   `torch.manual_seed(42)` 生成逻辑，生成输入 `.bin` + 参考输出 `.bin` +
   一份 `ref_..._preview.txt`（前 8 个元素的文本预览，格式跟 kernel 自己
   `printf` 到 stdout 的格式一致）。
2. 把输入数据和 kernel 源码一起打包上传真机，真机只需要编译 + 跑裸二进制
   （`./gemm 64 64 64 A.bin B.bin`这种，不需要 Python/torch），把二进制
   打印到 stdout 的那一行 preview 贴回来，跟本地生成的
   `ref_..._preview.txt` 逐位或在容差内比对。
3. 不需要修改 kernel 本身的 argv 接口去"支持真机"——本仓库三个 kernel
   本来就是`./bin <shape...> <in.bin> <out.bin>`的裸二进制接口，天然适合
   这种拆分。

**验证结果**：GEMM/Softmax/FlashAttn 在 boundary/non-2^n/large 三档
（共 9 个 shape，对应 `tests/stress_test.py` 的 `TESTS` 列表）全部用这套
方法验证，9/9 通过，GEMM/Softmax 全部逐位一致，FlashAttn 仅在 1e-6 量级
浮点舍入范围内有差异（符合预期，远低于 atol=1e-3）。

**现象 2（门户本身的稳定性问题，不是代码/数据问题）**：把三个算子的 9 个
shape 打包进同一个 zip（46 个文件，~2.8MB）一次性提交真机门户时，后端
返回 `Internal Server Error`（500），且门户前端还有个独立的 JS bug（对
同一个 `Response` 调用了两次 `.text()`），一度让人误以为是请求卡住而不是
报错。

**根因**：用浏览器 DevTools 直接查看该请求的原始响应确认是后端 500，不是
前端假象。把同一份数据拆成三个单算子小包（各自 12~18 个文件，
0.55~1.1MB）分别提交后，三个全部成功——说明不是任何一个算子的数据或
`gpu_compat.h`/`configure_smem()` 代码本身的问题，而是"单次请求文件数
过多/请求过于复杂"触发了门户后端某个未知限制或 bug。

**解决方案/方法论**：
- 真机门户不稳定时，优先怀疑门户侧问题而不是自己的代码/数据，用浏览器
  DevTools 直接查看原始 HTTP 响应确认是真报错还是前端假象，不要一直等待
  或重试同一个大包。
- 大规模、多文件的测试默认按算子/模块拆分成多个小请求分别提交，不要
  指望单次大请求能稳定跑通——尤其是后续要补跑真正的 extreme 档
  （GEMM 4096³、FlashAttn seq_len=4096）这类本身文件就不小的场景，更应该
  独立、小心地单独提交。

## 参考实现

- 符号兼容层（第 1、2 条）：
  [kernels/common/gpu_compat.h](../../kernels/common/gpu_compat.h)
- `configure_smem()` 完整实现（第 3、4 条）：
  [kernels/flash_attention/flash_attn.cu](../../kernels/flash_attention/flash_attn.cu)
- 完整审查报告与真机验证记录（含踩坑时的原始报错信息）：
  [docs/dcu_portability_review.md](../../docs/dcu_portability_review.md)
- 真机手工验证用的裸二进制 wrapper 脚本（第 5 条）：
  [kernels/gemm/run_gemm_stress.sh](../../kernels/gemm/run_gemm_stress.sh)、
  [kernels/softmax/run_softmax_stress.sh](../../kernels/softmax/run_softmax_stress.sh)、
  [kernels/flash_attention/run_flash_attn_stress.sh](../../kernels/flash_attention/run_flash_attn_stress.sh)
- 精度纪律（累加器精度、验证容差）：[dcu_numerics.md](dcu_numerics.md)
- 性能优化方法论：[dcu_perf.md](dcu_perf.md)
