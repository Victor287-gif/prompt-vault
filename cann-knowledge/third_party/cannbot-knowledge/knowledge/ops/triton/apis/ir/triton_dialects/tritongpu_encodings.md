---
type: API
title: TritonGPU 布局编码详解（blocked / mma / shared / fractal）
description: TritonGPU 布局编码属性语义：分布式布局（blocked、linear、NvidiaMma、AMDMfma/Wmma、slice、dot_op）、共享布局（swizzled_shared、padded_shared、nvmma_shared、amd_rotating_shared、昇腾
  fractal_shared）、CTALayout 与布局选择策略。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.programming_model.execution_unit.cube, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# TritonGPU 布局编码详解

布局编码决定张量数据在线程间的分布方式。形式化地，布局是函数 L：多维张量索引 i → 允许访问该元素的线程索引集合 T。两大类：**分布式布局**（数据在寄存器中，每线程持有部分数据）与**共享布局**（数据在共享内存中，全线程可访问）。

## 布局编码接口

- **LayoutEncodingTrait**：所有编码共有，提供 CTA 级信息（`getCTAsPerCGA()` / `getCTAOrder()` / `getCTASplitNum()` / `getRank()`）。
- **DistributedEncodingTrait**：分布式布局扩展，描述 4 级计算层次 CTAs Per CGA → Warps Per CTA → Threads Per Warp → Values Per Thread；方法 `getRepOrder()`、`getTotalElemsPerThread(shape)`、`toLinearLayout(shape)`。
- **SharedEncodingTrait**：共享布局接口，默认对齐 16 字节。

## CTALayoutAttr

描述 CTA（线程块）在 CGA（线程组集群）中的分布。参数 `CTAsPerCGA` / `CTASplitNum` / `CTAOrder`：

- 张量按 `CTASplitNum` 分割后分配给 `CTAsPerCGA` 个 CTA，每 CTA 处理 `tensor_shape / CTASplitNum` 的子张量。
- `CTASplitNum[i] < CTAsPerCGA[i]` 时为多播（多个 CTA 持有相同数据），典型于 GEMM：A 矩阵 dim1 多播、B 矩阵 dim0 多播、C 矩阵无多播。
- `CTAsPerCGA = [1,...,1]` 时 `CTAOrder` 规范化为 `[n-1,...,0]`。

## 分布式布局编码

### BlockedEncodingAttr（`#ttg.blocked`）

连续分块布局，每个 warp 拥有目标张量的连续部分，用于促进 Load/Store 内存合并。参数：`sizePerThread` / `threadsPerWarp` / `warpsPerCTA` / `order`（最快变化轴在前）/ `CTALayout`（文本 IR 可省略，默认单 CTA）。

```mlir
#ttg.blocked<{sizePerThread = {2, 2}, threadsPerWarp = {8, 4}, warpsPerCTA = {1, 2}, CTAsPerCGA = {1, 1}, CTASplitNum = {1, 1}}>
```

### LinearEncodingAttr（`#ttg.linear`）

基于 LinearLayout 的通用布局，可表达所有其他分布式布局。参数 `$linearLayout`（register / lane / warp / block 基向量）。

### NvidiaMmaEncodingAttr（`#ttg.nvidia_mma`）

NVIDIA Tensor Core MMA 输出布局。参数 `versionMajor`（1=Volta，隐式 warpTile [16,16]；2=Turing/Ampere，[16,8]）、`versionMinor`、`warpsPerCTA`、`instrShape`、`CTALayout`。

### AMDMfmaEncodingAttr / AMDWmmaEncodingAttr

AMD 矩阵核心布局：MFMA（CDNA，参数 version / tilesPerWarp / MDim / NDim / isTransposed / elementType）；WMMA（RDNA，warp tile 固定 [16,16]）。

### SliceEncodingAttr（`#ttg.slice`）

沿维度 `$dim` 压缩父布局 `$parent` 的切片布局，用于 expand_dims 的逆操作：压缩后每个线程拥有被压缩维上全部数据。

### DotOperandEncodingAttr（`#ttg.dot_op`）

Dot 操作数布局。`tt.dot a, b, c` 的 a、b 在 MMA v1/v2 下必须用此布局（MMA v3 操作数通常用共享布局）。参数：`opIdx`（0=A，1=B）、`parent`（Dot 结果布局）、`kWidth`（K 维每线程连续元素数；Ampere/Hopper 上 `kWidth = max(32/bitwidth, 1)`，其他 MMA 为 0）。

## 共享布局编码

### SwizzledSharedEncodingAttr（`#ttg.swizzled_shared`）

XOR Swizzle 避免银行冲突。参数 `vec`（向量化宽度）/ `perPhase`（每相位行数）/ `maxPhase` / `order` / `CTALayout`。元素 (r, c) Swizzle 后列坐标：`(c / vec) ^ (r / perPhase) % maxPhase`。

### PaddedSharedEncodingAttr（`#ttg.padded_shared`）

填充避免银行冲突。参数 `intervals` / `paddings` / `order`。1D 索引 i 的偏移：`offset = i + Σ(i / interval_k) * pad_k`。辅助方法 `getPaddedSize(shape)`。

### NVMMASharedEncodingAttr（`#ttg.nvmma_shared`）

NVIDIA MMAv3/v5（WGMMA）共享内存输入布局。参数 `swizzlingByteWidth`（0/32/64/128，按连续维度字节数规则选定）/ `transposed` / `elementBitWidth` / `fp4Padded`。

### AMDRotatingSharedEncodingAttr

旋转 Swizzle：每 `maxPhase*perPhase` 行后 Swizzle 模式旋转（`combinedPhase = phase ^ blockNo`），而非简单重复。

### FractalSharedEncodingAttr（`#ttg.fractal_shared`，昇腾适配）

昇腾 Cube 仿真的自定义 Fractal 共享内存布局（条件编译 `BSPRIV_DAVINCI_BISHENGIR`）。参数：`fractalM0` / `fractalN0`（Fractal 块大小）、`layoutType`、`CTALayout`。

FractalLayoutType 枚举：

| 枚举值 | 说明 |
|--------|------|
| `zN` | 内部 Z 形（列先行后），外部 N 形（M 块先 N 块后） |
| `nZ` | 内部 N 形（行先列后），外部 Z 形（N 块先 M 块后） |

该编码实现 `SharedEncodingTrait` 与 `LayoutEncodingTrait`，匹配 DaVinci Cube 矩阵格式，是昇腾路径上 Cube 操作数的布局选择（Ascend 内存层级背景见 [memory_hierarchy](../architecture/memory_hierarchy.md)）。

## SharedMemorySpaceAttr

助记符 `shared_memory`，标记内存描述符指向共享内存空间。

## 布局选择策略与转换路径

| 场景 | 推荐布局 | 原因 |
|------|----------|------|
| 全局内存 Load/Store | blocked | 促进合并访问 |
| Dot 操作数（MMA v1/v2） | dot_op | 匹配矩阵核心输入格式 |
| Dot 结果 | nvidia_mma / amd_mfma | 匹配矩阵核心输出格式 |
| 共享内存中间存储 | swizzled_shared / nvmma_shared | 避免银行冲突 |
| 昇腾 Cube 操作数 | fractal_shared | 匹配 DaVinci Cube 矩阵格式 |

典型转换路径：`BlockedEncoding (Load) → DotOperandEncoding (Dot 输入) → NvidiaMmaEncoding (Dot 输出) → BlockedEncoding (Store)`，每步由 `ttg.convert_layout` 完成，实现方式可能是寄存器重排（同 warp 内）、共享内存中转（跨 warp）或空操作（布局兼容时）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [TritonGPUAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)
