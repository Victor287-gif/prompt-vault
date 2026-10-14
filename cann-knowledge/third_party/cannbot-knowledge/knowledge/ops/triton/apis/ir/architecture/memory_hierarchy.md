---
type: API
title: NPU 内存层次详解（GM/L1/L0A/L0B/L0C/UB、数据通路、随路操作）
description: Ascend NPU 各级存储的容量与对齐、AddressSpace 枚举、源-目标地址空间到 Pipeline 的映射、Cube/Vector
  数据通路、随路操作（padding/量化/转置/原子）与紧耦合缓冲区机制。
tags: [task.lookup, topic.architecture, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']

---

# 内存层次详解

NPU 的各级存储与计算单元、执行流水线紧密绑定，数据搬运由专用 DMA 引擎（MTE 系列）完成，而非统一缓存自动管理。在 HIVM IR 中，每个 `memref` 必须用 `#hivm.address_space<...>` 标注所在存储空间，每次搬运必须使用正确的 Pipeline，跨空间的通路有严格硬件约束。

## 存储空间与 AddressSpace 枚举

| 枚举值（数值） | IR 标识符 | 硬件存储 | 所属计算单元 |
|----------------|-----------|----------|--------------|
| Zero(0) | `zero` | — | 默认/未指定 |
| GM(1) | `gm` | HBM/L2 全局内存 | 所有单元共享 |
| L1(2) | `cbuf` | L1 Cache | Cube |
| L0A(3) | `ca` | L0A Buffer | Cube A 端（矩阵 A 输入） |
| L0B(4) | `cb` | L0B Buffer | Cube B 端（矩阵 B 输入） |
| L0C(5) | `cc` | L0C Buffer | Cube C 端（乘法结果） |
| UB(6) | `ub` | UB 统一缓冲区 | Vector |

```mlir
memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
memref<256x256xf16, #hivm.address_space<gm>>
```

专用硬件缓冲区：BT Buffer（BiasTable，1KB/64B 对齐，经 `copy_cbuf_to_bt` 从 L1 加载）、FP Buffer（FixPipe 中间缓冲，7KB/128B，`hivm.fixpipe` 隐式使用）。

## 容量与对齐（各系列）

| 空间 | 910B/910_93 | 310B | 910_95/950PR/950DT | 对齐（全架构一致） |
|------|-------------|------|---------------------|---------------------|
| UB | 192KB | 256KB | 248KB（预留 8KB） | 32B |
| L1 | 512KB | 1024KB | 512KB | 32B |
| L0A / L0B | 64KB | 64KB | 64KB | TableGen 未显式定义 |
| L0C | 128KB | 128KB | 256KB | 512B |
| DCache | 无 | 无 | 32~120KB（SIMT 可配） | — |

## 源-目标地址空间 → Pipeline → IR 操作

| 源 → 目标 | Pipeline | IR 操作 | 说明 |
|-----------|----------|---------|------|
| GM → L1 | MTE2 | `hivm.nd2nz` | 支持 ND→NZ 随路转换 |
| GM → UB | MTE2 | `hivm.load` | 支持 Padding 随路操作 |
| L1 → GM | MTE2 | `copy_cbuf_to_gm` | 支持 NZ→ND |
| L1 → L0A/L0B/BT | MTE1 | 内部指令 / `copy_cbuf_to_bt` | Cube 输入加载 |
| L1 → UB | MTE1 | `hivm.l12ub` | NZ→ND 转换 |
| L0A/L0B → L0C | M | Cube 计算 | 矩阵乘法 |
| L0C → GM/L1/UB | FIX | `hivm.fixpipe` | L0C→UB 仅 950 系列 |
| UB → UB | V | `hivm.copy` | UB 内复制 |
| UB → GM | MTE3 | `hivm.store` | 支持原子操作 |
| UB → L1 | MTE3 | `hivm.copy` | 仅 950 系列 |

`hivm.copy` 的 Pipe 按地址空间动态推断：UB→UB 为 PIPE_V，L0C→GM 为 PIPE_FIX，GM→L1 为 PIPE_MTE2，UB→L1 为 PIPE_MTE3。（Pipe 推断规则详见 [pipeline_execution_model.md](pipeline_execution_model.md)）

## 数据通路

Cube 通路：
```
GM ─[MTE2]→ L1 ─[MTE1]→ L0A/L0B/BT ─[M]→ L0C ─[FIX]→ GM/L1/UB
```

Vector 通路：
```
GM ─[MTE2]→ UB ─[V]→ UB ─[MTE3]→ GM
```

950 特有通路：
- **L0C → UB 直通**：Cube 结果经 FixPipe 直接进 UB，Cube-Vector 混合计算免 GM 中转（非 950 必须 L0C→GM→UB 绕行）。
- **UB → L1**：Vector 处理结果可回送 L1 供后续 Cube 使用（PIPE_MTE3）。

## 随路操作（DMA 硬件顺带完成，无需额外计算指令）

| Pipeline | 流向 | 随路操作 | IR 属性/操作 |
|----------|------|----------|--------------|
| MTE1 | L1→L0A/L0B | 矩阵转置 | `a_transpose`/`b_transpose`（mmadL1 内） |
| MTE1 | L1→L0A/L0B | 布局转换 | zN↔nZ 互转 |
| MTE2 | GM→L1 | ND→NZ | `hivm.nd2nz` |
| MTE2 | L1→GM | NZ→ND | `hivm.nz2nd` |
| MTE2 | GM→UB | Padding | `pad_mode`/`pad_value`/`left/right_padding_num`（hivm.load） |
| MTE2/MTE3 | GM↔UB | 隐式转置 | `may_implicit_transpose_with_last_axis` |
| MTE3 | UB→GM | 原子操作 | `atomic_kind`：add/max/min/and/or/xor/CAS/XCHG |
| FIX | L0C→GM/L1/UB | 预量化 | `pre_quant`：FP32→FP16/BF16、INT32→INT8 |
| FIX | 同上 | 预激活 | `pre_relu`：ReLU/LeakyReLU/PReLU |
| FIX | 同上 | 布局转换 | `dma_mode`：NZ2ND/NZ2DN/NZ2NZ |
| FIX | L0C→UB | 双目标 | `dual_dst_mode`：ROW_SPLIT/COLUMN_SPLIT（仅 950） |
| FIX | L0C→GM/L1/UB | Channel Split | `channel_split`（仅 950） |

## 紧耦合缓冲区（TightlyCoupledBuffer）

950 特有的 Cube-Vector 通信机制（`#hivm.tightly_coupled_buffer<id>`），由 `InsertCVTightCoupledBuffer` Pass 在 Fixpipe 与 Vector 之间插入，支持 MoveToUb（L0C→UB）与 MoveToL1（UB→L1）两种搬运，避免 GM 中转。Pass 选择逻辑：

```
if (isAscend950)  → enableLayoutOptimization ? InsertCVDataMovement : InsertCVTightCoupledBuffer
else              → InsertLoadStoreForMixCV   // L0C→GM→UB 绕行
```

编译器侧的插入与内存复用详见 [tightly_coupled_buffer](../memory_management/tightly_coupled_buffer.md)；Pipeline 视角见 [pipeline_execution_model.md](pipeline_execution_model.md)；地址空间在内存规划中的使用见 [overview](../memory_management/overview.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [NPUTargetSpec.td#L64-L74](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L64-L74)
- [NPUTargetSpec.td#L156-L166](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L156-L166)
- [NPUTargetSpec.td#L196-L208](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L196-L208)
- [HIVMAttrs.td#L171-L197](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L197)
- [HIVMDMAOps.cpp#L616-L622](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L616-L622)
- [HIVMAttrs.td#L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)
- [HIVMDMAOps.cpp#L440-L448](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L440-L448)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
