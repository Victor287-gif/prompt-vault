---
type: API
title: HIVM 方言总览（hir 前缀、操作分类与核心属性枚举）
description: HIVM（Hybrid Intelligence Virtual Machine）方言的定位、操作基类、DMA/向量/宏/同步/Custom/内建六大操作分类，以及
  Address Space、PIPE、Core Type 三组核心枚举。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 方言总览

HIVM（Hybrid Intelligence Virtual Machine）是 BiShengIR 编译栈中承载 Ascend NPU 硬件映射的核心 IR 方言，处于硬件无关 IR（Triton/HFusion）与最终 LLVM/CCE 指令之间的桥梁层。方言名 `hivm`，C++ 命名空间 `::mlir::hivm`，所有操作以 `hir.` 为前缀（IR 中写作 `hivm.hir.<op>`）。分析 `last_pass.mlir` 时见到的 `hivm.hir.*` 操作均出自本方言，提取流程见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。

## 方言定义要素

- **操作基类**：`HIVM_Op`（所有操作基类，前缀 `hir.`）；`HIVM_StructuredOp`（结构化操作基类，实现 HIVMStructuredOpInterface / MemoryEffectsOpInterface / FlattenInterface / LibraryFunctionOpInterface）。
- **依赖方言**：arith、math、memref、memref_ext、tensor、hacc（硬件加速器配置，目标设备规格见 [hardware_specs_ir](../reference/hardware_specs_ir.md)）。
- **TableGen 定义文件**：`bishengir/include/bishengir/Dialect/HIVM/IR/` 下的 HIVMBase.td（基类）、HIVMAttrs.td（属性枚举）、HIVMDMAOps.td / HIVMVectorOps.td / HIVMMacroOps.td / HIVMSynchronizationOps.td / HIVMOps.td / HIVMIntrinOps.td（各类操作）。

## 操作分类速查

| 类别 | 定义文件 | 代表操作 | 详解卡 |
|------|---------|---------|--------|
| DMA 数据搬运 | HIVMDMAOps.td | hir.load / hir.store / hir.copy / hir.fixpipe / hir.nd2nz / hir.nz2nd / hir.l12ub / hir.atomic_cas / hir.atomic_xchg | [dma_overview.md](dma_overview.md) |
| 向量计算 | HIVMVectorOps.td | hir.vadd / hir.vmul / hir.vcast / hir.vcmp / hir.vreduce / hir.vbrc 等 40+ | [vector_overview.md](vector_overview.md) |
| 宏操作（矩阵乘） | HIVMMacroOps.td | hir.mmadL1 / hir.batchMmadL1 / hir.matmul / hir.mix_matmul / hir.mix_group_matmul | [macro_overview.md](macro_overview.md) |
| 同步控制 | HIVMSynchronizationOps.td | hir.set_flag / hir.wait_flag / hir.pipe_barrier / hir.sync_block 系列 | [sync_overview.md](sync_overview.md) |
| Custom / 基础 | HIVMOps.td | hir.get_block_idx / hir.set_atomic / hir.dcci / hir.pointer_cast / hir.gather_load / hir.scatter_store / hir.indirect_load / hir.indirect_store / hir.debug 等 | [dma_gather_scatter.md](dma_gather_scatter.md)、[dma_indirect_access.md](dma_indirect_access.md) |
| 内建指令 | HIVMIntrinOps.td | hir.intr.hivm.SET.FLAG.IMM / WAIT.FLAG.IMM / BARRIER / DCCI.DST 等 | [intrin_overview.md](intrin_overview.md) |

## 核心枚举

### Address Space（地址空间，#hivm.address_space<...>）

| IR 字面量 | 含义 |
|-----------|------|
| `gm` | 全局内存（片外） |
| `cbuf` | L1 缓存（Cube Buffer） |
| `ca` / `cb` / `cc` | L0A / L0B / L0C（Cube 矩阵输入与累加器） |
| `ub` | 统一缓冲区（Vector Buffer） |
| `zero` | 默认地址空间 |

### PIPE（流水线）

| 枚举值 | 含义 |
|--------|------|
| `PIPE_S` | Scalar 流水线 |
| `PIPE_V` | Vector 流水线 |
| `PIPE_M` | Cube 流水线 |
| `PIPE_MTE1` | L1 相关搬运（如 L1→UB） |
| `PIPE_MTE2` | GM→本地搬运（load / nd2nz） |
| `PIPE_MTE3` | 本地→GM 搬运（store / nz2nd） |
| `PIPE_FIX` | L0C 搬出（fixpipe） |
| `PIPE_MTE4` / `PIPE_MTE5` | 保留扩展 |

### Core Type（核心类型）

| 枚举值 | 含义 |
|--------|------|
| `CUBE` | Cube 核心（矩阵计算） |
| `VECTOR` | Vector 核心（向量计算） |
| `CUBE_OR_VECTOR` / `CUBE_AND_VECTOR` | 二选一 / 混合（Mix）模式 |

硬件层级背景见 [memory_hierarchy](../architecture/memory_hierarchy.md) 与 [pipeline_execution_model](../architecture/pipeline_execution_model.md)。

## 编译栈中的位置

```
Triton Python API
    → Triton Dialect (tt)      前端语义（tt.load / tt.dot）
    → HFusion Dialect          融合与 Tiling
    → HIVM Dialect (hir)       硬件映射：DMA、Pipeline 分配、同步插入、内存层级标注
    → LLVM / CCE 指令           库函数调用、内建指令
    → Ascend NPU 二进制
```

HIVM 阶段完成四件事：数据搬运映射为具体 DMA 操作、计算分配到 Cube/Vector 核心、确定 Pipeline 归属并插入同步、显式标注 GM/L1/UB/L0A/L0B/L0C 内存层级。阅读 HIVM IR 做性能分析的方法见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMBase.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td)
- [HIVMBase.td#L33-L37](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L33-L37)
- [HIVMBase.td#L52-L68](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td#L52-L68)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMVectorOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td)
- [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
- [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td)
- [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td)
- [HIVMIntrinOps.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td)
- [HIVMAttrs.td#L170-L197](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L170-L197)
- [HIVMAttrs.td#L203-L244](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L244)
- [HIVMAttrs.td#L298-L317](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L317)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR.git/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td)
