---
type: API
title: 内存对齐 Memory Alignment（对齐 Pass、stride 对齐属性、硬件对齐约束）
description: NPU 硬件对齐约束的编译期保障：hivm-align-alloc-size / mark-stride-align / enable-stride-align
  / lift-lowest-stride 四个 Pass、stride_align_dims 等标注属性、UB/L1/L0C 对齐数值。
tags: [task.lookup, topic.implementation.memory_management, topic.alignment, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# 内存对齐 — Memory Alignment

NPU 硬件对内存访问有严格对齐要求。对齐 Pass 确保缓冲区分配与 stride 满足硬件约束，避免未对齐访问的性能下降或执行错误。

## 对齐 Pass 链（mark → enable 两段式）

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hivm-align-alloc-size` | ModuleOp | 自动对齐 `memref.alloc` 的大小：某些操作访问粒度只能对齐到硬件单元大小，调整 alloc 尺寸避免越界 |
| `hivm-mark-stride-align` | FuncOp | 分析所有 HIVM 操作，为其 memref 操作数自动加 `stride_align_dims` + `stride_align_value_in_byte` 标注 |
| `hivm-enable-stride-align` | ModuleOp | 读取上述标注，重新分配 memref 以满足 stride 对齐要求 |
| `hivm-lift-lowest-stride` | FuncOp | 提升操作数最低 stride：末维不连续时将其拆出使最内层连续；例外：MacroOp 和 VArangeOp |
| `hivm-reduce-rank-subview` | FuncOp | 用 subview 降秩，辅助满足对齐约束 |

LiftLowestStride 示例：
```mlir
// 前：末维不连续（stride=8，元素 2 字节）
%0 : memref<16xf16, strided<[8]>>
// 后：拆出单位维，最内层连续
%0 : memref<16x1xf32, strided<[8, 1]>>
```

## 对齐标注属性

| 属性 | 功能 |
|------|------|
| `hivm.stride_align_dims` | 标记需 stride 对齐的维度 |
| `hivm.stride_align_value_in_byte` | stride 对齐的字节值 |
| `hivm.alloc_align_dims` | 分配时需对齐的维度 |
| `hivm.alloc_align_value_in_byte` | 分配对齐的字节值 |

这些标注经 `annotation.mark` 挂在 memref 上，由 mark/enable 两个 Pass 接力消费。

## 硬件对齐约束（NPUTargetSpec.td，全架构一致）

| 地址空间 | 对齐大小 | 源码字段 |
|----------|----------|----------|
| UB | 256 bits（32B） | `UbAlignSize = 256` |
| L1 | 256 bits（32B） | `L1AlignSize = 256` |
| L0C | 4096 bits（512B） | `L0cAlignSize = 4096` |

L0A/L0B 未在 TableGen 中显式定义对齐字段。注意 Reg-based 架构（310B/950）对非单位末维 stride 有更严格的对齐要求，架构差异见 [npu_hardware_overview](../architecture/npu_hardware_overview.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L437-L449](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L437-L449)
- [Passes.td#L451-L461](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L451-L461)
- [Passes.td#L463-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L463-L473)
- [Passes.td#L475-L490](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L475-L490)
- [HIVMAttrs.td#L968-L972](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L968-L972)
- [HIVMAttrs.td#L974-L979](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L974-L979)
- [HIVMAttrs.td#L981-L985](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L981-L985)
- [HIVMAttrs.td#L987-L992](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L987-L992)
- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- [Passes.td#L492-L497](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L492-L497)
- [HIVMAttrs.td#L968-L992](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L968-L992)
