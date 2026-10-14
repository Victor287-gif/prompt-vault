---
type: API
title: HIVM 类型系统约束（MemRef / Tensor / Vector）
description: HIVM 在标准 MLIR 类型上的约束：MemRef 的 AddressSpace 标注与各存储层次可用操作、strided memref
  子视图、Tensor/MemRef 双语义选择、元素类型清单与矩阵乘类型组合、类型转换路径与 rank Trait。
tags: [task.lookup, topic.compilation, topic.implementation.memory_management, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 类型系统约束

HIVM 在标准 MLIR 类型（MemRef/Tensor/Vector）上附加 NPU 存储层次、布局与元素类型约束。

## MemRef 约束

### AddressSpace 标注

```mlir
memref<16x16xf16, #hivm.address_space<gm>>     // 全局内存
memref<16x16xf16, #hivm.address_space<cbuf>>   // L1
memref<16x16xf16, #hivm.address_space<ca>>     // L0A
memref<16x16xf16, #hivm.address_space<cb>>     // L0B
memref<16x16xf16, #hivm.address_space<cc>>     // L0C
memref<16x16xf16, #hivm.address_space<ub>>     // UB
```

### 存储层次 × 可用操作

| 层次 | AddressSpace | 可用操作 | 说明 |
|------|-------------|---------|------|
| GM | `gm` | load, store, matmul, mix_matmul | 全局内存，所有 Block 共享 |
| L1 | `cbuf` | nd2nz, copy, mmadL1 输入 | CBuffer，Block 内共享 |
| L0A/L0B | `ca` / `cb` | mmadL1 内部使用 | Cube A/B 矩阵缓冲 |
| L0C | `cc` | fixpipe 输入 | Cube 累加器 |
| UB | `ub` | vector ops, fixpipe 输出 | Vector Core 使用 |

层次容量与对齐数值见 [memory_hierarchy](../architecture/memory_hierarchy.md)。

### Strided MemRef（子视图）

```mlir
memref<256x128xf16, strided<[2048, 1], offset: 0>>
memref<256x128xf16, strided<[2048, 1]>, #hivm.address_space<ub>>
```

## Tensor 约束

1. 遵循 MLIR 标准 Tensor 语义；HIVM 操作多实现 **DestinationStyleOpInterface**，Tensor 经 `outs` 操作数传入。
2. 要求 **RankedTensor**（形状确定）。
3. 大多数操作同时支持 Tensor 与 MemRef 双语义，但同一操作内通常全 Tensor 或全 MemRef。

| 场景 | 推荐类型 |
|------|---------|
| 函数间传递数据 | MemRef + AddressSpace（层次明确） |
| 操作间中间结果 | Tensor（类型更安全） |
| 子视图 | MemRef + subview（Tensor 用 extract_slice） |
| Bufferization 后 | MemRef（Tensor 被转换，见 [bufferization](../memory_management/bufferization.md)） |

## Vector 约束

元素类型须为硬件支持类型；形状须匹配 Vector Core 计算单元。不支持组合会触发标量降级（见 [vector_scalar_lowering.md](vector_scalar_lowering.md)）。

## 元素类型

支持：f16 / bf16 / f32 / f64（有限）/ i8 / i16 / i32 / i64 / i1。

### 矩阵乘类型组合

| A | B | C | 说明 |
|---|---|---|------|
| f16 | f16 | f32 | 标准 FP16 乘，FP32 累加 |
| f16 | f16 | f16 | FP16 累加 |
| bf16 | bf16 | f32 | BF16 乘 |
| i8 | i8 | i32 | INT8 量化乘 |

### 类型转换路径

- **bitcast**：位模式不变（总位宽相同）。
- **cast_signed / cast_unsigned**：有/无符号转换（vcast，见 [vector_cast_ops.md](vector_cast_ops.md)）。
- **FixpipePreQuantMode**：fixpipe 随路在线转换（F322F16/S322I8 等，见 [dma_fixpipe.md](dma_fixpipe.md)）。

## 类型相关 Trait

| Trait | 说明 |
|-------|------|
| `HIVMOpSameOperandsAndResultRank` | 操作数与结果 rank 相同（临时缓冲除外） |
| `ElementwiseNaryOpTrait<N>` | N 元逐元操作，rank 相同 |
| `StaticMaxRankTrait<MaxRank>` / `NoMaxRankTrait` | 静态最大 rank / 无限制 |
| `VectorOnlyTrait<idx>` / `ScalarOnlyTrait<idx>` | 指定操作数仅 Vector / 标量 |
| `OperElemTypeConstraints<indices, types>` | 操作数元素类型约束 |

Trait 完整体系见 [attr_traits.md](attr_traits.md)。

相关：[attr_overview.md](attr_overview.md)、[attr_parameterized_attrs.md](attr_parameterized_attrs.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
- [ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/ops.mlir)
