---
type: API
title: Triton 类型与属性详解（!tt.ptr / !tt.tensordesc 与枚举属性）
description: Triton 方言类型体系：PointerType（!tt.ptr，标量/张量指针双重角色）、TensorDescType（!tt.tensordesc，TMA
  抽象）、TT_Float/TT_Int/TT_PtrLike 等类型约束别名，以及 CacheModifier、EvictionPolicy、RMWOp、MemSemantic、InputPrecision
  等 12 组枚举属性。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.atomic]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Triton 类型与属性详解

Triton 方言（`::mlir::triton`）的类型与枚举属性定义，配合 [triton_ops.md](triton_ops.md) 中的操作签名阅读。

## 核心类型

### PointerType（`!tt.ptr`）

参数 `$pointeeType`（指向类型）与 `$addressSpace`（地址空间编号，int）。两种使用模式：

1. **标量指针** `!tt.ptr<f32>`：指向单个值或一维连续内存。
2. **张量指针** `!tt.ptr<tensor<8x8xf16>>`：指向多维张量，附带形状与步长信息。

### TensorDescType（`!tt.tensordesc`）

张量描述符类型，NVIDIA TMA 描述符的可移植抽象，参数 `$blockType: RankedTensorType`（如 `!tt.tensordesc<tensor<8x8xf16>>`）。构建器支持有符号/无符号语义；辅助方法 `getSignlessBlockType()`、`getTensorShape()`。

### 张量指针 vs 张量描述符

| 特性 | `!tt.ptr<tensor<...>>` | `!tt.tensordesc<tensor<...>>` |
|------|------------------------|-------------------------------|
| 创建 | `tt.make_tensor_ptr` | `tt.make_tensor_descriptor` |
| 偏移推进 | `tt.advance` | 索引参数 |
| 读 / 写 | `tt.load` / `tt.store` | `tt.descriptor_load` / `tt.descriptor_store` |
| 硬件映射 | 通用 | TMA 硬件加速 |
| 边界检查 | `boundaryCheck` 属性 | 描述符内置 |

## 类型约束别名

| 约束 | 定义 | 含义 |
|------|------|------|
| `TT_Float` | F8E4M3FN / F8E4M3FNUZ / F8E5M2 / F8E5M2FNUZ / F16 / BF16 / F32 / F64 | 全部浮点类型 |
| `TT_Int` | I1 / I4 / I8 / I16 / I32 / I64 | 全部整数类型 |
| `TT_Ptr` | `ptr<AnyType>` | 标量指针 |
| `TT_PtrTensor` | `RankedTensorOf<[TT_Ptr]>` | 指针张量 |
| `TT_PtrLike` | `TT_Ptr \| TT_PtrTensor` | 指针或指针张量 |
| `TT_TensorPtr` | `ptr<TT_Tensor>` | 张量指针 |
| `TT_Tensor` | `RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>` | 通用张量（含指针张量） |
| `TT_FloatLike` / `TT_IntLike` | 标量或对应张量 | FloatLike 含 TT_Float + TT_FloatTensor |
| `TT_BoolLike` | I1 或 I1 张量 | 布尔 |
| `TT_FpIntTensor` | `RankedTensorOf<[TT_Float, TT_Int]>` | 浮点/整数张量 |
| `TT_Type` | FloatLike / IntLike / PtrLike / TensorPtr 之并 | 几乎所有数据类型 |
| `TT_I32Like` / `TT_I64Like` | i32/i64 标量或张量 | — |

`TT_Type` 不覆盖 `!tt.tensordesc` 与 TritonGPU 的 `TTG_AsyncToken`。

## 枚举属性

### CacheModifier（`$cache`，默认 NONE）

`none` / `ca`（Cache All）/ `cg`（Cache Global）/ `wb`（写回）/ `cs`（流式）/ `wt`（写穿透）/ `cv`（易失）。使用者：`tt.load`、`tt.store`、`tt.descriptor_load`、`ttg.async_copy_global_to_local`。

### EvictionPolicy（`$evict`，默认 NORMAL）

`evict_normal` / `evict_first` / `evict_last`。使用者同 CacheModifier。

### PaddingOption

`zero`（PAD_ZERO）/ `nan`（PAD_NAN）。用于 `tt.load`（可选）与 `tt.make_tensor_descriptor`（默认 PAD_ZERO）。

### RMWOp（`tt.atomic_rmw` 的 `atomic_rmw_op`）

`and` / `or` / `xor` / `add` / `fadd` / `max` / `min` / `umax` / `umin` / `exch`（XCHG）。

### DescriptorReduceKind（`tt.descriptor_reduce` 的 kind）

`add` / `min` / `max` / `inc` / `dec` / `and` / `or` / `xor`。

### MemSemantic 与 MemSyncScope（原子操作）

- MemSemantic：`relaxed` / `acquire` / `release` / `acq_rel`。
- MemSyncScope：`gpu` / `cta` / `sys`。

### 其他枚举

| 属性 | 枚举值 | 用途 |
|------|--------|------|
| ProgramIDDim | `x` / `y` / `z` | 程序 ID 维度 |
| RoundingMode | `rtz` / `rtne` | 浮点转换舍入 |
| PropagateNan | `none` / `all` | `tt.clampf` 的 NaN 传播 |
| InputPrecision | `tf32` / `tf32x3` / `ieee` | `tt.dot` 输入精度（仅 f32 输入有效） |
| ScaleDotElemType | `e4m3` / `e5m2` / `e2m3` / `e3m2` / `e2m1` / `bf16` / `fp16` | `tt.dot_scaled` 缩放元素类型 |

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [TritonTypes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonTypes.td)
- [TritonAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
