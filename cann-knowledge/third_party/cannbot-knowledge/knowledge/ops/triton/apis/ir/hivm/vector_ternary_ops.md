---
type: API
title: HIVM 三元向量运算（hivm.hir.vsel）
description: 'hivm.hir.vsel 条件选择操作语义：cond ? src0 : src1 的逐元选择、cond 可为标量 i1、rank≤1、Extra
  Buffer 需求与 linalg.select 降级，以及 vcmp→vsel 典型链。'
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 三元向量运算（hivm.hir.vsel）

三元族只有一个操作 `hir.vsel`（`HIVM_ElementwiseTernaryOp`），按条件张量逐元素二选一，对应 Triton 的 `tl.where(cond, x, y)`。

## 语义与语法

```
dst[i] = condition[i] ? src0[i] : src1[i]
```

```mlir
%r = hivm.hir.vsel ins(%cond, %x, %y : tensor<16xi8>, tensor<16xf32>, tensor<16xf32>)
                   outs(%dst : tensor<16xf32>) -> tensor<16xf32>
// cond 可为标量 i1
%r = hivm.hir.vsel ins(%c, %x, %y : i1, tensor<16xf32>, tensor<16xf32>)
                   outs(%dst : tensor<16xf32>) -> tensor<16xf32>
```

## 操作数与约束

| 操作数 | 元素类型 | 说明 |
|--------|---------|------|
| `src[0]`（condition） | I1, I8 | AnyType，允许标量 i1 |
| `src[1]` / `src[2]` | I1, AnyI8, AnyI16, F16, BF16, AnyI32, F32, I64 | 两者类型必须一致 |
| `dst` | 同 src[1] | — |

- 最大 rank **1**（`StaticMaxRankTrait<1>`）——高维需先行展平或经 OTF 支持。
- 支持 OTF 广播（BroadcastableOTF）。
- 需要 Extra Buffer（ExtraBufferOpInterface，见 [extra_buffer](../memory_management/extra_buffer.md)）。

## 降级

vsel → `linalg.select`。典型链：`hir.vcmp` 生成条件 → `hir.vsel` 选择（对应 `tl.where(tl.x < 0, 0, tl.x)` 类模式）。vcmp 的整数比较标量降级会连累整条 vcmp→vsel 链，见 [vector_compare_ops.md](vector_compare_ops.md)。

相关：[vector_compare_ops.md](vector_compare_ops.md)、[vector_binary_ops.md](vector_binary_ops.md)、[vector_overview.md](vector_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L1016-L1018](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1016-L1018)
- [HIVMVectorOps.td#L1020-L1050](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1020-L1050)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
