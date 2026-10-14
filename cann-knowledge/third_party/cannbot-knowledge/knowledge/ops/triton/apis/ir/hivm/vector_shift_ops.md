---
type: API
title: HIVM 移位运算（hivm.hir.vshl / hivm.hir.vshr）
description: hivm.hir.vshl 与 hivm.hir.vshr 移位操作语义：仅整数 I16/I32/I64、移位量为统一标量、vshr 的
  round 属性（四舍五入算术右移）、arith.shli/shrsi 降级。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 移位运算（hivm.hir.vshl / hivm.hir.vshr）

移位操作属二元族变体：被移位的是向量，**移位量是统一标量**（全张量同一位数），仅支持整数 I16 / I32 / I64，最大 rank 3。对应 Triton 的 `<<` / `>>`。

## 语法

```mlir
%r = hivm.hir.vshl ins(%x, %n : tensor<16xi32>, i32) outs(%dst : tensor<16xi32>) -> tensor<16xi32>
%r = hivm.hir.vshr ins(%x, %n : tensor<16xi32>, i32) outs(%dst : tensor<16xi32>)
                   round = true -> tensor<16xi32>
```

## 约束

| 操作 | Traits | Extra Buffer | 降级 |
|------|--------|--------------|------|
| `hir.vshl` | VectorOnlyTrait<0> + ScalarOnlyTrait<1>（严格 Vector-Scalar） | 是 | arith.shli |
| `hir.vshr` | VectorOnlyTrait<0>；带 ImplByScalarOpInterface（Vec-Scalar） | 是 | round=false → arith.shrsi |

两者都支持 OTF 广播。

## vshr.round（BoolAttr，默认 true）

- `round = true`：四舍五入算术右移，`dst ≈ (src + (1 << (shift-1))) >> shift`，等价于除以 2^shift 后四舍五入，常用于定点量化回缩。
- `round = false`：普通算术右移（符号位扩展）。

## 注意

- 移位量必须标量；若需要逐元素不同移位量，硬件不直接支持，需展开为乘除或多路选择。
- 仅整数类型；浮点缩放应使用 `hir.vmul`（[vector_binary_ops.md](vector_binary_ops.md)）乘 2 的幂。

相关：[vector_binary_ops.md](vector_binary_ops.md)、[vector_scalar_lowering.md](vector_scalar_lowering.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L873-L904](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L873-L904)
- [HIVMVectorOps.td#L906-L942](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L906-L942)
