---
type: API
title: HIVM 二元向量运算（hir.vadd/vsub/vmul/vdiv/vmax/vmin/vor/vand/vxor/vmod/vmodui/vpow）
description: HIVM 二元逐元向量操作族语义与约束：12 个 op 的元素类型白名单、交换律、Vector-Scalar 支持、vdiv 的 isSigned
  属性、各 op 最大 rank 差异与 linalg/arith 降级映射。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 二元向量运算

二元操作继承 `HIVM_ElementwiseBinaryOp`（`ElementwiseNaryOpTrait<2>`：2 输入 1 输出、同 rank），对应 Triton 的 `+ - * / %`、`tl.minimum/maximum` 等。分类：算术（vadd/vsub/vmul/vdiv/vmod/vmodui/vpow）、极值（vmax/vmin）、位运算（vor/vand/vxor）。通用操作数结构见 [vector_overview.md](vector_overview.md)。

## 语法模式

```mlir
%r = hivm.hir.vadd ins(%a, %b : tensor<?x5x10xf32>, tensor<?x5x10xf32>)
                   outs(%dst : tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

// Vector-Scalar（src 为 AnyType，允许标量操作数）
hivm.hir.vadd ins(%vec, %scalar : tensor<23x77xi32>, i32) outs(%dst : tensor<23x77xi32>)

// OTF 广播：64x1 + 1x64 → 64x64
%r = hivm.hir.vsub ins(%a, %b : tensor<64x1xf32>, tensor<1x64xf32>)
                   outs(%dst : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
```

## 操作约束汇总

| 操作 | 元素类型 | 最大 rank | 交换律 | Vec-Scalar | OTF 广播 | Extra Buffer |
|------|---------|-----------|--------|-----------|---------|--------------|
| `hir.vadd` | I8, I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| `hir.vsub` | I8, I16, I32, F16, F32, I64 | 3 | 否 | 是 | 是 | 是 |
| `hir.vmul` | I16, I32, F16, F32, I64（**无 I8**） | 3 | 是 | 是 | 是 | 是 |
| `hir.vdiv` | F16, F32, I16, I32, I64（无 I8） | 3 | 否 | **否**（仅 Vec-Vec） | 是 | 是 |
| `hir.vmax` | I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| `hir.vmin` | I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| `hir.vor` | I1~UI64, F16, BF16, F32 | 3 | 是 | 否（双 VectorOnly） | 是 | 是 |
| `hir.vand` | I1~UI64, F16, BF16, F32 | 3 | 是 | 否 | 是 | 是 |
| `hir.vxor` | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64（**无浮点**） | **2** | 否 | 否 | **否** | 是 |
| `hir.vmod` | I16, I32, I64 | **1** | 否 | 否 | 否 | 否 |
| `hir.vmodui` | I16, I32, I64 | 1 | 否 | 否 | 否 | 否 |
| `hir.vpow` | **仅 I32** | 1 | 否 | 否 | 否 | 是 |

vor/vand 的浮点支持是位级操作（先 bitcast 到整数、位运算、bitcast 回），用于掩码类算法；vxor 硬件不支持浮点位操作。

## 特殊属性

- `vdiv.isSigned`（BoolAttr，默认 true）：整数除法的有/无符号选择，无符号需显式 `isSigned = false`。
- `CommutativeOpTrait`：带交换律的 op 允许编译器交换输入操作数做优化。

## 通用验证规则

1. 输入输出元素类型一致（`SameOperandsElementType`，vcmp 除外——见 [vector_compare_ops.md](vector_compare_ops.md)）、rank 一致。
2. `VectorOnlyTrait`/`ScalarOnlyTrait` 标记的操作数必须为向量/标量。
3. 移位操作（vshl/vshr）虽属二元族但有专属约束，见 [vector_shift_ops.md](vector_shift_ops.md)。

## 降级映射

| HIVM | 目标 | 备注 |
|------|------|------|
| vadd/vsub/vmul/vdiv | linalg.add/sub/mul/div | — |
| vmax/vmin | linalg.max/min | — |
| vor/vand | linalg.map { arith.ori/andi } | 浮点需 bitcast |
| vxor | linalg.map { arith.xori } | — |
| vmod / vmodui | arith.remsi / arith.remui | 有符号/无符号取模 |

## 性能提示

- vadd/vsub/vmul/vmax/vmin 在 i64 或 SIMT VF 模式下标量降级；vmul 不支持 I8 但 vadd/vsub 支持。降级条件详见 [vector_scalar_lowering.md](vector_scalar_lowering.md)。
- `hir.vmulext`（I32 乘法取高 32 位）虽继承二元基类但**始终标量降级**，归入 [vector_special_ops.md](vector_special_ops.md)。

相关：[vector_unary_ops.md](vector_unary_ops.md)、[vector_shift_ops.md](vector_shift_ops.md)、[vector_compare_ops.md](vector_compare_ops.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMVectorOps.td#L506-L508](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L506-L508)
- [HIVMVectorOps.td#L510-L542](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L510-L542)
- [HIVMVectorOps.td#L596-L633](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L596-L633)
- [HIVMVectorOps.td#L544-L576](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L544-L576)
- [HIVMVectorOps.td#L635-L674](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L635-L674)
- [HIVMVectorOps.td#L676-L708](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L676-L708)
- [HIVMVectorOps.td#L710-L742](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L710-L742)
- [HIVMVectorOps.td#L744-L775](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L744-L775)
- [HIVMVectorOps.td#L777-L808](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L777-L808)
- [HIVMVectorOps.td#L810-L843](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L810-L843)
- [HIVMVectorOps.td#L845-L857](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L845-L857)
- [HIVMVectorOps.td#L859-L871](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L859-L871)
- [HIVMVectorOps.td#L981-L1010](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L981-L1010)
- [HIVMVectorOps.td#L506-L1010](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L506-L1010)
- [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
