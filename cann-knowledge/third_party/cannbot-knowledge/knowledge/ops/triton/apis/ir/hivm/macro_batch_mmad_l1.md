---
type: API
title: HIVM 宏操作 BatchMMAD-L1（hivm.hir.batchMmadL1）
description: hivm.hir.batchMmadL1 操作语义：Cube Core 上带批量维的本地矩阵乘加 C[i] = C[i] + A[i]×B[i]，rank=3（第
  0 维为 batch），带 NoLibraryFunctionTrait 无库函数 lowering。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 宏操作 BatchMMAD-L1（hivm.hir.batchMmadL1）

`hir.batchMmadL1` 是 [mmadL1](macro_mmad_l1.md) 的批量版本：Cube Core 上执行 `C[i] = C[i] + A[i] × B[i] + (可选) channel_bias`，数据通路同样是 L1→L0C（MTE1→M）。矩阵 A/B/C 的 rank 必须为 3，第 0 维是批量维。Triton `tl.dot` 带 batch 维时可能映射为此操作。

## 与 mmadL1 的差异

| 特性 | mmadL1 | batchMmadL1 |
|------|--------|-------------|
| 矩阵 rank | 2 | 3（第 0 维为 batch） |
| A / B / C 形状 | `[M,K]` / `[K,N]` / `[M,N]` | `[Batch,M,K]` / `[Batch,K,N]` / `[Batch,M,N]` |
| OpLayoutInterface | 实现 `getOperandsTargetFractalLayout` | 未实现（Fractal 布局走默认推导） |
| LibraryFunctionTrait | 默认支持 | `NoLibraryFunctionTrait`（无预定义库函数，不能直接 lowering 到标准库调用） |
| `getOpName()` | `"mma_tile"` | 无（默认） |

操作数与属性（`init_condition`、`real_m/k/n`、`per_channel_bias`、`a_transpose`、`b_transpose`、`enable_HF32`、`unit_flag_*`）语义与 mmadL1 完全一致，此处不再重复。

## IR 示例

```mlir
%init = arith.constant 1 : i1
hivm.hir.batchMmadL1 ins(%ma, %mb, %init, %c256, %c128, %c256 :
                        memref<2x256x128xf16>, memref<2x128x256xf16>, i1, index, index, index)
                  outs(%mc : memref<2x256x256xf32>)
```

## IR 层约束

1. rank=3 且 A/B/C 批量维大小一致。
2. `CubeCoreTypeTrait`：仅 Cube Core；Pipeline 为 MTE1 + M。
3. 单操作处理整个批量可利用批量间数据局部性；等价循环多次 mmadL1 会引入更多同步开销（同步开销瓶颈见 [../ir_analysis_workflow.md](../../../guides/ir/ir_analysis_workflow.md)）。
4. `NoLibraryFunctionTrait` 意味着分析其 lowering 时找不到库函数命名，需走编译器内建展开路径。

相关：[macro_mmad_l1.md](macro_mmad_l1.md)、[macro_overview.md](macro_overview.md)、[attr_traits.md](attr_traits.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMMacroOps.td#L174-L184](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L174-L184)
- [ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/ops.mlir)
