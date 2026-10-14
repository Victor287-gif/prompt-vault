---
type: API
title: HIVM 宏操作 MixGroupMatmul（hivm.hir.mix_group_matmul）
description: hivm.hir.mix_group_matmul 操作语义：MoE 场景分组矩阵乘，3D 专家权重 [num_experts,K,N]
  × 2D token 矩阵 [total_tokens,K]，由 tokens_per_expert 指定每个 expert 的 token 数，支持 Vector
  后处理输入/输出与通信融合。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 宏操作 MixGroupMatmul（hivm.hir.mix_group_matmul）

`hir.mix_group_matmul` 是面向 Mixture-of-Experts（MoE）的分组矩阵乘：不同 token 被路由到不同 expert（各自独立的权重矩阵），单个操作内完成所有 expert 的分组计算，编译器可整体优化数据搬运与同步。计算语义仍为 `C = descale × (A × B + bias)`，Pipeline 为 MTE2 + MTE3（GM→GM）。

## 关键操作数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a` | AnyShaped | 是 | 权重矩阵，**3D** `[num_experts, K, N]`（每个 expert 一个权重） |
| `$b` | AnyShaped | 是 | Token 矩阵，2D `[total_tokens, K]` |
| `$tokens_per_expert` | AnyShaped | 是 | 1D `[num_experts]`，`[i]` = 第 i 个 expert 处理的 token 数，元素之和 = total_tokens |
| `$postVecFuncIns` / `$postVecFuncOuts` | Variadic\<AnyShaped\> | 否 | Vector 后处理函数输入/**输出**（输出为 mix_matmul 所无） |
| `$workspaceIns` | Variadic\<AnyShaped\> | 否 | 工作空间缓冲区 |
| `$tilingParams` / `$commParams` / `$bias` / `$descale` | AnyShaped | 否 | 同 [mix_matmul](macro_mix_matmul.md) |
| `$c` | AnyShaped | 是 | 输出矩阵 C |

属性与 I64 操作数（`aTranspose`/`bTranspose`、`descaleMode`、`blockSizes`、`processSizes`、`swizzleOffset`/`swizzleDirection`、`epiloguePTiles`）同 matmul 系。`getOpName()` 返回 `"mix_group_matmul"`。

## 与 mix_matmul 的差异

| 特性 | mix_matmul | mix_group_matmul |
|------|-----------|------------------|
| 权重矩阵 A | 2D `[M, K]` | 3D `[num_experts, K, N]` |
| Token 矩阵 B | 2D `[K, N]` | 2D `[total_tokens, K]` |
| tokens_per_expert | 不支持 | **必需**，1D `[num_experts]` |
| post_vector_func_outs | 不支持 | 支持（后处理可产额外输出） |

## IR 示例

```mlir
hivm.hir.mix_group_matmul
   ins(%A_gm, %B_gm, %tokens_per_expert_gm:
       memref<16x16x16xf16, #hivm.address_space<gm>>,
       memref<16x16xf16, #hivm.address_space<gm>>,
       memref<16xi64, #hivm.address_space<gm>>)
   post_vector_func_ins(%pv_ins : memref<1024x1024xf16, #hivm.address_space<gm>>)
   post_vector_func_outs(%pv_outs : memref<1024x1024xf16, #hivm.address_space<gm>>)
   outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
   tiling_params = %tiling_gm : memref<16xf16, #hivm.address_space<gm>>
   comm_params = %comm_gm : memref<16xi64, #hivm.address_space<gm>>
```

## IR 层约束

1. A 必须 3D；`tokens_per_expert` 必须 1D 且长度 = num_experts、元素和 = total_tokens。
2. Core Type 由 `HIVMInferCoreTypeInterface` 推断；`NoMaxRankTrait` 不限 rank。
3. 相比"循环多次 mix_matmul"，单操作分组计算避免循环展开带来的额外搬运与同步开销。

相关：[macro_mix_matmul.md](macro_mix_matmul.md)、[macro_matmul.md](macro_matmul.md)、[macro_overview.md](macro_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMMacroOps.td#L424-L535](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L424-L535)
- [ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/ops.mlir)
