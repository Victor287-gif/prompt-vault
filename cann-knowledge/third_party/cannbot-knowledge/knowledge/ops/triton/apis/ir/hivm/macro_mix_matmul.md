---
type: API
title: HIVM 宏操作 MixMatmul（hivm.hir.mix_matmul）
description: hivm.hir.mix_matmul 操作语义：GM→GM 矩阵乘 + tile 级 Vector 后处理融合（post_vector_func），支持
  workspace 缓冲区与通信参数（AllReduce 等），避免 Cube 结果回 GM 再读出的额外开销。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 宏操作 MixMatmul（hivm.hir.mix_matmul）

`hir.mix_matmul` 在 [matmul](macro_matmul.md)（GM→GM，`C = descale × (A × B + bias)`）基础上增加 **Cube+Vector 混合执行**：Cube Core 完成矩阵乘后，在 Vector Core 上对 tile 级结果直接执行后处理函数（激活、类型转换等），省掉"结果写回 GM 再读入向量单元"的往返开销。Triton `tl.dot` + 后续 elementwise 链在编译时可能融合为 mix_matmul。

## 相对 matmul 的新增参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `$postVecFuncIns` | Variadic\<AnyShaped\> | Vector 后处理函数的输入 |
| `$workspaceIns` | Variadic\<AnyShaped\> | 工作空间缓冲区（中间结果存储，如融合 AllReduce 时存部分和） |
| `$commParams` | Optional\<AnyShaped\> | 通信参数（拓扑、通信器、group 等），用于融合通信操作 |
| `post_vector_func`（属性） | StrAttr | 后处理函数名称（如 `"bishengir_gen_vector_epilogue_func"`） |

其余参数（`a`/`b`/`c`、`tilingParams`、`bias`、`descale`、`aTranspose`/`bTranspose`、`descaleMode`、`blockSizes`、`processSizes`、`swizzleOffset`/`swizzleDirection`、`epiloguePTiles`）与 matmul 相同。

## 与 matmul 的差异速查

| 特性 | matmul | mix_matmul |
|------|--------|------------|
| 后处理融合 | 不支持 | `post_vector_func` 属性 + `postVecFuncIns` 输入 |
| 工作空间 | 不支持 | `workspaceIns` |
| 通信融合 | 不支持 | `commParams` |
| `getOpName()` | `"matmul"` | `"mix_matmul"` |

## IR 示例

```mlir
hivm.hir.set_ffts_base_addr %arg0
hivm.hir.mix_matmul {post_vector_func = "bishengir_gen_vector_epilogue_func"}
  ins(%arg1, %arg2 :
      memref<1024x1024xf16, #hivm.address_space<gm>>, memref<1024x1024xf16, #hivm.address_space<gm>>)
  post_vector_func_ins(%arg3 : memref<1024x1024xf16, #hivm.address_space<gm>>)
  workspace_ins(%arg4 : memref<1024x1024xf16, #hivm.address_space<gm>>)
  outs(%arg5 : memref<1024x1024xf16, #hivm.address_space<gm>>)
  block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64)
  process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64)
  swizzle_offset = %c1_i64 : i64
  swizzle_direction = %c0_i64 : i64
  epilogue_p_tiles = %c4_i64 : i64
```

## IR 层约束

1. Core Type 由 `HIVMInferCoreTypeInterface` 推断；Pipeline 为 MTE2 + MTE3。
2. 使用 mix_matmul 的函数通常标记 `hivm.func_core_type = #hivm.func_core_type<AIV>` 或 `MIX`（Mix Kernel，见 [attr_enumerations.md](attr_enumerations.md) 的 TFuncCoreType）。
3. `commParams` 仅在融合通信（如 AllReduce）时出现；`set_ffts_base_addr` 常配套出现（FFTS 跨核同步，见 [sync_block_sync.md](sync_block_sync.md)）。
4. L0C 矩阵需要拆给两个 UB 做 Cube-Vector 协同时，可配合 fixpipe 双目标模式（见 [dma_fixpipe.md](dma_fixpipe.md)）。

相关：[macro_matmul.md](macro_matmul.md)、[macro_mix_group_matmul.md](macro_mix_group_matmul.md)、[macro_overview.md](macro_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMMacroOps.td#L320-L422](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L320-L422)
- [ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/ops.mlir)
