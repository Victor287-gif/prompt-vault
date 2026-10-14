---
type: API
title: Triton 方言体系总览（tt / ttg / gluon 三层与转换链）
description: Triton 编译器三层核心方言：Triton（tt，设备无关张量语义，即 TTIR）、TritonGPU（ttg，并行布局编码与共享内存管理）、Gluon（实验性布局推断），以及
  tt→ttg→LLVM 的转换 Pass 链、操作/类型/属性/编码分类速查。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Triton 方言体系总览

AscendNPU-IR 中的 Triton 编译器定义三层核心方言，对应编译的不同抽象层级；另有实验性的 Gluon 方言承担布局推断。提取到的 `<kernel>_ttir.mlir` 即 Triton（tt）层 IR，`<kernel>_ttadapter.mlir` 是其向 BiShengIR 适配的 IR，提取流程与文件含义见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。

```
Frontend (Python)
    → Triton Dialect (tt)      设备无关 IR，表达张量计算语义
    → TritonGPU Dialect (ttg)  并行布局编码、数据分布与共享内存管理   [ConvertTritonToTritonGPU]
    → LLVM Dialect             底层代码生成                          [TritonGPUToLLVM]
```

| 方言 | 命名空间 | 操作前缀 | 类型前缀 | 属性前缀 |
|------|----------|----------|----------|----------|
| Triton | `::mlir::triton` | `tt.` | `!tt.` | `#tt.` |
| TritonGPU | `::mlir::triton::gpu` | `ttg.` | `!ttg.` | `#ttg.` |
| Gluon | `::mlir::triton::gluon` | `gluon.` | — | — |

## tt → ttg 转换（ConvertTritonToTritonGPU）

核心语义：

- **类型增强**：`tensor<...>` → `tensor<..., #encoding>`，为每个张量附加分布式布局编码。
- **指针类型转换**：`tt.ptr<tensor<...>>` → `tt.ptr<tensor<..., #encoding>>`。
- **操作映射**：大部分 `tt.*` 操作映射到带布局约束的 TritonGPU 版本。
- **新增操作**：`ttg.convert_layout`（布局转换）、`ttg.local_alloc/dealloc/load/store`（共享内存管理）。

转换选项：`target`（如 `cuda:80`）、`num-warps`（默认 4）、`threads-per-warp`（默认 32）、`num-ctas`（默认 1）、`enable-source-remat`（默认 false）、`shared-memory-size`（默认 122880，昇腾条件编译）。

## ttg → LLVM 转换

子 Pass：`allocate-shared-memory`（共享内存分配与偏移标注）、`tritongpu-global-scratch-memory-allocation`、`tritongpu-allocate-warp-groups`。昇腾适配额外引入 `ConvertTritonAscendGPUToLLVM`，经 AscendDPX 方言（[ascend_dpx_dialect](../other_dialects/ascend_dpx_dialect.md)）把 TritonGPU 操作降到 LLVM。

## Gluon 的角色

`gluon.set_auto_layout` 把张量的 `auto` 布局编码替换为具体编码，是布局推断的基础设施（详见 [gluon_dialect.md](gluon_dialect.md)）。

## Triton（tt）操作分类速查

| 类别 | 代表操作 |
|------|---------|
| 类型转换 | `tt.int_to_ptr` / `tt.ptr_to_int` / `tt.bitcast` / `tt.fp_to_fp` |
| 特殊算术 | `tt.clampf` / `tt.precise_sqrt` / `tt.precise_divf` / `tt.mulhiui` |
| 指针算术 | `tt.addptr` / `tt.advance` |
| 内存访问 | `tt.load` / `tt.store` |
| 原子 | `tt.atomic_rmw` / `tt.atomic_cas` |
| 形状 | `tt.splat` / `tt.expand_dims` / `tt.reshape` / `tt.broadcast` / `tt.cat` / `tt.join` / `tt.split` / `tt.trans` |
| SPMD | `tt.get_program_id` / `tt.get_num_programs` |
| 矩阵乘 | `tt.dot` / `tt.dot_scaled` |
| 归约/扫描/映射 | `tt.reduce` / `tt.scan` / `tt.map_elementwise` |
| 其他 | `tt.make_range` / `tt.histogram` / `tt.gather` / `tt.print` / `tt.assert` |
| 张量指针/描述符 | `tt.make_tensor_ptr` / `tt.make_tensor_descriptor` / `tt.descriptor_load/store/reduce/gather/scatter` |
| 函数/外部调用 | `tt.func` / `tt.call` / `tt.return` / `tt.extern_elementwise` / `tt.elementwise_inline_asm` |

操作详解见 [triton_ops.md](triton_ops.md)，类型与枚举属性见 [triton_types_attrs.md](triton_types_attrs.md)。

## TritonGPU（ttg）操作分类速查

| 类别 | 代表操作 |
|------|---------|
| 布局转换 | `ttg.convert_layout` |
| 异步拷贝 | `ttg.async_copy_global_to_local` / `ttg.async_wait` / `ttg.async_commit_group` |
| 共享内存 | `ttg.local_alloc/dealloc/load/store` |
| 内存描述符视图 | `ttg.memdesc_index/subslice/trans/reshape/reinterpret` |
| 流水线 | `ttg.predicate_stage` / `ttg.mask` |
| 类型转换 | `ttg.fp4_to_fp` |
| 全局内存 | `ttg.global_scratch_alloc` |
| Warp 特化 | `ttg.warp_specialize` / `ttg.warp_yield` / `ttg.warp_return` |

操作详解见 [tritongpu_ops.md](tritongpu_ops.md)，布局编码（blocked/linear/mma/dot_op/shared/fractal 等）详解见 [tritongpu_encodings.md](tritongpu_encodings.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonOps.td)
- [TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)
- [GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Gluon/IR/GluonOps.td)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td)
- [TritonTypes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonTypes.td)
- [TritonAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
- [TritonGPUAttrDefs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)
