---
type: API
title: HFusion 变换 Pass（融合、调度、向量化与规范化）
description: HFusion 方言 40+ 变换 Pass 分类清单：HFusionOpFusion / AutoSchedule / AutoVectorize(V2)
  核心链路与关键选项，外联、规范化、归约优化、缓冲管理、Tiling、符号函数管理与 Triton 适配 Pass。
tags: [task.lookup, topic.compilation, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 变换 Pass

HFusion 阶段的变换完成算子融合、自动调度、向量化与各类规范化，是从前端图到硬件映射前的全部高层优化。HFusion 方言操作见 [overview](../hfusion/overview.md)，方言层变换视角见 [transforms](../hfusion/transforms.md)；在完整 pipeline 中的执行顺序见 [pass_dependencies.md](pass_dependencies.md)。

## 融合与调度

### HFusionOpFusion（`hfusion-fuse-ops`，ModuleOp）

将多个算子融合为一个 kernel。关键选项：

| 选项 | 默认 | 说明 |
|------|------|------|
| `output-mode` | `Multiple` | 外联函数输出模式：`multi`/`single`/`single-aggr` |
| `fusion-mode` | `Unknown` | 融合类型，由标签决定 |
| `always-inline` | false | 外联函数始终内联 |
| `move-out-to-param` | true | tensor 输出移到参数 |
| `max-horizontal-fusion-size` | -1 | 最大水平融合大小，-1 无限制 |
| `multi-kernel` | false | 启用多 kernel 融合 |
| `enable-symbol-analysis` | false | 启用符号分析 |

### AutoSchedule（`hfusion-auto-schedule`，ModuleOp）

自动调度融合后的 kernel。关键选项：`block-dim`（块数，默认 1）、`enable-auto-multi-buffer`、`enable-deterministic-computing`（默认 true）、`max-buffer-count-tuning`、`enable-count-buffer-dma-opt`、`cube-tiling-tuning`、`external-tiling-func-path`（外部 tiling 函数路径）。

## 向量化

| Pass | CLI 参数 | 说明与关键选项 |
|------|----------|----------------|
| AutoVectorize | `hfusion-auto-vectorize` | linalg 命名算子 → 向量操作；`vector-length`（默认 256 字节）、`peel-loops`、`max-vectorize-axes`、`tree-reduce` |
| AutoVectorizeV2 | `hfusion-auto-vectorize-v2` | 第二代，支持 tile/fuse/vectorize；`max-fused-ops`（默认 15）、`enable-multiple-consumer-fusion` |
| VectorizeOps | `hfusion-vectorize-ops` | HFusion 算子向量化；`for-manual-scope` 仅手动 scope 函数 |
| GenericUnroller | `hfusion-generic-unroller` | 不可向量化 linalg 算子展开（目前仅 linalg.reduce） |

## 函数外联

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| OutlineVectorFunction | `outline-vector-function` | 外联向量函数 |
| OutlineSingleOp | `hfusion-outline-single-op` | 单 linalg 算子外联为 kernel；`move-out-to-param` |
| PullSliceIntoVectorFunction | `hfusion-pull-slice-into-vector-function` | extract_slice/insert_slice 拉入 VF 调用者，减少缓冲拷贝 |

## 形状与类型规范化

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| FlattenOps | `hfusion-flatten-ops` | 展平 linalg/hfusion 算子；`flatten-mode`（greedy/tidy）、`multi-dynamic-shape`、`register-based`、`skip-scope` |
| Normalize | `hfusion-normalize-ops` | 规范化 HFusion 操作；`enable-high-precision`（sin/cos 高精度，默认 true） |
| NormalizeSliceOps | `hfusion-normalize-slice-ops` | 切片操作规范化 |
| LegalizeScalarPass | `hfusion-legalize-scalar` | 标量操作 → 张量操作 |
| LegalizeBF16Pass / LegalizeFP8Pass | `hfusion-legalize-bf16` / `hfusion-legalize-fp8` | BF16/FP8 → FP32 |
| LegalizeBoolPass | `hfusion-legalize-bool` | int8 ↔ int1；`enable-clamp` 伪布尔算术 |
| DowngradeFP64CstOpPass | `hfusion-downgrade-fp64` | FP64 常量 → FP32 |

## 算子变换

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| ConvertGenericToNamedOp | `hfusion-convert-generic-to-named` | linalg.generic → 命名算子 |
| HFusionInlineBrc | `hfusion-inline-brc` | 内联广播类操作 |
| SimplifyOps | `hfusion-simplify-ops` | 简化操作 |
| ReorderOpsByBFS | `hfusion-reorder-ops` | 按 BFS 顺序重排 |
| Decompose | `hfusion-decompose` | 分解 AggregatedOpInterface 算子；`hfusion-decompose-phase`（no-constraint/after-hfusion-flatten） |
| HFusionFoldUnitDims | `hfusion-fold-unit-dims` | 移除单位维度 |
| HFusionGeneralizePass | `hfusion-generalize` | hfusion 算子 → linalg.generic |
| RemoveMaskFromUnalignedReductionLoop | `remove-mask-from-unaligned-reduction-loop` | 非对齐归约循环移除掩码 |

## 归约优化

- **ComposeMultiReduce**（`hfusion-compose-multi-reduce`）：多归约组合；`max-compose`、`max-dist-diff`、`aggressive`（激进 reshape 匹配）。
- **DecomposeMulti**（`hfusion-decompose-multi`）：多输出算子分解为单输出算子。

## 缓冲与内存管理

CacheIO（`hfusion-cache-io`）、CacheIOForReturnArg、ReCacheIO、RemoveCacheIO 为一组 IO 缓存 Pass；HoistTensorEmpty（`hfusion-hoist-tensor-empty`）把 tensor.empty 提升并合并为一个函数参数；TensorResToOutParams（`hfusion-tensor-results-to-out-params`）把张量结果移到输出参数，支持 `include-symbols` 过滤与 `enable-manage-host-resources`。

## Tiling 与形状推断

- **PackTilingData**（`hfusion-pack-tiling-data`）：动态 tiling 信息打包为结构体；`pack-tiling-key`（默认 true）、`emit-get-tiling-struct-size-function`。
- **ConstantizeTilingData**（`hfusion-constantize-tiling-data`）：tiling 函数与设备函数间传播常量。
- **InferOutShapesPass**（`hfusion-infer-out-shapes`）：为 kernel 生成输出张量形状函数。

## 符号与函数管理

FoldSymbolicDim / UnfoldSymbolicDim（`hfusion-fold-symbolic-dim` / `hfusion-unfold-symbolic-dim`，tensor.dim ↔ hfusion::SymbolicDimOp）、DropSymbols、EliminateDuplicateFuncs、InferFuncFusionKind、SimplifyVFArgs、MergeVecScope（`hfusion-merge-vf`，`merge-level` 0/1/2、`merge-vf-num-limit` 默认 4）、WrapHostFunc（`hfusion-wrap-host-func`，`remove-unused-arguments`）。

## Triton 适配

- **AdaptTritonKernel**（`adapt-triton-kernel`）：适配 Triton kernel。
- **PreVectorizationFusion**（`hfusion-pre-vectorization-fusion`）：向量化前逐元素融合与泛化；`enable-triton-compile`、`max-fused-elementwise-ops`。
- **AddFFTSAddr**（`hfusion-add-ffts-addr`）：为函数参数和注解添加 FFTS 基地址；`force-add-ffts-addr`。

变换完成后由 `ConvertHFusionToHIVM` 进入硬件映射层，见 [hfusion_to_hivm.md](hfusion_to_hivm.md)。各 Pass 的开关由 HFusionPipelineOptions 控制，见 [pipeline_options.md](pipeline_options.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td)
