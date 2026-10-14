---
type: API
title: Pass 依赖关系与执行顺序（HFusion / HIVM / TritonGPU 三条链）
description: buildHFusionPipelines、buildConvertToHIVMPipeline、buildHIVMTensorOptimizations、buildLowerHIVMPipelines
  与 Triton 编译链的分阶段执行顺序，必须顺序执行的 Pass 对、可选 Pass 条件启用表，及 last_pass.mlir 的阶段定位。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Pass 依赖关系与执行顺序

本卡给出三条编译链的分阶段 Pass 顺序与硬依赖约束。各 Pass 的语义与选项见 [hfusion_transforms.md](hfusion_transforms.md)、[hivm_transforms.md](hivm_transforms.md)、[triton_ir_compilation.md](triton_ir_compilation.md)；选项字段见 [pipeline_options.md](pipeline_options.md)。

## HFusion Pipeline（buildHFusionPipelines）

```
阶段 1 前端转换    ConvertLinalgToHFusion → ConvertArithToHFusion → ConvertMathToHFusion
                  → ConvertTensorToHFusion → ConvertGPUToHFusion
阶段 2 规范化      HFusionNormalizeOps → LegalizeBF16 → LegalizeFP8 → LegalizeBool
                  → LegalizeScalar → ConvertGenericToNamedOp → HFusionFoldUnitDims
阶段 3 融合调度    HFusionOpFusion → ComposeMultiReduce → DecomposeMulti → HFusionInlineBrc
                  → InferFuncFusionKind → AutoSchedule
阶段 4 向量化      AutoVectorizeV2（或 AutoVectorize）→ VectorizeOps
                  → PreVectorizationFusion → GenericUnroller
阶段 5 展平优化    FlattenOps → HFusionDecompose → ReorderOpsByBFS → SimplifyOps
阶段 6 函数管理    OutlineVectorFunction → OutlineSingleOp → PullSliceIntoVectorFunction
                  → MergeVecScope → SimplifyVFArgs → EliminateDuplicateFuncs → WrapHostFunc
阶段 7 内存与形状  CacheIO/ReCacheIO → HoistTensorEmpty → TensorResToOutParams
                  → PackTilingData → ConstantizeTilingData → InferOutShapesPass
                  → DropSymbols/FoldSymbolicDim/UnfoldSymbolicDim → AddFFTSAddr
阶段 8 Triton 适配 AdaptTritonKernel
```

寄存器模式 `buildHFusionRegBasePipeline` 顺序类似，差异：FlattenOps 用 `register-based` 模式、跳过 AutoSchedule 与 AutoVectorize、直接走 ConvertHFusionToVector。

## HIVM Pipeline

### ConvertToHIVM（buildConvertToHIVMPipeline）

前端→HFusion（同上）→ `ConvertHFusionToHIVM`（`mm-map-mode`）→ `ConvertToHIVMOp`。

### Tensor 优化（buildHIVMTensorOptimizations）

```
阶段 1 类型推断    InferFuncCoreType → InferHIVMMemScope → InferVFMode → InferHIVMDataLayout
阶段 2 规范化      NormalizeMatmul → Normalize → HIVMDecomposeOp → HIVMAggregatedDecomposeOp
                  → HIVMRecognizeDeinterleaveOp → InlineOTFBroadcast → InlineFixpipe(V2)
阶段 3 布局转换    InsertConvertLayout → PropagateConvertLayout → ConvertLayoutToTranspose
                  → CombineOptimizedConvertLayout
阶段 4 Stride 对齐 MarkStrideAlign → EnableStrideAlign → LiftLowestStride → ReduceRankSubview
阶段 5 展平分解    HIVMFlattenOps → HIVMAggregatedDecomposeOp(after-hivm-flatten-ops)
                  → LiftZeroRank → ComposeCollapseExpand
阶段 6 缓冲与内存  CloneTensorEmpty → AllocExtraBuffer → AutoInferBufferSize
                  → MarkMultiBuffer → EnableMultiBuffer → CloneSCFIfYieldOperand
                  → NormalizeLoopIterator → PlanMemory
```

### LowerHIVM（buildLowerHIVMPipelines，含上述全部 + 以下）

```
阶段 7 同步注入    InjectSync → InjectBlockSync → GraphSyncSolver → CrossCoreGSS
阶段 8 Mix 处理    SplitMixKernel → MarkRealCoreType → SplitMixedIfConditionals
                  → InsertLoadStoreForMixCV → InsertCVTightCoupledBuffer
                  → InsertWorkSpaceForMixCV → InsertCVDataMovement → CVPipelining
阶段 9 后端转换    [SIMD] ConvertHIVMToTritonGPU → TritonGPU Passes → ConvertTritonAscendGPUToLLVM
                  [SIMT] ArithToHIVMLLVMConversionPass → GPUToDPX → FuncToLLVM
                  [公共] LowerMesh → LowerMemRefExt → StripMemRefAddressSpace
                        → InsertMemSemanticForSimtVF
```

`last_pass.mlir`（提取见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)）对应阶段 7 之后到阶段 9 LLVM lowering 之前的 IR：dump flag 回退项 `--bishengir-print-ir-after=hivm-inject-sync` 即锚定阶段 7 的 InjectSync；全量 dump 时取最后一个 `IR Dump After` 段，内容含 `hivm.hir.*` 或 `llvm.*`。

## Triton IR 编译链

```
阶段 1 TTIR 优化   TritonCombineOps → TritonReorderBroadcast → TritonRewriteTensorPointer
                  → TritonRewriteTensorDescriptorToPointer → (TritonLoopUnroll)
                  → TritonLoopInvariantCodeMotion → TritonLoopAwareCSE
阶段 2 TTIR→TTGIR  ConvertTritonToTritonGPU（target/num-warps/threads-per-warp）
阶段 3 TTGIR 优化  TritonGPUCoalesce → TritonGPURemoveLayoutConversions
                  → TritonGPUAccelerateMatmul → TritonGPUOptimizeDotOperands
                  → TritonGPUReduceDataDuplication → TritonGPUOptimizeThreadLocality
                  → TritonGPUReorderInstructions → TritonGPUCoalesceAsyncCopy
                  → TritonGPUCombineTensorSelectAndIf → TritonGPUOptimizeAccumulatorInit
                  → (TritonGPUF32DotTC)
阶段 4 流水线化    TritonGPUAssignLatencies → TritonGPUScheduleLoops → TritonGPUPipeline
                  → (TritonGPUPrefetch) → (TritonGPUFuseNestedLoops)
阶段 5 Warp 特化   TritonGPUPartitionScheduling → TritonGPULoadMMASpecialization
                  → TritonGPUAutomaticWarpSpecialization → TritonGPURewritePartitionDependencies
                  → TritonGPUPartitionLoops → TritonGPUOptimizePartitionWarps
                  → TritonGPUHoistTMEMAlloc → RelayoutTritonGPU
阶段 6 内存分配    AllocateSharedMemory → TritonGPUGlobalScratchAllocationPass
                  → TritonGPUAllocateWarpGroups
阶段 7 LLVM 转换   ConvertTritonGPUToLLVM / ConvertTritonAscendGPUToLLVM
```

## 必须顺序执行的 Pass 对

| 前置 | 后续 | 原因 |
|------|------|------|
| `InferFuncCoreType` | `InferHIVMMemScope` | 内存作用域推断需要 Core 类型 |
| `InferHIVMDataLayout` | `InsertConvertLayout` | 布局转换插入需要布局信息 |
| `MarkStrideAlign` | `EnableStrideAlign` | 先标记后启用 |
| `MarkMultiBuffer` | `EnableMultiBuffer` | 先标记后启用（见 [multi_buffer](../memory_management/multi_buffer.md)） |
| `PlanMemory` | `InjectSync` | 同步注入需要内存规划完成（规划见 [memory_planning](../memory_management/memory_planning.md)，注入见 [sync_sync_injection](../hivm/sync_sync_injection.md)） |
| `SplitMixKernel` | `CVPipelining` | 分裂后才能 CV 流水线化 |
| `ConvertTritonToTritonGPU` | `TritonGPUCoalesce` | 合并访问需要布局编码 |
| `TritonGPUAccelerateMatmul` | `TritonGPUPipeline` | dot 布局优化后才能流水线化 |
| `AllocateSharedMemory` | `ConvertTritonGPUToLLVM` | LLVM 转换需要 shared memory 偏移 |

## 可选 Pass 的条件启用

| Pass | 启用条件（来源选项） |
|------|----------------------|
| `SplitMixKernel` / `CVPipelining` | `enable-mixed-cv=true`（HIVMPipelineOptions） |
| `GraphSyncSolver` | `enable-hivm-graph-sync-solver=true` |
| `ConvertHIVMToTritonGPU` | `pure-simt=false` |
| `GPUToDPX` | `use-dpx=true` |
| `TritonGPUF32DotTC` | FP32 dot + Tensor Core（自动检测） |
| `TritonGPUAutomaticWarpSpecialization` | Hopper+ 架构（自动检测） |

## 依赖关系图（压缩版）

```
HFusion: ConvertGenericToNamedOp → HFusionOpFusion → AutoSchedule → AutoVectorize
                                     ↓ ComposeMultiReduce → DecomposeMulti → HFusionInlineBrc
                                     ↓ FlattenOps → HFusionDecompose
                                     ↓ OutlineVectorFunction → MergeVecScope
                                     ↓ PackTilingData → ConstantizeTilingData

HIVM:    InferFuncCoreType → InferHIVMMemScope → InferVFMode → InferHIVMDataLayout
         → Normalize → HIVMDecomposeOp → HIVMFlattenOps
         → InsertConvertLayout → PropagateConvertLayout
         → MarkStrideAlign → EnableStrideAlign → LiftLowestStride
         → MarkMultiBuffer → EnableMultiBuffer → PlanMemory
         → InjectSync → InjectBlockSync → CrossCoreGSS
         → SplitMixKernel → CVPipelining

TTGIR:   ConvertTritonToTritonGPU → TritonGPUCoalesce → TritonGPURemoveLayoutConversions
         → TritonGPUAccelerateMatmul → TritonGPUOptimizeDotOperands
         → TritonGPUAssignLatencies → TritonGPUScheduleLoops → TritonGPUPipeline
         → TritonGPUAutomaticWarpSpecialization → TritonGPUPartitionLoops
         → AllocateSharedMemory → TritonGPUAllocateWarpGroups → ConvertTritonGPUToLLVM
```
