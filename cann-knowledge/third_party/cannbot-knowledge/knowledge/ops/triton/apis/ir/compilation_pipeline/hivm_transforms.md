---
type: API
title: HIVM 变换 Pass（推断、多缓冲、内存规划、同步注入与 Mix 分裂）
description: HIVM 方言 70+ 变换 Pass 分类清单：InferFuncCoreType/InferHIVMMemScope/InferHIVMDataLayout
  推断链、MarkMultiBuffer+EnableMultiBuffer、PlanMemory、InjectSync/GraphSyncSolver/CrossCoreGSS
  同步链、SplitMixKernel 与 CVPipelining、布局转换与 Stride 对齐。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 变换 Pass

HIVM 阶段是硬件映射的核心：类型与内存推断、多缓冲、内存规划、同步注入、Mix kernel 分裂都在此完成。`last_pass.mlir` 中观察到的 `hivm.hir.*` 形态（地址空间标注、set_flag/wait_flag、多偏移 pointer_cast）均由本阶段 Pass 产生。方言操作语义见 [overview](../hivm/overview.md)，执行顺序见 [pass_dependencies.md](pass_dependencies.md)。

## 类型推断与规范化

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| InferFuncCoreType | `hivm-infer-func-core-type` | 推断函数 Core 类型（CUBE/VECTOR/Mix），枚举见 [overview](../hivm/overview.md) |
| ConvertToHIVMOp | `convert-to-hivm-op` | 其他方言操作 → HIVM 操作 |
| NormalizeMatmul | `hivm-normalize-matmul` | 规范化矩阵乘操作 |
| Normalize | `hivm-normalize-ops` | 规范化 HIVM 操作 |
| TritonGlobalKernelArgsToHIVMOp | `triton-global-kernel-args-to-hivm-op` | Triton 全局 kernel 参数 → HIVM 操作，依赖 hacc、annotation（见 [kernel_args](../hacc/kernel_args.md)） |

## 内存作用域与布局推断

- **InferHIVMMemScope**（`hivm-infer-mem-scope`）：推断操作的内存作用域（GM/L1/UB/L0x 等地址空间）。
- **InferHIVMDataLayout**（`hivm-infer-data-layout`）：推断数据布局（ND/Fractal 等，布局背景见 [data_layout](../architecture/data_layout.md)）。
- **InferVFMode**（`hivm-infer-vf-mode`）：推断 VF 模式（SIMD/SIMT/MIX）。

## 多缓冲（标记 → 启用两段式）

- **MarkMultiBuffer**（`hivm-mark-multi-buffer`）：标记 `hivm.multi_buffer` 属性，L0C 缓冲不标记。选项 `enable-auto`、`limit-auto-multi-buffer-only-for-local-buffer`、`limit-auto-multi-buffer-of-local-buffer`（no-limit/no-l0c）、`limit-mix-auto-multi-buffer-buffer`（only-cube/only-vector/no-limit）、`set-workspace-multibuffer`（默认 2）。
- **EnableMultiBuffer**（`hivm-enable-multi-buffer`）：按标记把单缓冲分配改为多缓冲分配。

机制与 IR 形态详解见 [multi_buffer](../memory_management/multi_buffer.md)。

## 内存规划

- **PlanMemory**（`hivm-plan-memory`，ModuleOp）：规划内存分配。选项 `mem-plan-mode`（`local-mem-plan` = memref.alloc / `global-work-space-plan` = memref_ext.alloc_workspace）、`enable-global-workspace-reuse`、`enable-print-memory-allocated-size`、`restrict-inplace-as-isa`、`simt-vf-dynamic-size`（SIMT VF 动态 UB 大小，默认 216 KB）、`disable-tightly-coupled-buffer-reuse`。详解见 [memory_planning](../memory_management/memory_planning.md)。
- 配套：CloneTensorEmpty、CloneSCFIfYieldOperand（避免 PlanMemory 就地操作）、AllocExtraBuffer（见 [extra_buffer](../memory_management/extra_buffer.md)）、ConstantizeBufferSize、SetBufferSize、AutoInferBufferSize、AlignAllocSize（对齐要求见 [memory_alignment](../memory_management/memory_alignment.md)）、HIVMOptFuncOutput。

## 同步注入

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| InjectSync | `hivm-inject-sync` | 启发式同步注入；`sync-mode`（normal/barrier-all）、`enable-unit-flag`、`assume-alive-loops` |
| InjectBlockSync | `hivm-inject-block-sync` | 块级同步注入；`block-all-sync`、`assume-alive-loops` |
| GraphSyncSolver | `hivm-graph-sync-solver` | 基于图的同步求解；`enable-unit-flag`、`enable-tester-mode`、`sync-tester-options` |
| CrossCoreGSS | `hivm-cross-core-gss` | 跨 Core 图同步求解；`block-all-sync`、`disable-auto-inject-block-sync`、`always-use-pipe-s`、`use-different-multibuffer-flag-ids`、`force-is-mem-based`/`force-is-reg-based` |
| SyncBlockHoisting | `hivm-sync-block-hoisting` | sync block 锁/解锁提升到父区域 |
| LowerCreateSyncBlockLock | `hivm-lower-create-sync-block-lock` | CreateSyncBlockLockOp → ViewOp |
| InsertInferSyncBlockLockNumAndInitFunc | `hivm-insert-infer-sync-block-lock-num-and-init-func` | 插入推断锁数量的 host 回调 |

InjectSync 与 GraphSyncSolver 的对比、选择逻辑与变换效果见 [sync_sync_injection](../hivm/sync_sync_injection.md)；同步操作语义见 [sync_overview](../hivm/sync_overview.md)。

## 算子分解与变换

- **HIVMDecomposeOp**（`hivm-decompose-op`）：按硬件能力分解复合操作（如 f32→i8 分解为 f32→f16→i8）。
- **HIVMAggregatedDecomposeOp**（`hivm-aggregated-decompose-op`）：分解 AggregatedOpInterface 操作；`decompose-phase` 可取 `no-constraint`/`before-hivm-align`/`after-hivm-recognize-deinterleave`/`after-hivm-recognize-broadcast`/`after-hivm-align`/`after-infer-hivm-data-layout`/`after-hivm-flatten-ops`。
- 其他：HIVMLowerToLoops、HIVMRecognizeDeinterleaveOp（非连续访存 deinterleave 优化）、HIVMOptSinglePointOp（单点操作标量化，见 [vector_scalar_lowering](../hivm/vector_scalar_lowering.md)）、HIVMFlattenOps、ComposeCollapseExpand。

## Stride 对齐

MarkStrideAlign（`hivm-mark-stride-align`，标注 stride_align）→ EnableStrideAlign（`hivm-enable-stride-align`，按标记重分配 memref）；LiftLowestStride（`hivm-lift-lowest-stride`，提升最低步幅，如 `memref<16xf16, strided<[8]>>` → `memref<16x1xf32, strided<[8, 1]>>`，MacroOp 与 VArangeOp 例外）；ReduceRankSubview。

## Mix Kernel（Cube+Vector）

- **SplitMixKernel**（`hivm-split-mix-kernel`，ModuleOp）：把 Mix 设备函数分裂为 AICube / AIVector 两个函数并标记父模块为 Mix 模块，二者分别走 SIMD / SIMT 路径，共享 workspace 传数据（见 [hivm_to_backend.md](hivm_to_backend.md)）。
- 配套：MarkRealCoreType（`remove-core-type-attrs` 时变为清理 Pass）、SplitMixedIfConditionals、InsertLoadStoreForMixCV、InsertCVTightCoupledBuffer（紧耦合缓冲见 [tightly_coupled_buffer](../memory_management/tightly_coupled_buffer.md)）、InsertWorkSpaceForMixCV（workspace 见 [workspace_management](../memory_management/workspace_management.md)）、InsertCVDataMovement。
- **CVPipelining**（`cv-pipelining`）：Cube/Vector 核间流水线化，配合多缓冲；`pipeline-depth`（-1 自动）、`enable-auto-balance`（已弃用）。

## 布局转换

InsertConvertLayout（`hivm-insert-convert-layout`，为矩阵乘插入布局转换）→ PropagateConvertLayout（`allow-agnostic-ops` 允许布局无关操作传播 reshape）→ ConvertLayoutToTranspose（布局转换分解为转置）→ CombineOptimizedConvertLayout（折叠优化布局转换模式）。ND↔NZ 转换操作语义见 [dma_nd2nz](../hivm/dma_nd2nz.md) 与 [dma_nz2nd](../hivm/dma_nz2nd.md)。

## 外联与函数管理

OutlineAllocInVF（外联 VF 中静态形状 memref.alloc）、OutlineCopyInVF（VF 参数时 hir.load → hir.copy）、InsertInferWorkSpaceSizeFunc、InsertInferVFModeFunc、BindWorkSpaceArg（hacc.workspace 参数绑定 AllocWorkspaceOp）、BindSyncBlockLockArg、SplitSimtModule（`split-simt-module`）。hacc 侧绑定机制见 [host_device_binding](../hacc/host_device_binding.md)。

## 其他变换

InlineOTFBroadcast、InitEntryKernel（入口 kernel 开头插 set_mask_norm()）、InlineFixpipe/InlineFixpipeV2（fixpipe 见 [dma_fixpipe](../hivm/dma_fixpipe.md)）、TileBatchMMIntoLoop（batch 维 tiling 为循环）、LiftZeroRank、InsertLoadStoreForScalar、HIVMMapForallToBlocks（scf.forall → 块操作，归纳变量重写为 block idx）、SinkOpToConsumerInLoop、NormalizeLoopIterator（PlanMemory 前规范化循环迭代器）、HIVMInlineOTFLoadStore、AnnotateVFAlias、RemoveCopyOps、AnalyzeArithVectorMask、TileAndBindSubBlock（`hivm-bind-sub-block`，`strict-mode`/`enable-tile`）、HIVMBubbleUpExtractSlice、HIVMVectorizeOps、MarkDisableLoad（标记禁用 dcache 的 memref.load）、InsertNZ2NDForDebug / InsertL12UBForDebug（调试）、AutoBlockifyParallelLoop（逻辑块数 > 物理块数时自动循环）、InferSimtVFMemEffect、StripMemRefAddressSpace、InsertMemSemanticForSimtVF、AutoScope（为 gather_load/scatter_store 建 scope，见 [dma_gather_scatter](../hivm/dma_gather_scatter.md)）、InsertAllocBasePlaceholder、WriteBackShared。

各 Pass 的开关集中由 HIVMPipelineOptions 控制，见 [pipeline_options.md](pipeline_options.md)；随后进入后端 lowering，见 [hivm_to_backend.md](hivm_to_backend.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td)
