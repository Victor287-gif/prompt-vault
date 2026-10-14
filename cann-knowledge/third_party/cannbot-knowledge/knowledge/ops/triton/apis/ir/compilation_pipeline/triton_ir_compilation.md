---
type: API
title: Triton IR 编译路径（TTIR → TTGIR → LLVM 与昇腾适配）
description: Triton 原生编译链各阶段 Pass：TTIR 变换（triton-combine、triton-licm 等）、ConvertTritonToTritonGPU
  选项、TTGIR 布局/矩阵乘/流水线/warp 特化 Pass、共享内存分配，及昇腾适配与 bishengir-compile、last_pass.mlir
  的关系。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Triton IR 编译路径

Triton 原生编译链为 `TTIR →（Triton 变换）→ ConvertTritonToTritonGPU → TTGIR →（TritonGPU 变换）→ TritonGPUToLLVM → LLVM IR`。昇腾场景下该链与 BiShengIR 路径交汇：既可直接以昇腾适配版 lowering 收尾，也可由 HIVM 经 SIMD 路径转回 TritonGPU 后复用本链后段（见 [hivm_to_backend.md](hivm_to_backend.md)）。方言操作语义见 [overview](../triton_dialects/overview.md)。

## TTIR 阶段变换

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| TritonCombineOps | `triton-combine` | 五种合并模式：`dot(a,b,0)+c→dot(a,b,c)`、连续 addptr 合并、条件 load 简化、常量广播简化、`sum(expand*expand)→dot` |
| TritonReorderBroadcast | `triton-reorder-broadcast` | broadcast/splat 移到逐元素操作之后：`elementwise(broadcast(a))→broadcast(elementwise(a))` |
| TritonRewriteTensorPointer | `triton-rewrite-tensor-pointer` | 张量指针 load/store 重写为传统 load/store，消除 `tt.make_tensor_ptr`/`tt.advance` |
| TritonRewriteTensorDescriptorToPointer | `triton-rewrite-tensor-descriptor-to-pointer` | 张量描述符重写为指针 load/store，消除 `tt.make_tensor_descriptor` |
| TritonLoopUnroll | `triton-loop-unroll` | 按 `tt.loop_unroll_factor` 属性展开循环 |
| TritonLoopInvariantCodeMotion | `triton-licm` | 增强 LICM：带掩码 load 可外提；scf.for 生成行程计数检查、scf.while 克隆条件 |
| TritonLoopAwareCSE | `triton-loop-aware-cse` | 循环体内 CSE，递归消除迭代参数与恒等子计算 |

## ConvertTritonToTritonGPU（`convert-triton-to-tritongpu`，ModuleOp）

| 选项 | 默认 | 说明 |
|------|------|------|
| `target` | `""` | GPU 目标，如 `cuda:80`、`hip:gfx942` |
| `num-warps` | 4 | warp 数量 |
| `threads-per-warp` | 32 | 每 warp 线程数 |
| `num-ctas` | 1 | CGA 中 CTA 数 |
| `enable-source-remat` | false | 源重物化 |
| `shared-memory-size` | 122880 | SIMT 共享内存大小（昇腾条件编译） |

转换语义：张量/指针类型附加布局编码（`tensor<shape, type, #encoding>`），`tt.*` 操作映射后新增 `ttg.convert_layout` 表达布局转换。配套 `relayout-tritongpu`（warp 特化期间重布局）。编码体系见 [tritongpu_encodings](../triton_dialects/tritongpu_encodings.md)。

## TTGIR 阶段变换

### 布局优化

- **TritonGPUCoalesce**（`tritongpu-coalesce`）：load/store 替换为合并访问布局（cache friendly）。
- **TritonGPURemoveLayoutConversions**（`tritongpu-remove-layout-conversions`）：移除多余布局转换，偏好 `BlockedEncodingAttr`（合并访问）与 `NvidiaMmaEncodingAttr`（张量操作）。
- **TritonGPUOptimizeDotOperands**（`tritongpu-optimize-dot-operands`）：重排 dot 操作数布局促进硬件转置；`hoist-layout-conversion`（默认 true）。
- **TritonGPUReduceDataDuplication**：`convert[distributed→dotOperand]` 分解为 `convert[distributed→shared→dotOperand]`。
- **TritonGPUCoalesceAsyncCopy**：异步拷贝合并访问，按共享编码 vec 裁剪 sizePerThread。

### 矩阵乘优化

- **TritonGPUAccelerateMatmul**（`tritongpu-accelerate-matmul`）：dot 输入/输出布局适配硬件加速器。
- **TritonGPUF32DotTC**：3xTF32 技巧，FP32 dot 分解为 4 个逐元素操作 + 3 个 FP16 dot。
- **TritonGPUOptimizeAccumulatorInit**：累加器零初始化替换为首次使用标志。

### 流水线

- **TritonGPUPipeline**（`tritongpu-pipeline`）：软件流水线化，load 转异步 + 多缓冲；`num-stages`（默认 3）、`dump-intermediate-steps`。
- **TritonGPUAssignLatencies** / **TritonGPUScheduleLoops**：流水线前延迟分配与循环调度。
- **TritonGPUPrefetch**（`tritongpu-prefetch`）：预取 dot 操作数，循环 prologue 预取首次迭代、扩展循环参数携带预取值。
- **TritonGPUFuseNestedLoops**：融合嵌套循环便于流水线化。

### Warp 特化

TritonGPUAutomaticWarpSpecialization（`num-stages`）、TritonGPUPartitionScheduling、TritonGPULoadMMASpecialization（异步 load 与异步 MMA 分区分离）、TritonGPURewritePartitionDependencies（分区间经共享内存传数据 + 多缓冲）、TritonGPUPartitionLoops、TritonGPUOptimizePartitionWarps、TritonGPUHoistTMEMAlloc（`hoist-out-of-if`）。

### 其他

TritonGPUOptimizeThreadLocality（归约/gather 布局优化减少线程同步开销）、TritonGPUReorderInstructions（降寄存器压力 + ptxas 友好序；昇腾条件编译选项 `enable-simt-reorder-instruction`）、TritonGPUCombineTensorSelectAndIf。

## TritonGPU → LLVM 转换

- **AllocateSharedMemory**（`allocate-shared-memory`）：ModuleAllocation 分析后为模块标注共享/本地内存总量、为操作标注偏移。
- **TritonGPUGlobalScratchAllocationPass**（`tritongpu-global-scratch-memory-allocation`）：全局 scratch 分配。
- **TritonGPUAllocateWarpGroups**（`tritongpu-allocate-warp-groups`）：分析 `ttg.warp_specialize`，确定总 warp 数并给每个 warpgroup 函数附加 warp ID 范围。

## 典型执行序列

标准链：`triton-combine` → `triton-reorder-broadcast` → `triton-rewrite-tensor-pointer` → （可选 `triton-loop-unroll`）→ `triton-licm` → `triton-loop-aware-cse` → `convert-triton-to-tritongpu` → coalesce / remove-layout-conversions / accelerate-matmul / optimize-dot-operands / reduce-data-duplication / optimize-thread-locality / reorder-instructions → `tritongpu-pipeline` → `allocate-shared-memory` → global-scratch / warp-groups 分配 → `ConvertTritonGPUToLLVM`。完整顺序与依赖见 [pass_dependencies.md](pass_dependencies.md)。

## 昇腾适配与 IR 提取的关系

昇腾适配编译时本链有两种接法：

1. 直接适配：`convert-triton-to-tritongpu`（`shared-memory-size=122880`）→ coalesce → accelerate-matmul（用 `FractalSharedEncodingAttr`）→ reorder-instructions（`enable-simt-reorder-instruction`）→ `ConvertTritonAscendGPUToLLVM`。
2. HIVM 汇入：前端 → HFusion → HIVM → `ConvertHIVMToTritonGPU`（SIMD 路径）→ TritonGPU Passes → `ConvertTritonAscendGPUToLLVM`。

`run_and_extract.sh` 提取 IR 时以 `bishengir-compile` 驱动整条 BiShengIR 链（固定参数含 `--enable-triton-kernel-compile=true` 等），dump flag 按 `--mlir-print-ir-after-all` > `--bishengir-print-ir-after=hivm-inject-sync` > `--print-after-all` 探测；环境变量 `BISHENGIR_PRINT_IR_FLAG` 可强制指定 dump flag，优先级高于上述自动探测。产出的 `last_pass.mlir` 取最后一个 `IR Dump After` 段，对应 BiShengIR 链最末阶段（HIVM 同步注入 `hivm-inject-sync` 之后到 LLVM lowering 之间），内容为 `hivm.hir.*` 或 `llvm.*` 操作。同步注入 Pass 语义见 [sync_sync_injection](../hivm/sync_sync_injection.md)，提取流程与环境变量详见 [../ir_extraction.md](../../../guides/ir/ir_extraction.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/Transforms/Passes.td)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Conversion/TritonGPUToLLVM/Passes.td)
