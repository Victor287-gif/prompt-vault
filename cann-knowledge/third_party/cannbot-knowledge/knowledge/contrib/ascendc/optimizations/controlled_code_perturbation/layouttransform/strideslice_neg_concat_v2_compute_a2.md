---
type: Optimization
title: "strideslice_neg_concat_v2：计算优化"
description: "strideslice_neg_concat_v2 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.tiling", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/docs/StrideSliceNegConcatV2.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/framework/tf_plugin/tensorflow_strideslice_neg_concat_v2_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/op_host/strideslice_neg_concat_v2.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/op_host/strideslice_neg_concat_v2_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/op_kernel/strideslice_neg_concat_v2.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/strideslice_neg_concat_v2/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# strideslice_neg_concat_v2：计算优化

## 来源信息

- experience ID：`044825391bd85e73`
- 源 schema version：`v2`
- 算子：`strideslice_neg_concat_v2`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`compute.redundant.vector.init`
- canonical family：`compute.redundant.vector.init`
- 源记录时间：`2026-04-14T21:02:38`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Pre-loop vector initialization for constant multiplication vectors
~~~~

### Original summary

~~~~text
Initialize constant multiplication vectors once in the pre-loop setup phase instead of inside per-tile compute loops. This optimization reduces redundant vector initialization operations that would otherwise be repeated for each tile, improving vector unit utilization and reducing overall floating-point operations. Apply this pattern when using constant vectors for element-wise operations across multiple tiles.
~~~~

### Original root mechanism

~~~~text
Moving constant vector initialization outside compute loops reduces redundant scalar operations and vector FLOPS per tile
~~~~

### Original causal chain

~~~~text
Moving Duplicate() vector initialization from Compute() to Process() -> reduces aiv_vec_fops by 25% (9.44M to 12.58M) and aiv_vec_misc_ratio_mean by 217% (0.031 to 0.098) -> decreases task duration by 8.7% (2779us to 2538us)
~~~~

## 适用条件

- Operator uses constant vectors for element-wise operations (e.g., Mul with constant vector)
- Vector initialization uses Duplicate() or similar broadcast operations
- Compute loops process multiple tiles with same constant vector
- Vector buffer (TBuf) is allocated once in Init() and reused across tiles

## 失效边界

- Vector values need per-tile recomputation or vary between tiles
- Memory constraints prevent pre-initialization of all constant vectors
- Vector size exceeds available on-chip buffer capacity
- Operator uses dynamic vector values that change per iteration

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/strideslice_neg_concat_v2.cpp

### 改动符号

- StridesliceNegConcatV2::Compute

### 代码签名

- AllocTensor
- DeQue
- EnQue
- FreeTensor
- TQue
- contiguous_access
- queue_api
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`8ea92f6b2b837b06d0f770fc32836f565256ff2d4969ced32928253696a64103`

````diff
--- Before/op_kernel/strideslice_neg_concat_v2.cpp
+++ After/op_kernel/strideslice_neg_concat_v2.cpp
@@ -22,19 +22,12 @@

      __aicore__ inline void Process()
     {
-        // 构造[1;-1]向量
-        LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>();
-        T scalar1 = 1.0;
-        T scalar2 = -1.0;
-        uint32_t repeatNum = this->tileLength / 2;
-        Duplicate(mulVecBufLocal, scalar1, repeatNum);
-        Duplicate(mulVecBufLocal[repeatNum], scalar2, repeatNum);
         uint32_t loopCount = 0;
         if (GetBlockIdx() == GetBlockNum() - 1) {
-	    	loopCount = this->tileNumLast;
-    	} else {
-    		loopCount = this->tileNumAverage;
-    	} for (int32_t i = 0; i < loopCount; i++) {
+ 	    	loopCount = this->tileNumLast;
+     	} else {
+     		loopCount = this->tileNumAverage;
+     	} for (int32_t i = 0; i < loopCount; i++) {
             CopyIn(i);
             Compute(i);
             CopyOut(i);
@@ -49,10 +42,16 @@
         inQueueInput.EnQue(inputLocal);
     }

-    __aicore__ inline void Compute(int32_t progress)
+     __aicore__ inline void Compute(int32_t progress)
     {
         LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
         LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>();
+        // Degradation: Initialize multiplication vector inside compute loop
+        T scalar1 = 1.0;
+        T scalar2 = -1.0;
+        uint32_t repeatNum = this->tileLength / 2;
+        Duplicate(mulVecBufLocal, scalar1, repeatNum);
+        Duplicate(mulVecBufLocal[repeatNum], scalar2, repeatNum);
         LocalTensor<T> outputLocal = outQueueOutput.AllocTensor<T>();
         Mul(outputLocal, inputLocal, mulVecBufLocal, this->tileLength);
         inQueueInput.FreeTensor(inputLocal);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2779.315674`
- fast duration us：`2538.490723`
- improvement ratio：`0.08664900977347564`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.030854 -> 0.097976 (+217.55%)
- memory_ub.ub_write_bw_vector_gbps: 6.21 -> 11.35 (+82.77%)
- aiv_utilization.vec_ratio_mean: 0.092536 -> 0.146963 (+58.82%)
- arithmetic.aiv_vec_fops: 9.43763e+06 -> 1.2583e+07 (+33.33%)
- arithmetic.aiv_vec_fp16_ratio_mean: 0.061682 -> 0.048986 (-20.58%)

### Affected metric groups

- arithmetic
- memory_ub
- aiv_utilization
- latency

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_misc_ratio_mean",
  "memory_ub.ub_write_bw_vector_gbps",
  "aiv_utilization.vec_ratio_mean",
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_fp16_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.185867,
  "overall_deviation": 0.074347,
  "metrics_checked": 1,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- Move Duplicate() calls for constant vector initialization from Compute() method to Process() method in op_kernel/strideslice_neg_concat_v2.cpp
- Initialize constant vectors once before the tile processing loop, not inside Compute()
- Use TBuf buffers allocated in Init() to store pre-initialized constant vectors
- Access pre-initialized vectors via mulVecBuf.Get<T>() in Compute() without re-initialization
- Apply to similar patterns where constant vectors are used with Mul, Add, Sub operations

## 检索线索

- vector initialization
- Duplicate
- constant vector
- pre-loop setup
- redundant computation
- scalar overhead
- Mul operation
- element-wise
- vector buffer
- TBuf
- AscendC
- tile processing
