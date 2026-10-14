---
type: Optimization
title: "strideslice_neg_concat_v2：循环计算优化"
description: "strideslice_neg_concat_v2 算子的循环计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.tiling", "topic.programming_model.execution_unit.vector"]
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
# strideslice_neg_concat_v2：循环计算优化

## 来源信息

- experience ID：`3a9f6ca04086c209`
- 源 schema version：`v2`
- 算子：`strideslice_neg_concat_v2`
- source iteration：`round_001_refactor`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.invariant.code.motion`
- canonical family：`loop.invariant.code.motion`
- 源记录时间：`2026-04-14T21:28:57`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Move constant multiplication vector initialization from Compute loop to Init method
~~~~

### Original summary

~~~~text
Moving constant vector initialization from the Compute loop to the Init method reduces scalar overhead by 17.33% and improves vector utilization by 61.25%. This optimization applies when multiplication vectors with constant values are recomputed in each Compute iteration despite being invariant across the kernel's lifetime. Initialize constant vectors once during Init() instead of recomputing them in Compute() to eliminate redundant scalar operations.
~~~~

### Original root mechanism

~~~~text
The multiplication vector values (1.0, -1.0) are constants and tileLength is known at initialization time, making the initialization loop-invariant
~~~~

### Original causal chain

~~~~text
Moving Duplicate() calls with constant scalars (1.0, -1.0) from Compute() to Init() → reduces aiv_vec_fops by 33.33% and aiv_vec_fp16_ratio by 19.37% → decreases scalar overhead bound → reduces task duration by 7.92% (220us)
~~~~

## 适用条件

- Constant vector values (e.g., 1.0, -1.0) are used repeatedly in Compute()
- Tile dimensions (tileLength) are known at kernel initialization time
- Vector initialization uses Duplicate() or similar broadcast operations with compile-time constants
- The same vector is reused across multiple Compute() iterations without modification

## 失效边界

- Vector values depend on dynamic inputs that vary per Compute() iteration
- Tile dimensions are not known until Compute() time
- Memory constraints require on-demand computation to reduce UB usage
- Vector values need to be recalculated based on per-tile indices or offsets

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- strideslice_neg_concat_v2/op_kernel/strideslice_neg_concat_v2.cpp

### 改动符号

- StridesliceNegConcatV2::Init
- StridesliceNegConcatV2::Compute

### 代码签名

- AllocTensor
- BUFFER_NUM
- DeQue
- FreeTensor
- InitBuffer
- TBuf
- TQue
- contiguous_access
- queue_api
- tbuf
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`a1d030efff02006509f4a4681e2fd14d1b4f4c8854c9e306f82abc79941083f1`

````diff
--- Before/op_kernel/strideslice_neg_concat_v2.cpp
+++ After/op_kernel/strideslice_neg_concat_v2.cpp
@@ -18,6 +18,14 @@
         pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(T));
         pipe.InitBuffer(outQueueOutput, BUFFER_NUM, this->tileLength * sizeof(T));
         pipe.InitBuffer(mulVecBuf, BUFFER_NUM * this->tileLength * sizeof(T));
+
+        // Initialize multiplication vector once during Init instead of in Compute loop
+        LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>();
+        T scalar1 = 1.0;
+        T scalar2 = -1.0;
+        uint32_t repeatNum = this->tileLength / 2;
+        Duplicate(mulVecBufLocal, scalar1, repeatNum);
+        Duplicate(mulVecBufLocal[repeatNum], scalar2, repeatNum);
     }

      __aicore__ inline void Process()
@@ -46,12 +54,6 @@
     {
         LocalTensor<T> inputLocal = inQueueInput.DeQue<T>();
         LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>();
-        // Degradation: Initialize multiplication vector inside compute loop
-        T scalar1 = 1.0;
-        T scalar2 = -1.0;
-        uint32_t repeatNum = this->tileLength / 2;
-        Duplicate(mulVecBufLocal, scalar1, repeatNum);
-        Duplicate(mulVecBufLocal[repeatNum], scalar2, repeatNum);
         LocalTensor<T> outputLocal = outQueueOutput.AllocTensor<T>();
         Mul(outputLocal, inputLocal, mulVecBufLocal, this->tileLength);
         inQueueInput.FreeTensor(inputLocal);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2779.315674`
- fast duration us：`2559.29126`
- improvement ratio：`0.0791649599425819`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.030389 -> 0.097976 (+222.41%)
- memory_ub.ub_write_bw_vector_gbps: 6.21 -> 11.35 (+82.77%)
- aiv_utilization.vec_ratio_mean: 0.091141 -> 0.146963 (+61.25%)
- arithmetic.aiv_vec_fops: 9.43763e+06 -> 1.2583e+07 (+33.33%)
- arithmetic.aiv_vec_fp16_ratio_mean: 0.060752 -> 0.048986 (-19.37%)

### Affected metric groups

- arithmetic
- memory_ub
- aiv_utilization
- critical_path

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
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 0,
  "metrics_wrong": 0,
  "performance_metrics_checked": 0,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- Identify Duplicate() or similar initialization calls inside Compute() methods
- Check if the scalar values are compile-time constants (e.g., 1.0, -1.0)
- Verify tile dimensions (tileLength) are available in Init() method
- Move the initialization code from Compute() to Init() method
- Use LocalTensor<T> mulVecBufLocal = mulVecBuf.Get<T>() in Init() to access the buffer
- Perform Duplicate() operations once during initialization
- Keep the buffer allocation in InitBuffer() calls as before

## 检索线索

- loop-invariant
- code-motion
- constant-initialization
- scalar-overhead
- vector-utilization
- Duplicate
- Init-method
- Compute-loop
- tileLength
- strideslice_neg_concat_v2
