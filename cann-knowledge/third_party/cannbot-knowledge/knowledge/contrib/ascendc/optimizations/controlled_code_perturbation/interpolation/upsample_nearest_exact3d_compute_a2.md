---
type: Optimization
title: "upsample_nearest_exact3d：计算优化"
description: "upsample_nearest_exact3d 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/docs/aclnnUpsampleNearestExact3d.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_host/aclnn_upsample_nearest_exact3d.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_host/aclnn_upsample_nearest_exact3d.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_host/upsample_nearest_exact3d.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_host/upsample_nearest_exact3d_l0.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_host/upsample_nearest_exact3d_l0.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_host/upsample_nearest_exact3d_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_kernel/upsample_nearest_exact3d.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_kernel/upsample_nearest_exact3d.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/op_kernel/upsample_nearest_exact3d_310p.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_nearest_exact3d/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# upsample_nearest_exact3d：计算优化

## 来源信息

- experience ID：`78e8e0ba9ec1b73a`
- 源 schema version：`v2`
- 算子：`upsample_nearest_exact3d`
- source iteration：`round_012_refactor`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`precompute.constant`
- canonical family：`precompute.constant`
- 源记录时间：`2026-04-15T06:27:12`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Precomputed scaleWLarge flag to avoid repeated comparison
~~~~

### Original summary

~~~~text
Precomputing runtime-constant boolean comparisons as member variables reduces pipeline stalls by eliminating repeated scalar comparisons in hot loops. This optimization applies when a comparison depends only on kernel parameters and is evaluated multiple times within the same kernel execution. The precomputed boolean flag reduces instruction cache misses and improves vector unit utilization.
~~~~

### Original root mechanism

~~~~text
The comparison `scales[W_INDEX] > BEST_PERFORMANCE_SCALE` is computed once during initialization and stored as a boolean member variable `scaleWLarge`, replacing multiple runtime comparisons with a simple boolean check.
~~~~

### Original causal chain

~~~~text
Precomputing `scaleWLarge = scales[W_INDEX] > BEST_PERFORMANCE_SCALE` during ParseTilingData -> reduces scalar comparisons in GetRangeW and GatherData loops -> decreases pipeline stall ratios (aiv_vec_wait_ratio_mean from 0.368571 to 0.299503) -> reduces kernel duration from 159.483μs to 150.743μs
~~~~

## 适用条件

- Scalar comparisons depend only on kernel parameters (scales, BEST_PERFORMANCE_SCALE)
- Comparison is evaluated multiple times in hot loops (GetRangeW, GatherData)
- Comparison result remains constant throughout kernel execution
- Operator uses template-based specialization patterns

## 失效边界

- Comparison result varies per iteration or depends on runtime data
- Comparison is evaluated only once or very few times
- Memory pressure outweighs compute savings
- Boolean flag would require additional branching complexity

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- upsample_nearest_exact3d/op_kernel/upsample_nearest_exact3d.cpp

### 改动符号

- UpsampleNearestExact3dND::scaleWLarge
- UpsampleNearestExact3dND::ParseTilingData
- UpsampleNearestExact3dND::GetRangeW
- UpsampleNearestExact3dND::GatherData

### 代码签名

- GetValue
- bool
- bool_control
- compile_time_specialization
- coreNum
- core_launch
- scalar_value_access
- template

### 原始 diff

- 源 diff SHA-256：`3c00f0457190d5bae4f1df03f9cce2f8062e4c479da1a9b10136d7f68854bf62`

````diff
--- Before/op_kernel/upsample_nearest_exact3d.cpp
+++ After/op_kernel/upsample_nearest_exact3d.cpp
@@ -116,6 +116,8 @@
     int64_t indexD = 0;
     int64_t srcIndexD = 0;
     int64_t depthCount = 0;
+
+    bool scaleWLarge = false;
 };

 template <typename T>
@@ -186,7 +188,7 @@
     GetRangeW(slideIndex);
     int64_t j = 0;
     while (startW < endW) {
-        if (scales[W_INDEX] > BEST_PERFORMANCE_SCALE) {
+        if (scaleWLarge) {
             srcStartW = static_cast<int64_t>(srcIndexTensorW.GetValue(j));
             j++;
         }
@@ -253,7 +255,7 @@
         CalculateSrcIndexTensor(startW, dataCount, W_INDEX, srcIndexTensorW);
         srcStartW = static_cast<int64_t>(srcIndexTensorW.GetValue(0));
         srcEndW = static_cast<int64_t>(srcIndexTensorW.GetValue(dataCount - 1)) + 1;
-        if (scales[W_INDEX] > BEST_PERFORMANCE_SCALE) {
+        if (scaleWLarge) {
             dataCount = 1;
             srcDataCount = 1;
         } else {
@@ -405,6 +407,8 @@
     inputRow = tilingData->inputRow;
     tailAvergingRow = tilingData->tailAvergingRow;
     needCoreNum = tilingData->needCoreNum;
+
+    scaleWLarge = scales[W_INDEX] > BEST_PERFORMANCE_SCALE;
 }
 }  // namespace UpsampleNearestExact3d
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`159.483185`
- fast duration us：`150.743011`
- improvement ratio：`0.054803106672342894`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.368571 -> 0.299503 (-18.74%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.366472 -> 0.299332 (-18.32%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0036 -> 0.003 (-16.67%)
- aiv_utilization.icache_miss_rate_mean: 0.000456 -> 0.000528 (+15.79%)
- aiv_utilization.mte2_ratio_mean: 0.132432 -> 0.124407 (-6.06%)

### Affected metric groups

- pipeline_stalls
- memory_ub
- aiv_utilization
- critical_path

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.368571,
      "after": 0.299503,
      "delta": -0.06906799999999996,
      "rel": -0.18739401634963132
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.366472,
      "after": 0.299332,
      "delta": -0.06714000000000003,
      "rel": -0.18320635682944408
    }
  },
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0036,
      "after": 0.003,
      "delta": -0.0005999999999999998,
      "rel": -0.16666666666666663
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 0.000456,
      "after": 0.000528,
      "delta": 7.200000000000002e-05,
      "rel": 0.15789473684210528
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.132432,
      "after": 0.124407,
      "delta": -0.00802499999999999,
      "rel": -0.06059713664371142
    }
  }
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

- Identify scalar comparisons in hot loops of upsample_nearest_exact3d.cpp
- Check if comparison depends only on kernel parameters (scales[W_INDEX] > BEST_PERFORMANCE_SCALE)
- Add boolean member variable to UpsampleNearestExact3dND struct (scaleWLarge)
- Compute boolean value once in ParseTilingData initialization
- Replace runtime comparisons with boolean flag in GetRangeW and GatherData
- Verify pipeline_stalls.aiv_vec_wait_ratio_mean improvement in profiling

## 检索线索

- precompute
- constant_folding
- boolean_flag
- pipeline_stall
- scalar_comparison
- scaleWLarge
- BEST_PERFORMANCE_SCALE
- UpsampleNearestExact3d
- GetRangeW
- GatherData
- ParseTilingData
