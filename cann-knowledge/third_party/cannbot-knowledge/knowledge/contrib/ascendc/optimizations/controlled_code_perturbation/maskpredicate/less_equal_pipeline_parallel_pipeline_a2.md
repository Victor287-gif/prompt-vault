---
type: Optimization
title: "less_equal：流水与并行优化"
description: "less_equal 算子的流水与并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/docs/LessEqual.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/framework/tf_plugin/tensorflow_less_equal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/op_host/less_equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/op_host/less_equal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/op_kernel/less_equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/LessEqual_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/test_less_equal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# less_equal：流水与并行优化

## 来源信息

- experience ID：`a71bd4d848604c05`
- 源 schema version：`v2`
- 算子：`less_equal`
- source iteration：`round_002_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism.reduction`
- canonical family：`pipeline.parallelism.reduction`
- 源记录时间：`2026-04-20T19:40:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase BUFFER_NUM from 1 to 2 for double buffering
~~~~

### Original summary

~~~~text
For Ascend C operators with data transfer bottlenecks, setting BUFFER_NUM to 2 enables double buffering which overlaps data transfer with computation. This optimization reduces pipeline stalls by allowing the next block of data to be loaded while the current block is being processed. Apply this technique when operators show high MTE wait ratios (aiv_mte2_wait_ratio_mean > 0.99) and low MTE utilization (mte2_ratio_mean < 0.6).
~~~~

### Original root mechanism

~~~~text
Increasing BUFFER_NUM from 1 to 2 enables double buffering, allowing data transfer and computation stages to overlap and reducing pipeline stalls.
~~~~

### Original causal chain

~~~~text
Setting BUFFER_NUM=2 enables double buffering → reduces MTE wait ratios (aiv_mte2_wait_ratio_mean decreases from 0.993 to 0.990) → increases MTE utilization (mte2_ratio_mean improves from 0.555 to 0.959) → reduces pipeline stalls (aiv_vec_total_cflt_ratio_mean decreases from 0.090 to 0.060) → improves overall latency by 16.5% (1242μs to 1037μs).
~~~~

## 适用条件

- Operator uses Ascend C with pipeline parallelism
- Profile shows high MTE wait ratios (aiv_mte2_wait_ratio_mean > 0.99)
- MTE utilization is low (mte2_ratio_mean < 0.6)
- Data transfer between GM and UB is a bottleneck (gm_to_ub_bw_usage_rate_pct < 15%)
- Operator has sufficient UB memory for double buffering

## 失效边界

- Operator already has BUFFER_NUM > 2
- UB memory is severely constrained
- Data transfer bandwidth is already saturated
- Operator is compute-bound rather than memory-bound

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/less_equal.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- ub

### 原始 diff

- 源 diff SHA-256：`07f14d48ae38bae0ea149b6b659db1f31baacfbb86195daea417234966333cf3`

````diff
--- Before/op_kernel/less_equal.cpp
+++ After/op_kernel/less_equal.cpp
@@ -4,7 +4,7 @@

 namespace LessEqualK{

-constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
+constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
 constexpr float NEGATIVE_ONE_FP32 = -1.0F;
 constexpr float POSITIVE_ONE_FP32 = 1.0F;
 constexpr int32_t NEGATIVE_ONE_I32 = -1;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1242.28479`
- fast duration us：`1037.040771`
- improvement ratio：`0.16521494962519825`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.028613 -> 0.059895 (+109.33%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.060293 -> 0.089842 (+49.01%)
- aiv_utilization.mte3_ratio_mean: 0.5149 -> 0.274451 (-46.70%)
- aiv_utilization.mte2_ratio_mean: 0.958716 -> 0.554924 (-42.12%)
- aiv_utilization.scalar_ratio_mean: 0.357186 -> 0.254896 (-28.64%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.028613,
      "after": 0.059895,
      "delta": 0.031282,
      "rel": 1.0932792786495649
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.060293,
      "after": 0.089842,
      "delta": 0.029549000000000006,
      "rel": 0.49009006020599416
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.5149,
      "after": 0.274451,
      "delta": -0.24044900000000002,
      "rel": -0.46698193824043505
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.958716,
      "after": 0.554924,
      "delta": -0.40379200000000004,
      "rel": -0.42117999490985863
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.357186,
      "after": 0.254896,
      "delta": -0.10228999999999999,
      "rel": -0.28637740560940234
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

- 1. In the operator kernel file (e.g., op_kernel/less_equal.cpp), locate the BUFFER_NUM constant definition
- 2. Change BUFFER_NUM from 1 to 2 to enable double buffering
- 3. Verify the operator has sufficient UB memory allocation for double buffering
- 4. Recompile and profile to check for improvements in MTE utilization and pipeline stall metrics
- 5. Monitor aiv_mte2_wait_ratio_mean and mte2_ratio_mean for expected improvements

## 检索线索

- BUFFER_NUM
- double_buffering
- pipeline_parallelism
- MTE_wait
- memory_transfer
- Ascend_C
- less_equal
- math_operator
- pipeline_stalls
- data_transfer_overlap
