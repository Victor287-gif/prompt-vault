---
type: Optimization
title: "gcd：冗余消除优化"
description: "gcd 算子的冗余消除优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/Gcd.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gcd：冗余消除优化

## 来源信息

- experience ID：`3a8293c83401cbf3`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_018_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-20T22:09:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating redundant reinterpret casts and no-op arithmetic operations in vector kernels
~~~~

### Original summary

~~~~text
Removing unnecessary reinterpret casts and zero-add operations in vector kernels reduces pipeline stalls and improves performance. The redundant operations increase vector floating-point operations (FLOPS) by 30.7% and vector functional unit utilization by 4.4%, causing bank group conflict stalls that degrade latency by 17.5%. Apply this optimization when you identify redundant type conversions or no-op arithmetic in tight loops.
~~~~

### Original root mechanism

~~~~text
Redundant reinterpret casts and subsequent no-op Adds operations consume vector functional units and increase pipeline stalls, particularly bank group conflicts, even though they don't affect computation results.
~~~~

### Original causal chain

~~~~text
Removing 7 redundant reinterpret casts and their 7 corresponding no-op Adds operations → reduces aiv_vec_fops by 31.2% and aiv_vec_total_cflt_ratio_mean by 14.9% → improves task_duration_us by 14.9% (261.8ms to 307.6ms).
~~~~

## 适用条件

- When vector kernels contain redundant reinterpret casts like a_div2.template ReinterpretCast<FpT>()
- When no-op arithmetic operations (Adds with zero) follow reinterpret casts
- When profiling shows high aiv_vec_total_cflt_ratio_mean (>0.15) and aiv_vec_fp32_ratio_mean (>0.4)
- When the BinaryGcd function or similar vector algorithms use temporary tensors with unnecessary type conversions

## 失效边界

- When reinterpret casts are necessary for type-safe operations between different data types
- When the compiler cannot optimize away redundant operations automatically
- When the code is not in a performance-critical loop or kernel

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- gcd/op_kernel/gcd.cpp

### 改动符号

- BinaryGcd

### 代码签名

- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`d5fdcdd5d0b70c789a6cd0545dd6d0ece9e7216643f7e7d1003cc7bbabf9d802`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -208,16 +208,6 @@
         auto mask_u16 = mask.template ReinterpretCast<uint16_t>();
         auto mask_not_u16 = mask_not.template ReinterpretCast<uint16_t>();

-        // Redundant reinterpret casts to degrade performance
-        auto a_div2_redundant = a_div2.template ReinterpretCast<FpT>();
-        auto b_div2_redundant = b_div2.template ReinterpretCast<FpT>();
-        auto ab_diff_redundant = ab_diff.template ReinterpretCast<FpT>();
-        auto zeros_redundant = zeros.template ReinterpretCast<FpT>();
-        auto ones_redundant = ones.template ReinterpretCast<FpT>();
-        auto a0_redundant = a0.template ReinterpretCast<FpT>();
-        auto b0_redundant = b0.template ReinterpretCast<FpT>();
-        auto c_redundant = c.template ReinterpretCast<FpT>();
-
         Not(a0_u16, a_u16, num_uint16 * TILE_SIZE);
         Not(b0_u16, b_u16, num_uint16 * TILE_SIZE);
         Adds(a0, a0, (U)1, TILE_SIZE);
@@ -240,16 +230,6 @@
             Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
             Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight

-            // Additional redundant operations using reinterpreted tensors to degrade performance
-            Adds(a_div2_redundant, a_div2_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(b_div2_redundant, b_div2_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(ab_diff_redundant, ab_diff_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(zeros_redundant, zeros_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(ones_redundant, ones_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(a0_redundant, a0_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(b0_redundant, b0_redundant, (FpT)0.0f, TILE_SIZE);
-            Adds(c_redundant, c_redundant, (FpT)0.0f, TILE_SIZE);
-
             And(a_u16, a0_u16, ones... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`307637.59375`
- fast duration us：`261830.34375`
- improvement ratio：`0.14890003995163545`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 1.01686e+10 -> 1.32953e+10 (+30.75%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.417929 -> 0.526656 (+26.02%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.127116 -> 0.100041 (-21.30%)
- critical_path.aiv_vector0_time_us_mean: 261824 -> 307630 (+17.50%)
- latency.task_duration_us: 261830 -> 307638 (+17.50%)

### Affected metric groups

- arithmetic
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_fops": {
      "before": 10168633536.0,
      "after": 13295290560.0,
      "delta": 3126657024.0,
      "rel": 0.30748054917415407
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.417929,
      "after": 0.526656,
      "delta": 0.10872700000000002,
      "rel": 0.26015662947534157
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.127116,
      "after": 0.100041,
      "delta": -0.027075000000000002,
      "rel": -0.2129944302841499
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 261823.706,
      "after": 307629.836,
      "delta": 45806.130000000005,
      "rel": 0.17495027742063968
    }
  },
  {
    "latency.task_duration_us": {
      "before": 261830.344,
      "after": 307637.594,
      "delta": 45807.24999999997,
      "rel": 0.1749501196087493
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

- Search for patterns of ReinterpretCast followed by Adds with zero in vector kernels (gcd/op_kernel/gcd.cpp:208-230)
- Remove redundant reinterpret casts of already correctly typed tensors (a_div2_redundant, b_div2_redundant, etc.)
- Eliminate no-op Adds operations that use the reinterpreted tensors (Adds with (FpT)0.0f)
- Verify that vector functional unit utilization (vec_ratio_mean) decreases while performance improves
- Check that aiv_vec_fops reduces by approximately 30% and aiv_vec_total_cflt_ratio_mean reduces by 15%

## 检索线索

- reinterpret_cast
- redundant_operations
- vector_kernel
- pipeline_stalls
- bank_group_conflicts
- no_op_adds
- type_conversion
- functional_unit_utilization
- gcd
- binary_gcd
- ascend_c
- vector_optimization
