---
type: Optimization
title: "gcd：冗余消除优化"
description: "gcd 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`171ccb8dbc3b48e8`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_014_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.computations`
- canonical family：`remove.redundant.computations`
- 源记录时间：`2026-04-20T21:18:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant Muls and Adds operations that are immediately overwritten
~~~~

### Original summary

~~~~text
Eliminating redundant Muls and Adds operations that are immediately overwritten by subsequent operations reduces vector instruction count and improves performance in GCD operators. This optimization is particularly effective when working with vectorized tile operations where redundant computations consume vector computation units without contributing to final results. The improvement comes from reducing aiv_vec_fops by 13% while decreasing scalar ratio by 14.8%.
~~~~

### Original root mechanism

~~~~text
Three redundant operations (Muls(a_div2, a, (U)1, TILE_SIZE), Muls(b_div2, b, (U)1, TILE_SIZE), Adds(ab_diff, a, (U)0, TILE_SIZE)) were dead code that consumed vector computation resources without affecting final results, as they were immediately overwritten by subsequent ShiftRight and Sub operations.
~~~~

### Original causal chain

~~~~text
Removing redundant Muls and Adds operations → reduces aiv_vec_fops by 13.0% and scalar_ratio_mean by 14.8% → decreases task_duration_us by 6.8% (703us improvement)
~~~~

## 适用条件

- GCD operator implementations using vectorized tile operations
- Code contains intermediate computations that are immediately overwritten
- Working with Ascend C vector operations like Muls, Adds, ShiftRight, Sub
- Performance profile shows high aiv_vec_fops with scalar overhead

## 失效边界

- Intermediate computations are actually used later in the computation
- The redundant operations serve a purpose like pipeline filling or dependency breaking
- Memory bandwidth is the primary bottleneck rather than computation

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

- 源 diff SHA-256：`8b8e9ffd1bee83eece5248112a111875bb142868dfae205ff65b3254bc75baa7`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -224,11 +224,6 @@
         for (int i = 0; i < REP; i++) {
             Max(a0, a, b, TILE_SIZE);
             Min(b0, a, b, TILE_SIZE);
-
-            // Redundant computations that will be overwritten - degrade performance
-            Muls(a_div2, a, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight

             And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
             And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`10999.399414`
- fast duration us：`10296.326172`
- improvement ratio：`0.06391923918183488`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.04 -> 0.03 (-25.00%)
- aiv_utilization.scalar_ratio_mean: 0.070729 -> 0.060288 (-14.76%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.07 -> 0.06 (-14.29%)
- arithmetic.aiv_vec_fops: 3.77866e+08 -> 4.27005e+08 (+13.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.14205 -> 0.1271 (-10.52%)

### Affected metric groups

- memory_ub
- aiv_utilization
- arithmetic
- memory_gm

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_bw_usage_rate_pct: 0.04 -> 0.03 (-25.00%)",
  "aiv_utilization.scalar_ratio_mean: 0.070729 -> 0.060288 (-14.76%)",
  "memory_ub.gm_to_ub_bw_usage_rate_pct: 0.07 -> 0.06 (-14.29%)",
  "arithmetic.aiv_vec_fops: 3.77866e+08 -> 4.27005e+08 (+13.00%)",
  "arithmetic.aiv_vec_misc_ratio_mean: 0.14205 -> 0.1271 (-10.52%)"
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

- Examine the BinaryGcd function in gcd/op_kernel/gcd.cpp for redundant Muls/Adds operations
- Check if intermediate results like a_div2, b_div2, ab_diff are immediately overwritten
- Remove Muls operations that compute values later overwritten by ShiftRight operations
- Remove Adds operations that compute values later overwritten by Sub operations
- Verify the removed operations don't affect final results in the tile computation loop

## 检索线索

- gcd
- redundant
- Muls
- Adds
- dead code
- vector operations
- tile
- Ascend C
- scalar overhead
- computation reduction
