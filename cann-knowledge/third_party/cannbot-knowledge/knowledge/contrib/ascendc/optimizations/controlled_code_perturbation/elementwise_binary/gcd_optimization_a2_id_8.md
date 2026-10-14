---
type: Optimization
title: "gcd：冗余消除优化"
description: "gcd 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
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

- experience ID：`ca0edefccfbee38c`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_012_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.computation`
- canonical family：`remove.redundant.computation`
- 源记录时间：`2026-04-20T21:00:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove redundant Muls and Adds operations that are immediately overwritten
~~~~

### Original summary

~~~~text
In Ascend C GCD operator, removing redundant Muls(a_div2, a, (U)1, TILE_SIZE), Muls(b_div2, b, (U)1, TILE_SIZE), and Adds(ab_diff, a, (U)0, TILE_SIZE) operations that compute values immediately overwritten by subsequent ShiftRight and Sub operations reduces scalar overhead by 14.78% and total vector FLOPS by 13.00%. This optimization applies when intermediate tile results are computed but never used before being overwritten.
~~~~

### Original root mechanism

~~~~text
Redundant Muls and Adds operations that compute values immediately overwritten by subsequent ShiftRight and Sub operations create unnecessary scalar overhead and increase vector floating-point operations.
~~~~

### Original causal chain

~~~~text
Removing redundant Muls and Adds operations → scalar_ratio_mean decreases 14.78% → aiv_vec_fops decreases 13.00% → task_duration_us decreases 6.40%
~~~~

## 适用条件

- When Muls or Adds operations compute intermediate tile results that are immediately overwritten by subsequent operations
- When scalar operations (Muls, Adds) precede vector operations (ShiftRight, Sub) on the same tile buffers
- When aiv_utilization.scalar_ratio_mean is above 0.06 and arithmetic.aiv_vec_fops shows room for reduction

## 失效边界

- When the intermediate computations are actually used later in the kernel
- When removing operations would break data dependencies or change algorithm correctness
- When scalar operations serve as pipeline barriers or synchronization points

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- gcd/op_kernel/gcd.cpp

### 改动符号

- KernelGcd::BinaryGcd

### 代码签名

- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`53bcfcd4f8c59c7b63286c81e066e00f21f232451d98afea478e15e0d469d8b1`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -225,10 +225,7 @@
             Max(a0, a, b, TILE_SIZE);
             Min(b0, a, b, TILE_SIZE);

-            // Redundant computations that will be overwritten - degrade performance
-            Muls(a_div2, a, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight
+

             And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
             And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`10999.399414`
- fast duration us：`10295.946289`
- improvement ratio：`0.06395377588567673`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.04 -> 0.03 (-25.00%)
- aiv_utilization.scalar_ratio_mean: 0.07074 -> 0.060288 (-14.78%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.07 -> 0.06 (-14.29%)
- arithmetic.aiv_vec_fops: 3.77866e+08 -> 4.27005e+08 (+13.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.142058 -> 0.1271 (-10.53%)

### Affected metric groups

- memory_ub
- aiv_utilization
- arithmetic
- memory_gm

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "aiv_utilization.scalar_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_misc_ratio_mean"
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

- 1. Examine KernelGcd::BinaryGcd in gcd/op_kernel/gcd.cpp for Muls/Adds operations on tile buffers
- 2. Check if results are immediately overwritten by subsequent ShiftRight or Sub operations on same buffers
- 3. Remove redundant Muls(a_div2, a, (U)1, TILE_SIZE), Muls(b_div2, b, (U)1, TILE_SIZE), Adds(ab_diff, a, (U)0, TILE_SIZE) operations
- 4. Verify scalar_ratio_mean decreases and aiv_vec_fops reduces in profile metrics

## 检索线索

- redundant computation
- scalar overhead
- Muls
- Adds
- tile operations
- immediate overwrite
- vector FLOPS
- Ascend C
- GCD operator
- binary gcd
