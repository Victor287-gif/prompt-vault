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

- experience ID：`2a0f293575ae5c88`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_009_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.computation`
- canonical family：`remove.redundant.computation`
- 源记录时间：`2026-04-20T20:28:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant arithmetic operations in Ascend C kernels
~~~~

### Original summary

~~~~text
In Ascend C kernels, removing redundant arithmetic operations that compute values immediately overwritten by subsequent operations can improve performance by reducing scalar overhead. This optimization is particularly effective in gcd.cpp's BinaryGcd function where temporary calculations for a_div2, b_div2, and ab_diff were computed but never used before being overwritten. The reduction in scalar operations decreases aiv_utilization.scalar_ratio_mean by 14.76% and improves overall latency by 6.4%.
~~~~

### Original root mechanism

~~~~text
Removing Muls and Adds operations that compute values immediately overwritten by subsequent ShiftRight and Sub operations reduces scalar overhead and improves vector unit utilization.
~~~~

### Original causal chain

~~~~text
Removing redundant Muls and Adds operations → Reduced scalar operation ratio from 0.07073 to 0.060288 (-14.76%) → Decreased vector floating-point operations from 427M to 378M (-13.0%) → Reduced task duration from 10,999μs to 10,296μs (-6.4%)
~~~~

## 适用条件

- When Ascend C kernels contain temporary variable calculations that are immediately overwritten
- When scalar operation ratio (aiv_utilization.scalar_ratio_mean) is above 0.06
- When vector floating-point operations (aiv_vec_fops) show unnecessary computation
- In gcd.cpp BinaryGcd function pattern with Muls/Adds followed by ShiftRight/Sub operations

## 失效边界

- When temporary calculations are actually used by subsequent operations
- When removing operations would break data dependencies
- When scalar operations are essential for correctness

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

- 源 diff SHA-256：`5da18ff0357837250130dc82c7ec66b13c4996e5e6df84970ddcf0e362483546`

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
+            // Removed redundant computations that were immediately overwritten

             And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
             And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`10999.399414`
- fast duration us：`10296.446289`
- improvement ratio：`0.06390831885832635`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.04 -> 0.03 (-25.00%)
- aiv_utilization.scalar_ratio_mean: 0.07073 -> 0.060288 (-14.76%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.07 -> 0.06 (-14.29%)
- arithmetic.aiv_vec_fops: 3.77866e+08 -> 4.27005e+08 (+13.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.14206 -> 0.1271 (-10.53%)

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

- Inspect gcd.cpp:225-235 for redundant Muls(a_div2, a, (U)1, TILE_SIZE) and Adds(ab_diff, a, (U)0, TILE_SIZE) operations
- Check if computed values are read before being overwritten by ShiftRight or Sub operations
- Remove operations that compute values immediately overwritten without being read
- Verify functional correctness after removal by testing with edge cases
- Profile to confirm reduction in aiv_utilization.scalar_ratio_mean and aiv_vec_fops

## 检索线索

- redundant computation
- scalar overhead
- vector operations
- Ascend C
- gcd
- BinaryGcd
- Muls
- Adds
- ShiftRight
- temporary variables
- arithmetic optimization
