---
type: Optimization
title: "gcd：冗余消除优化"
description: "gcd 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
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

- experience ID：`269898dbe668dbac`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_001_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.computation`
- canonical family：`remove.redundant.computation`
- 源记录时间：`2026-04-20T19:11:00`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant Muls/Adds operations that are immediately overwritten
~~~~

### Original summary

~~~~text
In Ascend C operators, avoid redundant arithmetic operations that compute values which are immediately overwritten by subsequent operations. The gcd operator had Muls(a_div2, a, (U)1) and Muls(b_div2, b, (U)1) operations whose results were immediately overwritten by ShiftRight operations, and Adds(ab_diff, a, (U)0) that was overwritten by Sub and ShiftRight. Removing these redundant operations reduces vector floating-point operations (FLOPS) by 12.95% and improves performance by 6.3%.
~~~~

### Original root mechanism

~~~~text
Redundant Muls and Adds operations that compute values immediately overwritten by subsequent ShiftRight and Sub operations create unnecessary pipeline pressure and increase arithmetic operations without functional benefit
~~~~

### Original causal chain

~~~~text
Removing redundant Muls/Adds operations -> reduces aiv_vec_fops by 12.95% (from 110.6M to 97.9M) and decreases aiv_vec_fp32_ratio_mean by 10.47% -> reduces critical path aiv_vector0_time_us_mean by 6.77% (from 2839us to 2659us) -> improves overall task_duration_us by 6.7% (from 2863us to 2682us)
~~~~

## 适用条件

- When Ascend C operators contain Muls or Adds operations whose results are immediately overwritten by subsequent operations like ShiftRight or Sub
- When profiling shows high aiv_vec_fops with low arithmetic efficiency
- When vector arithmetic operations show high fp32 ratio without corresponding functional benefit
- When operators use tile-based computations with temporary variables that get reassigned

## 失效边界

- When operations have side effects beyond value computation
- When operations are required for data dependency or synchronization
- When removing operations would break functional correctness
- When operations serve as explicit barriers or memory ordering constraints

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- gcd/op_kernel/gcd.cpp

### 改动符号

- BinaryGcd

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`485afa6d590f25fcdce3f02e9cfbab7e6115093aa4e153abe43020d57c5e0263`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -224,11 +224,6 @@
         for (int i = 0; i < REP && (max_val >> (i / 2)) > 0; i++) {
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

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2863.337158`
- fast duration us：`2682.293701`
- improvement ratio：`0.06322813102682467`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 9.79115e+07 -> 1.10593e+08 (+12.95%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.141773 -> 0.126928 (-10.47%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.464184 -> 0.415581 (-10.47%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.12 -> 0.11 (-8.33%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.13 -> 0.12 (-7.69%)

### Affected metric groups

- arithmetic
- memory_gm
- memory_ub
- critical_path

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "memory_ub.ub_write_bw_vector_gbps",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "aiv_utilization.mte2_ratio_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean"
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

- Examine Ascend C operator kernels for sequences where temporary variables are computed then immediately reassigned
- Look for patterns like: Muls(tmp, a, constant) followed by ShiftRight(tmp, a, shift) where tmp is overwritten
- Check for Adds operations that compute values later replaced by Sub or other arithmetic operations
- In gcd.cpp:224-229, remove Muls(a_div2, a, (U)1) and Muls(b_div2, b, (U)1) when followed by ShiftRight operations
- Remove Adds(ab_diff, a, (U)0) when ab_diff is later computed via Sub(a, b) or similar operations
- Verify removal doesn't affect functional correctness by checking all variable uses after the redundant operations

## 检索线索

- redundant computation
- Muls
- Adds
- arithmetic optimization
- Ascend C
- gcd
- vector operations
- pipeline pressure
- FLOPS reduction
- tile-based computation
