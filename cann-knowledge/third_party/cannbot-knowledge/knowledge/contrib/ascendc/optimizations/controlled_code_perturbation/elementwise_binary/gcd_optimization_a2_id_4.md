---
type: Optimization
title: "gcd：冗余消除优化"
description: "gcd 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
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

- experience ID：`30d8e63ea2e2a0e6`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_004_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`dead.code.removal`
- canonical family：`dead.code.removal`
- 源记录时间：`2026-04-20T19:42:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tile operations in binary GCD loop
~~~~

### Original summary

~~~~text
In Ascend C vector kernels, redundant tile operations that compute intermediate values immediately overwritten by subsequent operations create unnecessary vector floating-point operations and scalar overhead. Removing these redundant operations reduces aiv_vec_fops by 12.98% (from 285M to 252M) and scalar_ratio_mean by 14.53%, improving overall kernel performance by 6.4%. Apply this optimization when tile operations compute values that are never used or are immediately overwritten.
~~~~

### Original root mechanism

~~~~text
Removing redundant Muls and Adds operations that compute values immediately overwritten by subsequent ShiftRight and Sub operations reduces vector floating-point operations and scalar overhead
~~~~

### Original causal chain

~~~~text
Removed three redundant tensor operations (Muls, Muls, Adds) → Reduced vector floating-point operations by 32.8M (12.98%) and scalar ratio by 14.53% → Decreased critical path aiv_vector0_time_us_mean from 7302μs to 6838μs → Reduced task_duration_us from 7361μs to 6893μs (6.4% improvement)
~~~~

## 适用条件

- When using tile operations (Muls, Adds, etc.) in Ascend C vector kernels
- When intermediate tile values are computed but immediately overwritten by subsequent operations
- When working with binary algorithms like GCD that use bitwise operations and shifts
- When scalar overhead (scalar_ratio_mean > 0.06) indicates potential for instruction reduction

## 失效边界

- When tile operations have side effects or influence control flow
- When intermediate values are used in subsequent computations before being overwritten
- When compiler optimizations already eliminate dead code automatically
- When removing operations would break semantic equivalence

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

- 源 diff SHA-256：`859b5f80ce1009bd610f8f711f7651d44f26fbb6a20a3dc5b63ada356cf78a51`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -225,11 +225,6 @@
             Max(a0, a, b, TILE_SIZE);
             Min(b0, a, b, TILE_SIZE);

-            // Redundant computations that will be overwritten - degrade performance
-            Muls(a_div2, a, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
-            Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight
-
             And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
             And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`7360.947266`
- fast duration us：`6893.078125`
- improvement ratio：`0.0635609961724728`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.071717 -> 0.061295 (-14.53%)
- arithmetic.aiv_vec_fops: 2.52457e+08 -> 2.85217e+08 (+12.98%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.465727 -> 0.416837 (-10.50%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.141963 -> 0.127061 (-10.50%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.1 -> 0.09 (-10.00%)

### Affected metric groups

- aiv_utilization
- arithmetic
- memory_gm
- memory_ub

### Changed metrics top

~~~~json
[
  "aiv_utilization.scalar_ratio_mean",
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "memory_ub.ub_write_bw_vector_gbps",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us"
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

- 1. Examine the BinaryGcd function in gcd/op_kernel/gcd.cpp for redundant tile operations
- 2. Identify Muls, Adds, or other tile operations whose results are immediately overwritten
- 3. Verify semantic equivalence by checking that removed operations don't affect final outputs
- 4. Remove redundant operations like 'Muls(a_div2, a, (U)1, TILE_SIZE)' when followed by 'ShiftRight'
- 5. Profile before/after to confirm reduction in aiv_vec_fops and scalar_ratio_mean metrics

## 检索线索

- dead.code.removal
- redundant.tile.operations
- binary.gcd
- ascend.c
- vector.kernel
- tile.overhead
- scalar.ratio
- aiv_vec_fops
- Muls
- Adds
- ShiftRight
