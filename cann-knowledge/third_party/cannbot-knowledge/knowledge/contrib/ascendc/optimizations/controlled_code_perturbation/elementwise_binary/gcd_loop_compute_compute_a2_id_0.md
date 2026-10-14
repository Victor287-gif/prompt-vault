---
type: Optimization
title: "gcd：循环计算优化"
description: "gcd 算子的循环计算优化经验，关注混合或尚未确定的瓶颈。"
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
# gcd：循环计算优化

## 来源信息

- experience ID：`16974ecb0bdc423b`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_013_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.hoist`
- canonical family：`loop.hoist`
- 源记录时间：`2026-04-20T21:09:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Avoid premature loop-invariant hoisting of tensor operations in BinaryGcd
~~~~

### Original summary

~~~~text
In Ascend C kernels, avoid hoisting tensor operations out of loops when they compute values that are immediately overwritten by subsequent operations. The BinaryGcd operator showed 6.3% performance degradation when three tensor operations (Muls, Muls, Adds) were moved outside the loop, despite being redundant computations. This pattern occurs when operations like `Muls(a_div2, a, (U)1, TILE_SIZE)` produce values that are overwritten by `ShiftRight` operations inside the loop.
~~~~

### Original root mechanism

~~~~text
Moving redundant tensor operations (Muls, Adds) outside the BinaryGcd loop unexpectedly increased pipeline stalls and vector floating-point operations (FLOPS), showing that Ascend C's tensor operations should remain inside loops even when semantically redundant if they are immediately overwritten.
~~~~

### Original causal chain

~~~~text
Moving redundant Muls/Adds operations outside loop → Increased aiv_vec_fops by 12.8% and vector bandwidth usage → Higher pipeline stall ratios (aiv_vec_total_cflt_ratio_mean +6.3%) → Increased task_duration_us by 6.8%
~~~~

## 适用条件

- Working with Ascend C tensor operations in loops
- When operations compute values that are immediately overwritten by subsequent operations in the same iteration
- When using Muls, Adds, or similar tensor operations with constant operands
- In BinaryGcd or similar reduction algorithms with iterative updates

## 失效边界

- Avoid hoisting tensor operations out of loops when they are immediately overwritten
- Avoid moving operations like Muls(a, b, (U)1) or Adds(a, b, (U)0) outside loops if results are overwritten by ShiftRight, Sub, or other operations
- When working with tile-based operations using TILE_SIZE parameter
- When performance is sensitive to pipeline stalls and vector operation counts

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- gcd/op_kernel/gcd.cpp

### 改动符号

- BinaryGcd

### 代码签名

- GetValue
- bool_control
- scalar_value_access
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`b74c8e6d897d83e146ae5cb8414bd686d43934775a6307da407b23a4eafda63e`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -221,14 +221,14 @@
         ReduceMax(c1_fp, a0_fp, c2_fp, len, false);
         U max_val = c1.GetValue(0);

+        // Redundant computations that will be overwritten - degrade performance
+        Muls(a_div2, a, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
+        Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
+        Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight
+
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

- bottleneck type：`mixed_or_unknown`
- slow duration us：`10999.399414`
- fast duration us：`10303.646484`
- improvement ratio：`0.06325371993624003`
- machine confidence：`0.85`

### Profile evidence

- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.04 -> 0.03 (-25.00%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.07 -> 0.06 (-14.29%)
- arithmetic.aiv_vec_fops: 3.78405e+08 -> 4.27005e+08 (+12.84%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.141916 -> 0.1271 (-10.44%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.466398 -> 0.417848 (-10.41%)

### Affected metric groups

- memory_ub
- arithmetic
- memory_gm
- aiv_utilization

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "arithmetic.aiv_vec_fops",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "memory_gm.aiv_main_mem_read_bw_gbps",
  "aiv_utilization.mte2_ratio_mean",
  "memory_ub.ub_write_bw_vector_gbps",
  "aiv_utilization.mte3_ratio_mean",
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

- 1. In gcd/op_kernel/gcd.cpp, keep redundant tensor operations inside the BinaryGcd loop even if they appear semantically redundant
- 2. For operations like Muls(a_div2, a, (U)1, TILE_SIZE) that compute a*1, leave them inside loops if results are overwritten by ShiftRight operations
- 3. Monitor aiv_vec_fops metric - unexpected increases indicate inefficient operation placement
- 4. Check aiv_vec_total_cflt_ratio_mean and aiv_vec_bankgroup_cflt_ratio_mean for pipeline stall changes
- 5. Use the pattern: Keep tensor operations that compute immediately-overwritten values inside loops to maintain Ascend C pipeline efficiency

## 检索线索

- loop hoist
- tensor operations
- Muls
- Adds
- BinaryGcd
- Ascend C
- pipeline stalls
- aiv_vec_fops
- redundant computation
- performance degradation
- vector operations
- TILE_SIZE
- immediate overwrite
- ShiftRight
- tile-based
