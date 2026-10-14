---
type: Optimization
title: "gcd：循环计算优化"
description: "gcd 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise"]
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

- experience ID：`8dcf0e9d8f4647c4`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_002_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.inefficient.termination`
- canonical family：`loop.inefficient.termination`
- 源记录时间：`2026-04-20T19:24:47`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove early loop exit optimization in binary GCD algorithm
~~~~

### Original summary

~~~~text
Removing early loop exit conditions in binary GCD algorithm reduces pipeline stalls and improves performance by 22.3%. This optimization eliminates data-dependent loop termination that causes unpredictable control flow, allowing better vector unit utilization and reducing scalar operations. Apply this pattern to iterative algorithms where loop termination depends on runtime values.
~~~~

### Original root mechanism

~~~~text
Removed data-dependent early exit condition from binary GCD loop, forcing maximum iterations regardless of input value magnitude.
~~~~

### Original causal chain

~~~~text
Removed early exit condition -> reduced scalar operations (scalar_ratio_mean decreased 22.8%) -> reduced pipeline stalls (aiv_vec_wait_ratio_mean decreased 23.0%) -> decreased vector floating operations (aiv_vec_fops decreased 28.9%) -> improved latency (task_duration_us decreased 28.7%)
~~~~

## 适用条件

- Implementing binary GCD or similar iterative reduction algorithms
- Loop termination depends on runtime data values (e.g., while (value > 0))
- Algorithm has data-dependent early exit conditions
- Performance profiling shows high scalar_ratio_mean or pipeline stalls
- Vector unit utilization is below optimal due to control flow divergence

## 失效边界

- Loop bounds are compile-time constants
- Early exit provides significant computational savings for most inputs
- Algorithm requires data-dependent termination for correctness
- Input data distribution heavily favors early termination

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gcd.cpp

### 改动符号

- KernelGcd::BinaryGcd

### 代码签名

- GetValue
- bool_control
- scalar_value_access
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`99044a587422f500d137a87ad908098c4319d5f0139144a374ef15f2b51f415b`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -221,7 +221,7 @@
         ReduceMax(c1_fp, a0_fp, c2_fp, len, false);
         U max_val = c1.GetValue(0);

-        for (int i = 0; i < REP && (max_val >> (i / 2)) > 0; i++) {
+        for (int i = 0; i < REP; i++) {
             Max(a0, a, b, TILE_SIZE);
             Min(b0, a, b, TILE_SIZE);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`7360.947266`
- fast duration us：`5719.93457`
- improvement ratio：`0.22293498875882312`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0005 -> 0.0002 (-60.00%)
- arithmetic.aiv_vec_fops: 2.21183e+08 -> 2.85217e+08 (+28.95%)
- critical_path.aiv_vector0_time_us_mean: 5673.95 -> 7302.18 (+28.70%)
- latency.task_duration_us: 5719.94 -> 7360.95 (+28.69%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.07 -> 0.05 (-28.57%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_ub.ub_read_bw_scalar_gbps",
  "arithmetic.aiv_vec_fops",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "aiv_utilization.mte3_ratio_mean"
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

- 1. Identify loops with data-dependent termination conditions in op_kernel files
- 2. Replace while/for loops with early exit conditions with fixed iteration count loops
- 3. Use maximum possible iterations instead of runtime-dependent termination
- 4. Apply to binary GCD algorithm in gcd.cpp: change 'for (int i = 0; i < REP && (max_val >> (i / 2)) > 0; i++)' to 'for (int i = 0; i < REP; i++)'
- 5. Verify the algorithm still computes correct results with fixed iteration count
- 6. Profile to confirm reduction in scalar_ratio_mean and pipeline stalls

## 检索线索

- binary_gcd
- loop_termination
- early_exit
- pipeline_stall
- scalar_operations
- vector_utilization
- data_dependent_loop
- control_flow_divergence
- gcd_algorithm
- iterative_reduction
