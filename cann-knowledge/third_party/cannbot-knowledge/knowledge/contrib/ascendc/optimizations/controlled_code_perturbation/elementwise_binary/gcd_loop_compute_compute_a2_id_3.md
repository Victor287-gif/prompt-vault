---
type: Optimization
title: "gcd：循环计算优化"
description: "gcd 算子的循环计算优化经验，关注流水线停顿。"
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
# gcd：循环计算优化

## 来源信息

- experience ID：`8f9dde59b4722a90`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_006_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.bounds.tightening`
- canonical family：`loop.bounds.inflation`
- 源记录时间：`2026-04-20T20:08:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce loop iteration bounds in binary GCD algorithm
~~~~

### Original summary

~~~~text
Applying tighter loop bounds to the BinaryGcd template function reduces floating-point operations (FLOPS) by 33% and improves latency by 33%. This optimization works by limiting the maximum number of binary GCD algorithm iterations to match the mathematical requirements rather than using overly conservative bounds. Use this technique when template parameters control loop iteration counts and the actual required iterations are lower than the configured maximum.
~~~~

### Original root mechanism

~~~~text
Reducing the maximum iteration count (REP template parameter) in BinaryGcd function by 33% eliminates unnecessary loop iterations, decreasing arithmetic operations and improving pipeline efficiency.
~~~~

### Original causal chain

~~~~text
Reducing BinaryGcd template REP parameter from 45 to 30 for int16_t and from 93 to 62 for int32_t → aiv_vec_fops decreases from 427M to 285M (-33%) → task_duration_us decreases from 10999μs to 7361μs (-33%)
~~~~

## 适用条件

- Template functions use compile-time iteration bounds (REP parameters)
- Binary GCD or similar iterative algorithms with known maximum iteration requirements
- Loop bounds are set conservatively higher than mathematically necessary
- Performance profiling shows high aiv_vec_fops relative to expected computation

## 失效边界

- Loop bounds are data-dependent and cannot be determined at compile time
- Maximum iteration count is already minimal for worst-case inputs
- Algorithm requires variable iteration count based on input values
- Reducing bounds would cause incorrect results for some inputs

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gcd.cpp

### 改动符号

- KernelGcd::Compute

### 代码签名

- AllocTensor
- DeQue
- GetValue
- SetValue
- compile_time_specialization
- constexpr
- queue_api
- scalar_value_access
- template

### 原始 diff

- 源 diff SHA-256：`f370a6b29af00b4841bc5cdc5ed5bf71951645d7f4591e06cf43ef7b9c4bbf71`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -293,9 +293,9 @@
         auto x2 = inX2.DeQue<T>();
         auto y = outY.AllocTensor<T>();
         if constexpr(std::is_same<T, int16_t>::value) {
-            BinaryGcd<int16_t, 30>(y, x1, x2, len);
+            BinaryGcd<int16_t, 45>(y, x1, x2, len);  // Increased from 30 to 45 (50% more iterations)
         } else if constexpr(std::is_same<T, int32_t>::value) {
-            BinaryGcd<int32_t, 62>(y, x1, x2, len);
+            BinaryGcd<int32_t, 93>(y, x1, x2, len);   // Increased from 62 to 93 (50% more iterations)
         } else {
             for (int i = 0;i < len;i++) {
                 y.SetValue(i, gcd_1(x1.GetValue(i), x2.GetValue(i)));
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`10999.399414`
- fast duration us：`7360.947266`
- improvement ratio：`0.33078643760940163`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0002 -> 0.0001 (-50.00%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0002 -> 0.0001 (-50.00%)
- arithmetic.aiv_vec_fops: 2.85217e+08 -> 4.27005e+08 (+49.71%)
- latency.task_duration_us: 7360.95 -> 10999.4 (+49.43%)
- critical_path.aiv_vector0_time_us_mean: 7302.18 -> 10911.3 (+49.42%)

### Affected metric groups

- memory_ub
- arithmetic
- latency
- critical_path

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0002,
      "after": 0.0001,
      "delta": -0.0001,
      "rel": -0.5
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0002,
      "after": 0.0001,
      "delta": -0.0001,
      "rel": -0.5
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 285216768.0,
      "after": 427005312.0,
      "delta": 141788544.0,
      "rel": 0.49712555469389513
    }
  },
  {
    "latency.task_duration_us": {
      "before": 7360.947,
      "after": 10999.399,
      "delta": 3638.4519999999993,
      "rel": 0.49429129159604046
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 7302.175,
      "after": 10911.261,
      "delta": 3609.0860000000002,
      "rel": 0.4942480836189218
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

- 1. Examine template parameters in BinaryGcd or similar iterative functions in op_kernel/gcd.cpp
- 2. Calculate the theoretical maximum iterations needed for the algorithm (for binary GCD: log2(max_input_value) * 2)
- 3. Compare current REP parameter value with calculated maximum
- 4. Reduce REP parameter to match calculated maximum while preserving correctness
- 5. Verify performance improvement by measuring aiv_vec_fops and task_duration_us metrics

## 检索线索

- loop_bounds
- template_parameters
- REP
- BinaryGcd
- gcd
- iteration_count
- compile_time
- aiv_vec_fops
- arithmetic_operations
- pipeline_stall
