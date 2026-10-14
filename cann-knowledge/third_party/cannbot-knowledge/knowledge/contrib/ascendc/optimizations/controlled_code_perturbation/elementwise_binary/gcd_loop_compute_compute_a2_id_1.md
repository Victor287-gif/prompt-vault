---
type: Optimization
title: "gcd：循环计算优化"
description: "gcd 算子的循环计算优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`2f96af5becbc40ef`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`loop.redundant.computation`
- canonical family：`loop.redundant.computation`
- 源记录时间：`2026-04-20T19:03:53`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant vector operations in GCD computation loop
~~~~

### Original summary

~~~~text
In Ascend C GCD vector operations, redundant computations that produce values immediately overwritten by subsequent operations add unnecessary computational overhead. Removing three redundant vector operations (two Muls and one Add) from the main BinaryGcd loop reduces vector floating point operations by 12.95% and improves overall performance by 6.3%. This optimization applies to vector operations where intermediate results are computed but never used due to immediate overwriting.
~~~~

### Original root mechanism

~~~~text
Removing redundant vector operations (Muls, Muls, Adds) that compute values immediately overwritten by subsequent operations reduces computational workload without affecting correctness.
~~~~

### Original causal chain

~~~~text
Removing redundant Muls and Adds operations → reduces aiv_vec_fops by 12.95% → decreases vector computation time → reduces task_duration_us by 6.3%
~~~~

## 适用条件

- Vector operations contain redundant Muls or Adds that compute values immediately overwritten
- GCD or similar binary reduction algorithms using vector operations with TILE_SIZE loops
- Operations in KernelGcd::BinaryGcd function pattern with contiguous_access and UB memory usage
- When aiv_vec_fops metric shows high computational workload relative to actual needed operations

## 失效边界

- When the redundant operations serve as pipeline fillers to hide memory latency
- When operations are needed for correctness despite being overwritten later
- When the overhead of removing operations exceeds the benefit due to increased register pressure
- In scalar-dominated kernels where vector operation reduction has minimal impact

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gcd.cpp

### 改动符号

- KernelGcd::BinaryGcd

### 代码签名

- contiguous_access
- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`28af33eb952520a975bba363886f7672d36359ef7ba22a5915d0f705b94171a2`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -225,6 +225,11 @@
             Max(a0, a, b, TILE_SIZE);
             Min(b0, a, b, TILE_SIZE);

+            // Redundant computations that will be overwritten - degrade performance
+            Muls(a_div2, a, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
+            Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
+            Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight
+
             And(a_u16, a0_u16, ones_u16, num_uint16 * TILE_SIZE);
             And(b_u16, b0_u16, ones_u16, num_uint16 * TILE_SIZE);
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2863.337158`
- fast duration us：`2682.993652`
- improvement ratio：`0.06298367815195298`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 9.79115e+07 -> 1.10593e+08 (+12.95%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.141734 -> 0.126928 (-10.45%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.464056 -> 0.415581 (-10.45%)
- aiv_utilization.mte2_ratio_mean: 0.006668 -> 0.006073 (-8.92%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.12 -> 0.11 (-8.33%)

### Affected metric groups

- arithmetic
- aiv_utilization
- memory_gm
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_fops": {
      "before": 97911520.0,
      "after": 110592736.0,
      "delta": 12681216.0,
      "rel": 0.12951709870299224
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.141734,
      "after": 0.126928,
      "delta": -0.014805999999999986,
      "rel": -0.10446329038903852
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.464056,
      "after": 0.415581,
      "delta": -0.048475000000000046,
      "rel": -0.10445937559260099
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.006668,
      "after": 0.006073,
      "delta": -0.000595,
      "rel": -0.08923215356928614
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.12,
      "after": 0.11,
      "delta": -0.009999999999999995,
      "rel": -0.0833333333333333
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.344345,
  "overall_deviation": 0.137738,
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

- 1. Review vector operations in loops (like KernelGcd::BinaryGcd in op_kernel/gcd.cpp) for redundant computations
- 2. Identify Muls/Adds operations where results are immediately overwritten by subsequent ShiftRight or other operations
- 3. Remove redundant Muls(a_div2, a, (U)1, TILE_SIZE) and Adds(ab_diff, a, (U)0, TILE_SIZE) operations
- 4. Verify that removed operations don't affect correctness by checking data flow dependencies
- 5. Monitor arithmetic.aiv_vec_fops metric for reduction and latency.task_duration_us for improvement
- 6. Check that vector utilization (aiv_utilization.vec_ratio_mean) remains stable or improves

## 检索线索

- gcd
- vector
- redundant
- loop
- computation
- muls
- adds
- binarygcd
- tile
- ub
- arithmetic
- fops
- performance
- optimization
