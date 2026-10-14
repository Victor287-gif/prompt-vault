---
type: Optimization
title: "gcd：内存优化"
description: "gcd 算子的内存优化经验，关注流水线停顿。"
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
# gcd：内存优化

## 来源信息

- experience ID：`ec74958dc4063363`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_014_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tensor.view.redundancy`
- canonical family：`tensor.view.redundancy`
- 源记录时间：`2026-04-20T21:17:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor view casts and no-op arithmetic operations
~~~~

### Original summary

~~~~text
Eliminating redundant tensor reinterpret casts and corresponding no-op Add operations in the BinaryGcd loop reduces vector floating-point operations by 23.5% (from 558M to 427M) and improves task duration by 14.9%. This optimization reduces unnecessary tensor view management overhead and computational workload without affecting correctness. Apply this technique when profiling shows high aiv_vec_fops and aiv_vec_fp32_ratio_mean metrics with pipeline stall bottlenecks.
~~~~

### Original root mechanism

~~~~text
Removing redundant tensor reinterpret casts and no-op Add operations reduces unnecessary tensor view management overhead and computational workload, decreasing vector floating-point operations and improving pipeline efficiency.
~~~~

### Original causal chain

~~~~text
Removing 8 redundant ReinterpretCast operations and 8 no-op Add operations → reduces aiv_vec_fops by 131M operations (-23.5%) → decreases aiv_vec_fp32_ratio_mean from 52.7% to 41.8% → reduces aiv_vector0_time_us_mean from 12820μs to 10911μs → improves overall task_duration_us from 12923μs to 10999μs.
~~~~

## 适用条件

- Operator uses tensor ReinterpretCast operations within loops
- Profiling shows high aiv_vec_fops (>400M) and high aiv_vec_fp32_ratio_mean (>0.5)
- Pipeline stall bottlenecks are present (aiv_vec_total_cflt_ratio_mean > 0.15)
- Vector unit utilization is high (vec_ratio_mean > 0.9)
- Operator has redundant tensor view operations that don't affect computation results

## 失效边界

- Tensor reinterpret casts are necessary for type conversion between computation stages
- Memory bandwidth is the primary bottleneck (ub_read_bw_vector_gbps > 200 Gbps)
- Scalar operations dominate (scalar_ratio_mean > 0.5)
- Operator already has minimal tensor view operations

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/gcd.cpp

### 改动符号

- KernelGcd::BinaryGcd

### 代码签名

- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`8dec6ecd55b0a3163c253445ddeff6d18beb537702e3697c1e3fa42dd42f3d79`

````diff
--- Before/op_kernel/gcd.cpp
+++ After/op_kernel/gcd.cpp
@@ -207,6 +207,16 @@
         auto odd_ab_u16 = odd_ab.template ReinterpretCast<uint16_t>();
         auto mask_u16 = mask.template ReinterpretCast<uint16_t>();
         auto mask_not_u16 = mask_not.template ReinterpretCast<uint16_t>();
+
+        // Redundant reinterpret casts to degrade performance
+        auto a_div2_redundant = a_div2.template ReinterpretCast<FpT>();
+        auto b_div2_redundant = b_div2.template ReinterpretCast<FpT>();
+        auto ab_diff_redundant = ab_diff.template ReinterpretCast<FpT>();
+        auto zeros_redundant = zeros.template ReinterpretCast<FpT>();
+        auto ones_redundant = ones.template ReinterpretCast<FpT>();
+        auto a0_redundant = a0.template ReinterpretCast<FpT>();
+        auto b0_redundant = b0.template ReinterpretCast<FpT>();
+        auto c_redundant = c.template ReinterpretCast<FpT>();

         Not(a0_u16, a_u16, num_uint16 * TILE_SIZE);
         Not(b0_u16, b_u16, num_uint16 * TILE_SIZE);
@@ -230,6 +240,16 @@
             Muls(b_div2, b, (U)1, TILE_SIZE);  // Will be overwritten by ShiftRight
             Adds(ab_diff, a, (U)0, TILE_SIZE); // Will be overwritten by Sub and ShiftRight

+            // Additional redundant operations using reinterpreted tensors to degrade performance
+            Adds(a_div2_redundant, a_div2_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(b_div2_redundant, b_div2_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(ab_diff_redundant, ab_diff_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(zeros_redundant, zeros_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(ones_redundant, ones_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(a0_redundant, a0_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(b0_redundant, b0_redundant, (FpT)0.0f, TILE_SIZE);
+            Adds(c_reuduudant, c_redundant, (FpT)0.0f, TILE_SIZE);
+
    ... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`12922.578125`
- fast duration us：`10999.399414`
- improvement ratio：`0.14882314445284117`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 4.27005e+08 -> 5.583e+08 (+30.75%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.417848 -> 0.526555 (+26.02%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.1271 -> 0.100028 (-21.30%)
- critical_path.aiv_vector0_time_us_mean: 10911.3 -> 12820.5 (+17.50%)
- latency.task_duration_us: 10999.4 -> 12922.6 (+17.48%)

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
      "before": 427005312.0,
      "after": 558300480.0,
      "delta": 131295168.0,
      "rel": 0.307479003914593
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.417848,
      "after": 0.526555,
      "delta": 0.108707,
      "rel": 0.26015919664567017
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.1271,
      "after": 0.100028,
      "delta": -0.027071999999999985,
      "rel": -0.2129976396538158
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 10911.261,
      "after": 12820.497,
      "delta": 1909.235999999999,
      "rel": 0.17497849240339855
    }
  },
  {
    "latency.task_duration_us": {
      "before": 10999.399,
      "after": 12922.578,
      "delta": 1923.179,
      "rel": 0.17484400738622174
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

- 1. Examine the operator's kernel code for ReinterpretCast operations in loops, particularly in op_kernel/gcd.cpp in the BinaryGcd function
- 2. Identify redundant reinterpret casts that don't change tensor data types meaningfully
- 3. Remove unnecessary ReinterpretCast operations that only add tensor view management overhead
- 4. Remove corresponding no-op arithmetic operations (Adds with zero) thatzia use reinterpreted tensors
- 5. Verify the removal doesn't affect correctness by checking tensor type compatibility
- 6. Profile before/after to confirm reductionardy in aiv_vec_fops and aiv_vec_fp32_ratio_mean metrics

## 检索线索

- tensor
- ReinterpretCast
- redundant
- view
- ardy
- gcd
- BinaryGcd
- arithmetic
- fops
- pipeline_stall
- vector_operations
- no_op
- Adds
- zero
