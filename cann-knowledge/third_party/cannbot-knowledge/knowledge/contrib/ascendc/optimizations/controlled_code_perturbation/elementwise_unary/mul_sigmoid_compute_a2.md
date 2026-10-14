---
type: Optimization
title: "mul_sigmoid：计算优化"
description: "mul_sigmoid 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/docs/MulSigmoid.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/framework/tf_plugin/tensorflow_mul_sigmoid_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/op_host/mul_sigmoid.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/op_host/mul_sigmoid_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/op_kernel/mul_sigmoid.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/mul_sigmoid/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# mul_sigmoid：计算优化

## 来源信息

- experience ID：`2b83e41cf974504b`
- 源 schema version：`v2`
- 算子：`mul_sigmoid`
- source iteration：`round_014_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`compute.redundant.identity`
- canonical family：`compute.redundant.identity`
- 源记录时间：`2026-04-21T00:15:11`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove redundant identity multiplication operations from compute kernel
~~~~

### Original summary

~~~~text
Eliminating redundant identity multiplication operations (Muls with scalar 1.0) from Ascend C compute kernels reduces vector floating-point operations by 42% and improves vector unit utilization by 14%. This optimization is applicable to operators where identity operations were added for testing or debugging purposes, particularly in math category kernels using vector operations with half-precision data types.
~~~~

### Original root mechanism

~~~~text
Removing redundant Muls operations that multiply by 1.0 (identity) reduces unnecessary computational overhead while maintaining mathematical correctness
~~~~

### Original causal chain

~~~~text
Removing four redundant Muls operations → aiv_vec_fops reduced by 41,019,456 operations (42.05%) → vector utilization (vec_ratio_mean) increased from 0.662853 to 0.756678 (14.15%) → vector wait ratio reduced from 0.354496 to 0.255011 (28.06%) → task duration reduced from 1273.345μs to 1132.443μs (11.07% improvement)
~~~~

## 适用条件

- Operator uses AscendC::Muls with scalar 1.0 (identity operation)
- Vector compute kernels show high aiv_vec_fops but low vector utilization (vec_ratio_mean < 0.7)
- Kernel contains redundant arithmetic operations that don't affect mathematical results
- Operator is in the math category and uses half-precision (half) data types
- Profile shows high vector wait ratio (aiv_vec_wait_ratio_mean > 0.3)

## 失效边界

- Identity operations serve a purpose (e.g., memory alignment, pipeline balancing)
- Operator already has optimal vector utilization (vec_ratio_mean > 0.8)
- Kernel is memory-bound rather than compute-bound
- Redundant operations are required for compiler optimization or instruction scheduling

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- mul_sigmoid/op_kernel/mul_sigmoid.cpp

### 改动符号

- MulSigmoid::Compute

### 代码签名

- AllocTensor
- DeQue
- bool_control
- queue_api
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`4b948fb2a1b8169d013ca7dad04a6f30aaf73e6516957dd72770aff6135dff0b`

````diff
--- Before/op_kernel/mul_sigmoid.cpp
+++ After/op_kernel/mul_sigmoid.cpp
@@ -68,15 +68,37 @@
                 AscendC::LocalTensor<half>  x1_local = this->x1_queue.DeQue<half>();
                 AscendC::LocalTensor<half> out_local = this->out_queue.AllocTensor<half>();

-                /* 计算过程 */
-                AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);
-
-                AscendC::Sigmoid<half, false>(out_local, x1_local, this->tile_length);
-
-                AscendC::LocalTensor<uint8_t> less_local = this->less_local_buf.Get<uint8_t>();
-                AscendC::CompareScalar(less_local, out_local, this->t2, AscendC::CMPMODE::LT, this->tile_length);
-
-                AscendC::Muls(x1_local, out_local, half_two, this->tile_length);
+                 /* 计算过程 */
+                 AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);
+
+                 // REDUNDANT: Add multiple unnecessary arithmetic operations that don't affect result
+                 // These operations create artificial computational overhead
+
+                 // First redundant multiply by 1.0 (identity operation)
+                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
+
+                 // Second redundant multiply by 1.0
+                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
+
+                 // Third redundant multiply by 1.0
+                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
+
+                 // Fourth redundant multiply by 1.0
+                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
+
+                 AscendC::Sigmoid<half, false>(out_local, x1_local, this->tile_length);
+
+                 // R... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1273.345459`
- fast duration us：`1132.442627`
- improvement ratio：`0.11065562059698679`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005347 -> 0.012453 (+132.90%)
- arithmetic.aiv_vec_fops: 9.75448e+07 -> 1.38564e+08 (+42.05%)
- aiv_utilization.mte2_ratio_mean: 0.32174 -> 0.225245 (-29.99%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.354496 -> 0.255011 (-28.06%)
- memory_ub.ub_write_bw_vector_gbps: 186.42 -> 226.34 (+21.41%)

### Affected metric groups

- pipeline_stalls
- arithmetic
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.005347,
      "after": 0.012453,
      "delta": 0.007106,
      "rel": 1.3289695156162333
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 97544768.0,
      "after": 138564224.0,
      "delta": 41019456.0,
      "rel": 0.42051928402761696
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.32174,
      "after": 0.225245,
      "delta": -0.09649500000000003,
      "rel": -0.29991608130788844
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.354496,
      "after": 0.255011,
      "delta": -0.09948499999999999,
      "rel": -0.28063786333273155
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 186.42,
      "after": 226.34,
      "delta": 39.920000000000016,
      "rel": 0.21414011372170377
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

- Examine Compute method in op_kernel/mul_sigmoid.cpp for redundant AscendC::Muls operations
- Identify Muls calls with scalar value 1.0 that don't change tensor values
- Remove unnecessary identity operations while preserving mathematical correctness
- Verify vector utilization improves (vec_ratio_mean should increase)
- Check that aiv_vec_fops decreases while maintaining same functional behavior
- Monitor pipeline stalls (aiv_vec_wait_ratio_mean) for reduction

## 检索线索

- mul_sigmoid
- identity_operation
- redundant_compute
- vector_utilization
- muls_1.0
- ascendc_muls
- floating_point_operations
- compute_overhead
- math_operator
- half_precision
