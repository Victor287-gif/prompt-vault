---
type: Optimization
title: "mul_sigmoid：冗余消除优化"
description: "mul_sigmoid 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
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
# mul_sigmoid：冗余消除优化

## 来源信息

- experience ID：`b7e54e0bf8922176`
- 源 schema version：`v2`
- 算子：`mul_sigmoid`
- source iteration：`round_015_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T00:21:04`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant identity multiplication operations in Ascend C kernels
~~~~

### Original summary

~~~~text
In Ascend C kernels, redundant identity multiplication operations that multiply values by 1.0 introduce unnecessary computational overhead without changing mathematical results. Removing these operations reduces vector FLOPS by 29.6% (from 138.6M to 97.5M) and improves vector unit utilization by 13.9% (from 0.757 to 0.664), resulting in 11% performance improvement. This optimization is applicable when operators contain unnecessary arithmetic operations that don't affect final results.
~~~~

### Original root mechanism

~~~~text
Removing redundant identity multiplication operations (multiply by 1.0) reduces vector floating-point operations (FLOPS) by 29.6% and improves vector unit utilization, allowing the pipeline to execute more efficiently.
~~~~

### Original causal chain

~~~~text
Removing 6 redundant AscendC::Muls<half> operations with constant 1.0 multiplier → vector FLOPS reduced by 29.6% (138.6M to 97.5M) → vector unit utilization improved by 13.9% → vector wait ratio decreased by 27.8% → task duration reduced by 11.0% (1273μs to 1133μs)
~~~~

## 适用条件

- Ascend C kernels contain identity multiplication operations (multiply by 1.0)
- Operator has high vector FLOPS count relative to computation complexity
- Profile shows high vector unit utilization (>0.75) with significant wait ratios
- Operator uses AscendC::Muls with constant 1.0 multiplier
- Code contains redundant arithmetic operations that don't affect final results

## 失效边界

- Multiplication by 1.0 serves a purpose like type conversion or pipeline synchronization
- Operator is already memory-bound rather than compute-bound
- Vector unit utilization is already low (<0.5)
- The multiplication operations have side effects like preventing compiler optimizations

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- mul_sigmoid/op_kernel/mul_sigmoid.cpp

### 改动符号

- Ascend::MulSigmoid::Compute

### 代码签名

- bool_control
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`021b72d345056533ced183db81b1df40f7d3bcca0d59d467108a60332b3fab5a`

````diff
--- Before/op_kernel/mul_sigmoid.cpp
+++ After/op_kernel/mul_sigmoid.cpp
@@ -71,29 +71,7 @@
                  /* 计算过程 */
                  AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);

-                 // REDUNDANT: Add multiple unnecessary arithmetic operations that don't affect result
-                 // These operations create artificial computational overhead
-
-                 // First redundant multiply by 1.0 (identity operation)
-                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
-
-                 // Second redundant multiply by 1.0
-                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
-
-                 // Third redundant multiply by 1.0
-                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
-
-                 // Fourth redundant multiply by 1.0
-                 AscendC::Muls<half>(x1_local, x1_local, static_cast<half>(1.0f), this->tile_length);
-
                  AscendC::Sigmoid<half, false>(out_local, x1_local, this->tile_length);
-
-                 // REDUNDANT: Add unnecessary operations on out_local that don't affect final result
-                 // Multiply out_local by 1.0 (identity operation)
-                 AscendC::Muls<half>(out_local, out_local, static_cast<half>(1.0f), this->tile_length);
-
-                 // Another redundant multiply by 1.0
-                 AscendC::Muls<half>(out_local, out_local, static_cast<half>(1.0f), this->tile_length);

                  AscendC::LocalTensor<uint8_t> less_local = this->less_local_buf.Get<uint8_t>();
                  AscendC::CompareScalar(less_local, out_local, this->t2, AscendC::CMPMODE::LT, this->tile_length);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1273.345459`
- fast duration us：`1133.302612`
- improvement ratio：`0.10998024613837332`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005347 -> 0.012453 (+132.90%)
- arithmetic.aiv_vec_fops: 9.75448e+07 -> 1.38564e+08 (+42.05%)
- aiv_utilization.mte2_ratio_mean: 0.319866 -> 0.225245 (-29.58%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.353301 -> 0.255011 (-27.82%)
- memory_ub.ub_write_bw_vector_gbps: 186.41 -> 226.34 (+21.42%)

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
      "before": 0.319866,
      "after": 0.225245,
      "delta": -0.09462099999999998,
      "rel": -0.29581449732075304
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.353301,
      "after": 0.255011,
      "delta": -0.09828999999999999,
      "rel": -0.2782047036379744
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 186.41,
      "after": 226.34,
      "delta": 39.93000000000001,
      "rel": 0.21420524649965134
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

- Search for AscendC::Muls calls with constant 1.0 multiplier in kernel files
- Remove identity multiplication operations that don't affect mathematical results
- Focus on operators with high vector FLOPS count in arithmetic.aiv_vec_fops metric
- Verify optimization by checking reduction in vector FLOPS and improved vector unit utilization
- Monitor pipeline_stalls.aiv_vec_wait_ratio_mean for improvements after removal

## 检索线索

- redundant operations
- identity multiplication
- AscendC::Muls
- vector FLOPS
- vector unit utilization
- scalar overhead
- mul_sigmoid
- arithmetic optimization
- compute-bound
- pipeline stalls
