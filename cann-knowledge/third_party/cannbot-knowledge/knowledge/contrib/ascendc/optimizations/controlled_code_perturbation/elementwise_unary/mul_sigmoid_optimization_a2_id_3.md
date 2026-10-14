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

- experience ID：`d1bf430d4c4d971d`
- 源 schema version：`v2`
- 算子：`mul_sigmoid`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T00:57:45`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating redundant identity multiplication operations in AscendC kernels
~~~~

### Original summary

~~~~text
In AscendC vector kernels, redundant identity multiplication operations (multiplying by 1.0) create unnecessary computational overhead even though they don't change results. These operations increase vector floating-point operations (FLOPS) by 42% and cause higher pipeline stall ratios, particularly in vector wait cycles. Removing these operations reduces aiv_vec_fops from 138.6M to 97.5M and improves vector unit utilization from 75.7% to 66.4% while decreasing pipeline stalls by 28.5%.
~~~~

### Original root mechanism

~~~~text
Removing redundant AscendC::Muls operations that multiply values by 1.0 (identity operations) reduces vector unit pipeline stalls and improves compute efficiency by eliminating unnecessary instruction issue overhead.
~~~~

### Original causal chain

~~~~text
Removing six redundant AscendC::Muls operations that multiply by 1.0 → reduces aiv_vec_fops by 42.1% (138.6M to 97.5M) → decreases aiv_vec_wait_ratio_mean by 28.5% (0.255 to 0.357) → reduces task_duration_us by 10.9% (1273μs to 1134μs)
~~~~

## 适用条件

- AscendC kernels contain identity operations like AscendC::Muls with scalar 1.0
- Vector kernels show high aiv_vec_fops with low vector unit utilization (<80%)
- Pipeline stall metrics show high aiv_vec_wait_ratio_mean (>0.25)
- Code contains commented 'REDUNDANT' operations marked as artificial computational overhead

## 失效边界

- Operations are necessary for numerical stability or algorithmic correctness
- Identity operations serve as placeholders for future functionality
- Kernel is already compute-bound with vector utilization >90%
- Memory bandwidth is the primary bottleneck (GM bandwidth >80% utilization)

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
- fast duration us：`1134.222656`
- improvement ratio：`0.10925770537498732`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005356 -> 0.012453 (+132.51%)
- arithmetic.aiv_vec_fops: 9.75448e+07 -> 1.38564e+08 (+42.05%)
- aiv_utilization.mte2_ratio_mean: 0.320022 -> 0.225245 (-29.62%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.356618 -> 0.255011 (-28.49%)
- memory_ub.ub_write_bw_vector_gbps: 186.72 -> 226.34 (+21.22%)

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
      "before": 0.005356,
      "after": 0.012453,
      "delta": 0.007097000000000001,
      "rel": 1.325056011949216
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
      "before": 0.320022,
      "after": 0.225245,
      "delta": -0.09477699999999997,
      "rel": -0.29615776415371436
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.356618,
      "after": 0.255011,
      "delta": -0.101607,
      "rel": -0.28491831595713063
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 186.72,
      "after": 226.34,
      "delta": 39.620000000000005,
      "rel": 0.21218937446443875
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

- Search for AscendC::Muls operations multiplying by 1.0 in mul_sigmoid/op_kernel/mul_sigmoid.cpp
- Remove all identity multiplication operations that don't affect results
- Verify vector unit utilization improves from pipeline_stalls.aiv_vec_wait_ratio_mean reduction
- Confirm aiv_vec_fops decreases by 30-50% without affecting functional correctness
- Check that aiv_utilization.vec_ratio_mean increases while mte2_ratio_mean decreases

## 检索线索

- AscendC
- Muls
- identity_operation
- redundant_arithmetic
- vector_pipeline_stalls
- mul_sigmoid
- scalar_overhead
- computational_overhead
