---
type: Optimization
title: "mul_sigmoid：冗余消除优化"
description: "mul_sigmoid 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
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

- experience ID：`2f643e29788334a2`
- 源 schema version：`v2`
- 算子：`mul_sigmoid`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.ops`
- canonical family：`remove.redundant.ops`
- 源记录时间：`2026-04-21T01:32:03`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove identity multiplication operations that multiply by 1.0
~~~~

### Original summary

~~~~text
Eliminating AscendC::Muls operations that multiply tensors by 1.0 reduces arithmetic operations by 29.6% (from 138.6M to 97.5M FLOPS) and improves performance by 11.6%. These identity operations consume vector unit cycles and increase pipeline stalls even though they don't change the data values.
~~~~

### Original root mechanism

~~~~text
Identity multiplication operations (multiplying by 1.0) introduce unnecessary arithmetic overhead and pipeline stalls despite being mathematically redundant
~~~~

### Original causal chain

~~~~text
Removing 6 redundant Muls operations → reduces aiv_vec_fops by 41M operations (-29.6%) → decreases aiv_vec_wait_ratio_mean from 0.255 to 0.354 (-27.99%) → reduces task_duration_us from 1273.3μs to 1125.3μs
~~~~

## 适用条件

- When using AscendC::Muls with constant value 1.0 (identity operation)
- When performing redundant arithmetic operations that don't change data values
- When vector unit utilization (vec_ratio_mean) is high (>0.65) and could be optimized
- When pipeline stalls (aiv_vec_wait_ratio_mean) exceed 0.25

## 失效边界

- When multiplication by 1.0 is required for data type conversion or precision preservation
- When the compiler cannot optimize away the identity operation automatically
- When the operation is part of a larger fused kernel where removing it breaks the pattern

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- mul_sigmoid/op_kernel/mul_sigmoid.cpp

### 改动符号

- Ascend::MulSigmoid::Compute

### 代码签名

- AllocTensor
- DeQue
- bool_control
- queue_api
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`a414089c196ee0f50addc5b3ee901f2327c5e18176846ae5136e104fb53057e3`

````diff
--- Before/op_kernel/mul_sigmoid.cpp
+++ After/op_kernel/mul_sigmoid.cpp
@@ -68,32 +68,10 @@
                 AscendC::LocalTensor<half>  x1_local = this->x1_queue.DeQue<half>();
                 AscendC::LocalTensor<half> out_local = this->out_queue.AllocTensor<half>();

-                 /* 计算过程 */
-                 AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);
+                  /* 计算过程 */
+                  AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);

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
-                 AscendC::Sigmoid<half, false>(out_local, x1_local, this->tile_length);
-
-                 // REDUNDANT: Add unnecessary operations on out_local that don't affect final result
-                 // Multiply out_local by 1.0 (identity operation)
-                 AscendC::Muls<half>(out_local, out_local, static_cast<half>(1.0f), this->tile_length);
-
-                 // Another redundant multiply by 1.0
-                 AscendC::Muls<half>(out_local, out_local, static_cast<half>(1.0f... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1273.345459`
- fast duration us：`1125.30249`
- improvement ratio：`0.11626300463368587`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005371 -> 0.012453 (+131.86%)
- arithmetic.aiv_vec_fops: 9.75448e+07 -> 1.38564e+08 (+42.05%)
- aiv_utilization.mte2_ratio_mean: 0.318586 -> 0.225245 (-29.30%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.354123 -> 0.255011 (-27.99%)
- memory_ub.ub_write_bw_vector_gbps: 187.23 -> 226.34 (+20.89%)

### Affected metric groups

- pipeline_stalls
- arithmetic
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "arithmetic.aiv_vec_fops",
  "aiv_utilization.mte2_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "memory_ub.ub_write_bw_vector_gbps"
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

- 1. Search for AscendC::Muls calls with constant value 1.0 in op_kernel files
- 2. Verify the multiplication doesn't serve other purposes (type conversion, precision)
- 3. Remove the Muls operation entirely when it's a pure identity operation
- 4. Test that the mathematical equivalence is preserved after removal
- 5. Profile the operator to verify reduced aiv_vec_fops and improved vec_ratio_mean

## 检索线索

- identity multiplication
- redundant Muls
- AscendC::Muls
- multiply by 1.0
- arithmetic overhead
- pipeline stalls
- vector unit utilization
- remove.redundant.ops
