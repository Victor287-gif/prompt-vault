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

- experience ID：`ed70468be3a5e4cc`
- 源 schema version：`v2`
- 算子：`mul_sigmoid`
- source iteration：`round_018_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.ops`
- canonical family：`remove.redundant.ops`
- 源记录时间：`2026-04-21T01:25:18`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant identity multiplication operations in Ascend C kernels
~~~~

### Original summary

~~~~text
In Ascend C vector kernels, removing redundant multiplication-by-1.0 operations that don't affect computational results improves performance by 11%. These identity operations consume vector unit resources and create unnecessary pipeline stalls, particularly affecting aiv_vec_wait_ratio_mean and aiv_vec_bank_cflt_ratio_mean. Apply this optimization when profiling shows high vector wait ratios and bank conflict stalls despite sufficient computational work.
~~~~

### Original root mechanism

~~~~text
Removing identity multiplication-by-1.0 operations reduces arithmetic instruction overhead and vector unit contention, allowing better pipeline utilization.
~~~~

### Original causal chain

~~~~text
Remove 6 redundant AscendC::Muls<half> operations with static_cast<half>(1.0f) -> Reduces aiv_vec_fops by 41.2M operations (42% reduction) -> Decreases vector wait ratio by 28.1% and bank conflict ratio by 61.9% -> Improves vector unit utilization from 66.3% to 75.7% -> Reduces task duration by 139.8μs (11% improvement)
~~~~

## 适用条件

- Ascend C kernel contains AscendC::Muls operations with constant 1.0 multiplier
- Profile shows high aiv_vec_wait_ratio_mean (>0.3) and low vector unit utilization (<0.7)
- aiv_vec_fops count is significantly higher than necessary for the computation
- Performance is scalar-overhead bound with high pipeline stall ratios

## 失效边界

- Multiplication by 1.0 is required for data type conversion or precision preservation
- The operation is part of a fused kernel where identity multiplication serves as a placeholder
- Performance is already memory-bound with low vector unit utilization

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
- bool_control
- queue_api
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`aeca9445247d49f1d97b85e5d3a33994a0232e819d64e455141180b1a5094e4f`

````diff
--- Before/op_kernel/mul_sigmoid.cpp
+++ After/op_kernel/mul_sigmoid.cpp
@@ -69,31 +69,8 @@
                 AscendC::LocalTensor<half> out_local = this->out_queue.AllocTensor<half>();

                  /* 计算过程 */
-                 AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);
-
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
-                 AscendC::Muls<half>(out_local, out_local, static_cast<half>(1.0f), this->tile_length);
+                  AscendC::Muls<half>(x1_local, x1_local, this->t1, this->tile_length);
+                  AscendC::Sigmoid<half, false>(out_local, x1_local, this->tile_length);

 ... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1273.345459`
- fast duration us：`1133.582642`
- improvement ratio：`0.10976032938442341`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005344 -> 0.012453 (+133.03%)
- arithmetic.aiv_vec_fops: 9.75448e+07 -> 1.38564e+08 (+42.05%)
- aiv_utilization.mte2_ratio_mean: 0.321664 -> 0.225245 (-29.98%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.354823 -> 0.255011 (-28.13%)
- memory_ub.ub_write_bw_vector_gbps: 186.31 -> 226.34 (+21.49%)

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

- Search for AscendC::Muls calls with static_cast<half>(1.0f) or similar identity constants in mul_sigmoid.cpp
- Verify the multiplication is truly redundant by checking if result equals input
- Remove all redundant multiplication-by-1.0 operations while preserving data dependencies
- Recompile and profile to confirm aiv_vec_wait_ratio_mean reduction and vector unit utilization improvement
- Monitor aiv_vec_fops count to ensure only identity operations were removed

## 检索线索

- redundant multiplication
- identity operation
- AscendC::Muls
- scalar overhead
- vector wait ratio
- bank conflict
- mul_sigmoid
- remove.redundant.ops
