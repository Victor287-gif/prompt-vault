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

- experience ID：`4c0063d35fde3c20`
- 源 schema version：`v2`
- 算子：`mul_sigmoid`
- source iteration：`round_016_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`redundant.op.elimination`
- canonical family：`redundant.op.elimination`
- 源记录时间：`2026-04-21T00:38:21`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant identity multiplication operations
~~~~

### Original summary

~~~~text
Removing redundant AscendC::Muls operations that multiply by 1.0 reduces floating-point operation count by 29.5% (from 138.6M to 97.5M FLOPS) and improves performance by 11.4%. This optimization works by eliminating mathematically unnecessary operations that still consume vector unit resources and increase pipeline stalls.
~~~~

### Original root mechanism

~~~~text
Removing multiplication by 1.0 identity operations reduces floating-point operation count and vector unit utilization overhead, improving overall pipeline efficiency
~~~~

### Original causal chain

~~~~text
Eliminate 5 redundant Muls(half) operations multiplying by static_cast<half>(1.0f) → Reduce aiv_vec_fops by 41M operations (-29.5%) → Decrease vector wait ratio from 0.356 to 0.255 (-28.3%) → Reduce MTE2 utilization from 0.320 to 0.225 (-29.5%) → Improve task duration from 1273µs to 1128µs (-11.4%)
~~~~

## 适用条件

- AscendC kernels contain Muls operations multiplying by 1.0 or other identity values
- Vector operations show high aiv_vec_fops with low computational benefit
- MTE2 utilization exceeds 30% with high vector wait ratios
- Kernels have redundant arithmetic operations marked as identity transformations

## 失效边界

- Multiplication by 1.0 serves architectural purpose (e.g., pipeline synchronization)
- Identity operations are required for data dependency or timing constraints
- Compiler optimizations already eliminate such redundant operations
- Operations have side effects beyond value computation

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- mul_sigmoid/op_kernel/mul_sigmoid.cpp

### 改动符号

- Compute

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
- fast duration us：`1127.922607`
- improvement ratio：`0.11420534072050273`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.005367 -> 0.012453 (+132.03%)
- arithmetic.aiv_vec_fops: 9.75448e+07 -> 1.38564e+08 (+42.05%)
- aiv_utilization.mte2_ratio_mean: 0.319673 -> 0.225245 (-29.54%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.355704 -> 0.255011 (-28.31%)
- memory_ub.ub_write_bw_vector_gbps: 187.1 -> 226.34 (+20.97%)

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

- Search for AscendC::Muls calls with constant value 1.0 in op_kernel/*.cpp files
- Identify operations where dst = src * 1.0f or similar identity transformations
- Verify mathematical equivalence before removal using unit tests
- Profile before/after to confirm aiv_vec_fops reduction and performance improvement
- Check for MTE2 utilization reduction as validation signal

## 检索线索

- redundant multiplication
- identity operation
- Muls 1.0
- scalar overhead
- vector wait ratio
- MTE2 utilization
- floating point operations
- AscendC optimization
