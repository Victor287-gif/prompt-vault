---
type: Optimization
title: "fast_gelu_grad：流水与并行优化"
description: "fast_gelu_grad 算子的流水与并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/docs/FastGeluGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/tf_plugin/tensorflow_fast_gelu_grad_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_host/fast_gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_host/fast_gelu_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_kernel/fast_gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/FastGeluGrad_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/test_fast_gelu_grad.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# fast_gelu_grad：流水与并行优化

## 来源信息

- experience ID：`8161e53dacdacdf1`
- 源 schema version：`v2`
- 算子：`fast_gelu_grad`
- source iteration：`round_005_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism.reduce`
- canonical family：`pipeline.parallelism.reduce`
- 源记录时间：`2026-04-20T14:44:09`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Pipeline serialization via single buffering
~~~~

### Original summary

~~~~text
Applying single buffering (BUFFER_NUM=1) reduces pipeline stalls by serializing data transfers and computation. This optimization improves performance when scalar overhead dominates and double-buffering introduces excessive pipeline stalls. Use this technique for vector operators where MTE utilization is high and vector wait ratios indicate pipeline contention.
~~~~

### Original root mechanism

~~~~text
Reduced buffer queue size from 2 to 1, eliminating double-buffering parallelism that overlaps computation with data transfer.
~~~~

### Original causal chain

~~~~text
Reducing BUFFER_NUM from 2 to 1 -> Decreased MTE2 utilization from 0.93952 to 0.398819 and MTE3 utilization from 0.397178 to 0.097549 -> Reduced vector wait ratio from 0.444715 to 0.32557 -> Improved task duration from 1539.571μs to 1279.286μs (16.9% faster)
~~~~

## 适用条件

- Vector operators with high MTE utilization (>0.9 for MTE2 or >0.3 for MTE3)
- Operators showing high vector wait ratios (>0.4) indicating pipeline stalls
- Memory-bound kernels where double-buffering creates contention
- AscendC kernels using BUFFER_NUM=2 with scalar overhead bottlenecks

## 失效边界

- Compute-bound kernels where overlapping computation with data transfer is beneficial
- Operators with low MTE utilization (<0.5 for MTE2 and <0.2 for MTE3)
- Kernels where memory bandwidth is the primary bottleneck
- When aiv_vec_wait_ratio_mean is already low (<0.2)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/fast_gelu_grad.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- ub

### 原始 diff

- 源 diff SHA-256：`7c12aa5d3fe5c811577d715e4ff21e99d39a8b770ccd6328293daebc8a44d238`

````diff
--- Before/op_kernel/fast_gelu_grad.cpp
+++ After/op_kernel/fast_gelu_grad.cpp
@@ -3,7 +3,7 @@
  */
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
+constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

 class KernelFastGeluGrad {
 public:
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1539.570801`
- fast duration us：`1279.285645`
- improvement ratio：`0.16906345315911206`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.mte3_ratio_mean: 0.397178 -> 0.097549 (-75.44%)
- aiv_utilization.mte2_ratio_mean: 0.93952 -> 0.398819 (-57.55%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.027652 -> 0.014107 (-48.98%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.32557 -> 0.444715 (+36.60%)
- critical_path.aiv_vector0_time_us_mean: 1267.31 -> 1529.28 (+20.67%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "aiv_utilization.mte3_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean"
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

- In op_kernel/fast_gelu_grad.cpp, change BUFFER_NUM from 2 to 1
- Verify MTE utilization metrics: aiv_utilization.mte2_ratio_mean and aiv_utilization.mte3_ratio_mean should decrease significantly
- Monitor aiv_vec_wait_ratio_mean - it should decrease with single buffering
- Check that scalar_ratio_mean and vec_ratio_mean increase, indicating better compute utilization
- Validate that task_duration_us decreases by 10-20% for similar vector operators

## 检索线索

- BUFFER_NUM
- single_buffering
- pipeline_serialization
- MTE_utilization
- vector_wait_ratio
- scalar_overhead
- AscendC
- fast_gelu_grad
- vector_operator
- pipeline_stalls
