---
type: Optimization
title: "reciprocal：计算优化"
description: "reciprocal 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/Reciprocal.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/tensorflow_reciprocal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_kernel/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/Reciprocal_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/test_reciprocal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reciprocal：计算优化

## 来源信息

- experience ID：`f55c5017cb96917a`
- 源 schema version：`v2`
- 算子：`reciprocal`
- source iteration：`round_013_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computation.redundant.duplication`
- canonical family：`computation.redundant.duplication`
- 源记录时间：`2026-04-21T01:00:53`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant reciprocal computation in vector kernels
~~~~

### Original summary

~~~~text
In Ascend C vector kernels performing reciprocal operations, removing duplicate calculation blocks reduces arithmetic operations by half, decreasing aiv_vec_fops from 3.49B to 1.75B. This optimization reduces vector wait ratio from 76.6% to 2.2% and improves MTE utilization while maintaining the same mathematical result. Apply this pattern when reciprocal operations appear multiple times in the same kernel with identical inputs.
~~~~

### Original root mechanism

~~~~text
Removing duplicate reciprocal calculations reduces arithmetic operations by 50% and eliminates pipeline stalls caused by redundant computation overhead.
~~~~

### Original causal chain

~~~~text
Removing redundant Div and Duplicate calls → arithmetic.aiv_vec_fops reduces by 50% → pipeline_stalls.aiv_vec_wait_ratio_mean decreases from 76.6% to 2.2% → latency.task_duration_us improves from 24.5ms to 13.7ms
~~~~

## 适用条件

- Kernel contains duplicate Div operations with identical inputs in the same code block
- Vector kernels show high aiv_vec_wait_ratio_mean (>70%) with moderate aiv_vec_fops
- Reciprocal operations appear multiple times in KernelReciprocal::Compute function
- Same AscendC::LocalTensor used for both Duplicate and Div operations repeatedly

## 失效边界

- Different input tensors require separate reciprocal calculations
- Kernel already has minimal arithmetic operations (aiv_vec_fops < 1e9)
- Pipeline stalls are caused by memory bandwidth limitations rather than computation
- MTE utilization is already low (<20%) indicating other bottlenecks

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reciprocal/op_kernel/reciprocal.cpp

### 改动符号

- KernelReciprocal::Compute

### 代码签名

- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`6198d647390a96d1a1ac0e0a9449bdc7645a03acf016e6c9901d3f5f61ed5c23`

````diff
--- Before/op_kernel/reciprocal.cpp
+++ After/op_kernel/reciprocal.cpp
@@ -93,9 +93,15 @@
         {
             Duplicate(yLocal, 1.0f, this->processDataNum);
             Div(yLocal, yLocal, xLocal, this->processDataNum);
+            // Redundant computation for degradation
+            Duplicate(yLocal, 1.0f, this->processDataNum);
+            Div(yLocal, yLocal, xLocal, this->processDataNum);
         }
         else if constexpr ( std::is_same_v< DTYPE_X, float16_t>)
         {
+            Duplicate(yLocal, static_cast<half>(1.0f), this->processDataNum);
+            Div(yLocal, yLocal, xLocal, this->processDataNum);
+            // Redundant computation for degradation
             Duplicate(yLocal, static_cast<half>(1.0f), this->processDataNum);
             Div(yLocal, yLocal, xLocal, this->processDataNum);
         }
@@ -103,6 +109,10 @@
         {
             AscendC::LocalTensor<float> p1 = tmp1.Get<float>();
             AscendC::LocalTensor<float> p2 = tmp2.Get<float>();
+            Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
+            Duplicate(p2, 1.0f, this->processDataNum);
+            Div(p1, p2, p1, this->processDataNum);
+            // Redundant computation for degradation
             Cast(p1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
             Duplicate(p2, 1.0f, this->processDataNum);
             Div(p1, p2, p1, this->processDataNum);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`24479.167969`
- fast duration us：`13726.915039`
- improvement ratio：`0.43924094738908076`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_fops: 1.74781e+09 -> 3.49281e+09 (+99.84%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.766372 -> 0.021657 (-97.17%)
- critical_path.aiv_vector0_time_us_mean: 13726.1 -> 24478.1 (+78.33%)
- latency.task_duration_us: 13726.9 -> 24479.2 (+78.33%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 41.45 -> 23.24 (-43.93%)

### Affected metric groups

- arithmetic
- pipeline_stalls
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_fops: 1.74781e+09 -> 3.49281e+09 (+99.84%)",
  "pipeline_stalls.aiv_vec_wait_ratio_mean: 0.766372 -> 0.021657 (-97.17%)",
  "critical_path.aiv_vector0_time_us_mean: 13726.1 -> 24478.1 (+78.33%)",
  "latency.task_duration_us: 13726.9 -> 24479.2 (+78.33%)",
  "memory_ub.ub_to_gm_bw_usage_rate_pct: 41.45 -> 23.24 (-43.93%)"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 1,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- Inspect KernelReciprocal::Compute function in reciprocal.cpp for duplicate Div calls
- Check for repeated Duplicate(yLocal, 1.0f, processDataNum) followed by Div(yLocal, yLocal, xLocal, processDataNum)
- Remove redundant computation blocks while maintaining the first valid calculation
- Verify that aiv_vec_fops reduces by approximately 50% after optimization
- Monitor pipeline_stalls.aiv_vec_wait_ratio_mean improvement from >70% to <5%

## 检索线索

- reciprocal
- vector
- redundant
- duplication
- Div
- Duplicate
- AscendC
- LocalTensor
- arithmetic
- pipeline_stalls
- aiv_vec_fops
- aiv_vec_wait_ratio_mean
