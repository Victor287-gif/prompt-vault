---
type: Optimization
title: "foreach_tanh：计算优化"
description: "foreach_tanh 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/docs/aclnnForeachTanh.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/op_host/foreach_tanh.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/op_kernel/foreach_tanh.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_tanh/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_tanh：计算优化

## 来源信息

- experience ID：`d31fd12c841102e8`
- 源 schema version：`v2`
- 算子：`foreach_tanh`
- source iteration：`round_001_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computation.loop.redundant.validation`
- canonical family：`computation.loop.redundant.validation`
- 源记录时间：`2026-04-19T08:25:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor boundary validation loops in compute kernels
~~~~

### Original summary

~~~~text
Removing redundant boundary validation loops in compute kernels improves performance by reducing pipeline stalls and increasing vector unit utilization. This optimization is effective when artificial validation loops are inserted after core computation in Ascend C operators, particularly in ForeachTriangle::Compute implementations. The technique eliminates unnecessary scalar operations that break pipeline efficiency while maintaining functional correctness.
~~~~

### Original root mechanism

~~~~text
Removing artificial computation loops that simulate expensive tensor boundary validation in ForeachTriangle::Compute method eliminates pipeline stalls and improves vector unit utilization.
~~~~

### Original causal chain

~~~~text
Removing artificial validation loop -> aiv_vec_wait_ratio_mean decreases from 0.642 to 0.134 (-79.2%) -> aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean drop to 0 (-100%) -> vector unit utilization (vec_ratio_mean) improves from 0.164 to 0.344 (+109%) -> task_duration_us reduces from 3371μs to 1626μs (-51.8%)
~~~~

## 适用条件

- ForeachTriangle::Compute implementations with artificial validation loops
- Operators showing high scalar_ratio_mean (>0.8) with low vec_ratio_mean (<0.3)
- High aiv_mte2_wait_ratio_mean or aiv_mte3_wait_ratio_mean (>0.8) indicating memory transfer engine stalls
- Code patterns with volatile dummy calculations after core compute operations

## 失效边界

- Actual boundary validation is required for correctness
- Operators already have optimized validation logic
- Scalar operations are essential for algorithm correctness

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_tanh/op_kernel/foreach_tanh.cpp

### 改动符号

- ForeachTriangle::Compute

### 代码签名

- EnQue
- FreeTensor
- TQue
- queue_api
- template
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`e9dad0f969d1203d6035a5236f86c298611b51acde4f94e777cd0eb2ead54594`

````diff
--- Before/op_kernel/foreach_tanh.cpp
+++ After/op_kernel/foreach_tanh.cpp
@@ -359,6 +359,15 @@
         InnerComputer<T, P, op, paramsCount> computer;
         computer.Compute(dataLocal, outLocal, float32Tensor, Base::maxCastDataCount, dataCount);

+        // Add redundant tensor boundary validation after computation
+        // This breaks pipeline efficiency by adding unnecessary work
+        volatile int64_t dummy_validation = 0;
+        for (int i = 0; i < 256; ++i) {
+            // Simulate expensive boundary check calculations
+            dummy_validation += (index * dataCount + i) % (dataCount + 1);
+            dummy_validation ^= (dummy_validation << 3) | (dummy_validation >> 5);
+        }
+
         Base::dataQueue.FreeTensor(dataLocal);
         Base::outQueue.template EnQue<T>(outLocal);
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`6742.715088`
- fast duration us：`3246.524903`
- improvement ratio：`0.5185137054392472`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.15192 -> 0.999248 (+557.75%)
- critical_path.aiv_vector0_time_us_mean: 1613.97 -> 3366.69 (+108.60%)
- latency.task_duration_us: 1626.15 -> 3370.77 (+107.28%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.991337 -> 0 (-100.00%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.991187 -> 0 (-100.00%)

### Affected metric groups

- aiv_utilization
- critical_path
- latency
- pipeline_stalls

### Changed metrics top

~~~~json
[
  "aiv_utilization.scalar_ratio_mean: 0.15192 -> 0.999248 (+557.75%)",
  "critical_path.aiv_vector0_time_us_mean: 1613.968 -> 3366.694 (+108.60%)",
  "latency.task_duration_us: 1626.152 -> 3370.768 (+107.28%)",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.991337 -> 0 (-100.00%)",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.991187 -> 0 (-100.00%)"
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

- 1. Search for ForeachTriangle::Compute implementations in foreach operators
- 2. Identify artificial validation loops with volatile dummy calculations after InnerComputer::Compute calls
- 3. Remove loops like 'for (int i = 0; i < 256; ++i) { dummy_validation += ... }' in op_kernel/foreach_tanh.cpp:362-369
- 4. Verify removal doesn't affect functional correctness by checking tensor boundaries are still validated elsewhere
- 5. Profile to confirm reduction in aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean to near zero

## 检索线索

- foreach_tanh
- ForeachTriangle::Compute
- tensor boundary validation
- redundant loop
- pipeline stall
- vector utilization
- scalar ratio
- mte wait ratio
- volatile dummy calculations
- artificial validation
