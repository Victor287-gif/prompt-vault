---
type: Optimization
title: "is_inf：计算优化"
description: "is_inf 算子的计算优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_kernel/is_inf.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_kernel/is_inf.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# is_inf：计算优化

## 来源信息

- experience ID：`67fbbbb2922de1e1`
- 源 schema version：`v2`
- 算子：`is_inf`
- source iteration：`round_014_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computation.redundant.vector.operation`
- canonical family：`computation.redundant.vector.operation`
- 源记录时间：`2026-04-20T22:01:57`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant vector min operations in infinity check
~~~~

### Original summary

~~~~text
Removing redundant vector min operations in the CompareInf function reduces vector floating-point operations by 27.8% (from 77.2M to 60.4M) and improves performance by 6.4%. This optimization applies when min/max operations are performed on values that are already bounded by the same limits, such as after previous clamping operations.
~~~~

### Original root mechanism

~~~~text
Redundant Mins operation on values already clamped to 0 or -1 adds unnecessary vector computation that increases vector floating-point operations and memory bandwidth usage.
~~~~

### Original causal chain

~~~~text
Remove redundant Mins(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount) → Reduces aiv_vec_fops by 16.8M operations → Decreases vector computation overhead → Lowers vector wait ratio by 9.2% → Reduces task duration by 178.8μs (6.4% improvement)
~~~~

## 适用条件

- Vector operations follow clamping operations with the same bounds
- IsInf operator or similar infinity detection logic using vector comparisons
- Code contains redundant min/max operations after previous bounds checks
- Performance profiling shows high aiv_vec_fops and vector wait ratios

## 失效边界

- Min/max operations have different bounds than previous clamping
- Vector operations require explicit bounds for correctness
- Code uses dynamic bounds that may change between operations

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- is_inf/op_kernel/is_inf.cpp

### 改动符号

- IsInf::CompareInf

### 代码签名

- contiguous_access
- ub

### 原始 diff

- 源 diff SHA-256：`4045c3c9cbd4f39e2c566e1c8c50cb45efe3bebc20eebd570af3c051f9fbf92f`

````diff
--- Before/op_kernel/is_inf.cpp
+++ After/op_kernel/is_inf.cpp
@@ -132,7 +132,11 @@
         Adds(tmpInt32Tensor, tmpInt32Tensor, (int32_t)-INF_MASK, actualCalCount);
         And(ubX, ubX, cacheTensor, dataLength);
         Mins(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount);
+        // Add redundant arithmetic operations that don't change result
+        Adds(tmpInt32Tensor, tmpInt32Tensor, (int32_t)0, actualCalCount);  // Redundant add 0
         Muls(tmpInt32Tensor, tmpInt32Tensor, (int32_t)-1, actualCalCount);
+        // Redundant min/max operations
+        Mins(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount);  // Already clamped to 1
         Adds(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount);
         Cast(ubX.ReinterpretCast<float>(), tmpInt32Tensor, RoundMode::CAST_NONE, actualCalCount);
         Cast(ubX.ReinterpretCast<half>(), ubX.ReinterpretCast<float>(), RoundMode::CAST_NONE, actualCalCount);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2815.116211`
- fast duration us：`2636.352783`
- improvement ratio：`0.06350126055240148`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 6.03764e+07 -> 7.71536e+07 (+27.79%)
- aiv_utilization.scalar_ratio_mean: 0.259979 -> 0.220179 (-15.31%)
- memory_ub.ub_write_bw_vector_gbps: 107.45 -> 123.03 (+14.50%)
- memory_ub.ub_read_bw_vector_gbps: 122.84 -> 137.45 (+11.89%)
- aiv_utilization.vec_ratio_mean: 0.43801 -> 0.484379 (+10.59%)

### Affected metric groups

- arithmetic
- aiv_utilization
- memory_ub
- pipeline_stalls

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_fops",
  "aiv_utilization.scalar_ratio_mean",
  "memory_ub.ub_write_bw_vector_gbps",
  "memory_ub.ub_read_bw_vector_gbps",
  "aiv_utilization.vec_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.356142,
  "overall_deviation": 0.142457,
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

- Inspect IsInf::CompareInf function in op_kernel/is_inf.cpp for redundant min/max operations
- Check for Mins operations applied after values are already clamped to 0 or -1
- Remove redundant Adds(tmpInt32Tensor, tmpInt32Tensor, (int32_t)0, actualCalCount) operations
- Eliminate duplicate Mins(tmpInt32Tensor, tmpInt32Tensor, (int32_t)1, actualCalCount) when values already bounded
- Verify vector operation count reduction in arithmetic.aiv_vec_fops metric
- Monitor aiv_utilization.vec_ratio_mean and pipeline_stalls.aiv_vec_wait_ratio_mean improvements

## 检索线索

- redundant vector operations
- min max optimization
- infinity check
- vector computation reduction
- Ascend C optimization
- CompareInf function
- Mins operation
- vector floating-point operations
