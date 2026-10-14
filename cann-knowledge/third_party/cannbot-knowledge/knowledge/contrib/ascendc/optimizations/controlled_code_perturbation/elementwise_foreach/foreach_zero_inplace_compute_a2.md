---
type: Optimization
title: "foreach_zero_inplace：计算优化"
description: "foreach_zero_inplace 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/docs/aclnnForeachZeroInplace.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/op_host/foreach_zero_inplace.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/op_kernel/foreach_zero_inplace.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/op_kernel/foreach_zero_inplace.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_zero_inplace/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_zero_inplace：计算优化

## 来源信息

- experience ID：`b890abc1ac19b4e3`
- 源 schema version：`v2`
- 算子：`foreach_zero_inplace`
- source iteration：`round_000_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`computation.loop.scalarization`
- canonical family：`computation.loop.scalarization`
- 源记录时间：`2026-04-19T07:55:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use vectorized Duplicate for zero initialization instead of scalar loops
~~~~

### Original summary

~~~~text
Using the Duplicate API for zero initialization instead of scalar loops significantly improves performance by leveraging vectorized hardware operations. This optimization reduces pipeline stalls and improves memory bandwidth utilization. Apply this technique when initializing arrays with constant values in Ascend C kernels.
~~~~

### Original root mechanism

~~~~text
Vectorized Duplicate operations process multiple elements per instruction, eliminating per-element loop overhead and reducing pipeline stalls from scalar operations.
~~~~

### Original causal chain

~~~~text
Replace scalar loop with Duplicate -> Vector utilization increases from 1e-06 to 0.077661 -> Vector FLOPS increase from 128 to 5,654,976 -> UB vector write bandwidth increases from 0 to 31 Gbps -> Task duration reduces from 31,473μs to 1,358μs
~~~~

## 适用条件

- When initializing local tensors with constant values in Ascend C kernels
- When using LocalTensor<T> allocations from data queues
- When the initialization value is a compile-time constant (e.g., zero)
- When operating on contiguous memory regions in ComputeAndCopyOut methods

## 失效边界

- When initialization requires per-element computation or non-constant values
- When working with non-contiguous or strided memory access patterns
- When the data type doesn't support vectorized operations
- When the operation must be conditional or have data-dependent branching

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/foreach_zero_inplace.cpp

### 改动符号

- ForeachZeroInplaceND::ComputeAndCopyOut

### 代码签名

- AllocTensor
- SetValue
- bool
- bool_control
- contiguous_access
- queue_api
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`a6d87eda7fce77e7bc4a52dd7017da77a227ef02252fbe629ff89f2596ae41ea`

````diff
--- Before/op_kernel/foreach_zero_inplace.cpp
+++ After/op_kernel/foreach_zero_inplace.cpp
@@ -112,7 +112,10 @@
 __aicore__ inline void ForeachZeroInplaceND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount, bool isRemainder) {
     LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();

-    Duplicate(dataLocal, (T)0, dataCount);
+    // Degraded: Replace vectorized Duplicate with scalar loop
+    for (int64_t i = 0; i < dataCount; ++i) {
+        dataLocal.SetValue(i, (T)0);
+    }

     // Transport can be performed only after the Muls is complete.
     event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`62970.361328`
- fast duration us：`2713.254272`
- improvement ratio：`0.9569122009977488`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.030696 -> 0.989961 (+3125.05%)
- critical_path.aiv_vector0_time_us_mean: 1345.1 -> 31444.7 (+2237.73%)
- latency.task_duration_us: 1358.25 -> 31473.5 (+2217.22%)
- memory_ub.ub_write_bw_vector_gbps: 31 -> 0 (-100.00%)
- memory_ub.ub_write_bw_scalar_gbps: 0 -> 5.3003 (+100.00%)

### Affected metric groups

- aiv_utilization
- critical_path
- latency
- memory_ub

### Changed metrics top

~~~~json
[
  "aiv_utilization.scalar_ratio_mean: 0.030696 -> 0.989961 (+3125.05%)",
  "critical_path.aiv_vector0_time_us_mean: 1345.096 -> 31444.705 (+2237.73%)",
  "latency.task_duration_us: 1358.247 -> 31473.531 (+2217.22%)",
  "memory_ub.ub_write_bw_vector_gbps: 31.0 -> 0.0 (-100.00%)",
  "memory_ub.ub_write_bw_scalar_gbps: 0.0 -> 5.3003 (+100.00%)"
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

- Replace scalar loops of the form 'for (int64_t i = 0; i < dataCount; ++i) { dataLocal.SetValue(i, (T)0); }' with 'Duplicate(dataLocal, (T)0, dataCount);'
- Apply in ForeachZeroInplaceND::ComputeAndCopyOut method in op_kernel/foreach_zero_inplace.cpp
- Ensure the tensor is properly allocated via dataQueue.AllocTensor<T>() before calling Duplicate
- Maintain the same event synchronization pattern with GetTPipePtr()->FetchEventID(HardEvent::V_MTE3)

## 检索线索

- Duplicate
- vectorized
- zero initialization
- scalar loop
- SetValue
- LocalTensor
- foreach_zero_inplace
- Ascend C
- ComputeAndCopyOut
- AllocTensor
