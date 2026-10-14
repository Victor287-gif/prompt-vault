---
type: Optimization
title: "foreach_zero_inplace：内存优化"
description: "foreach_zero_inplace 算子的内存优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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
# foreach_zero_inplace：内存优化

## 来源信息

- experience ID：`84d53acc1ce2a50a`
- 源 schema version：`v2`
- 算子：`foreach_zero_inplace`
- source iteration：`round_005_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.access.redundant`
- canonical family：`memory.access.redundant`
- 源记录时间：`2026-04-19T08:33:40`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant read-before-write operations in zeroing loops
~~~~

### Original summary

~~~~text
In Ascend C operators performing zero-initialization, avoid redundant read-before-write patterns where data is read from local memory immediately before being overwritten with zero. This optimization reduces UB write bandwidth usage by 37.2% (from 5.3003 Gbps to 3.3281 Gbps) and decreases MTE wait ratios by 38.3-87.5%, leading to 37.2% faster execution. Apply this pattern to any operator that writes constant values (like zero) to local memory buffers.
~~~~

### Original root mechanism

~~~~text
Removing redundant local memory read operations before writing zero eliminates unnecessary memory traffic and reduces pipeline stalls in the hot path.
~~~~

### Original causal chain

~~~~text
Removing redundant dataLocal.GetValue(i) calls before dataLocal.SetValue(i, (T)0) → reduces UB write bandwidth from 5.3003 to 3.3281 Gbps and decreases MTE2/MTE3 wait ratios → reduces aiv_vector0_time_us_mean from 50078 to 31445 us → improves task_duration_us from 50139 to 31474 us.
~~~~

## 适用条件

- Operator performs zero-initialization or constant-value writes to local memory buffers
- Code pattern includes dataLocal.GetValue() calls immediately before dataLocal.SetValue() with constant values
- Profile shows high UB write bandwidth (ub_write_bw_scalar_gbps > 3 Gbps) with low computational intensity
- MTE2/MTE3 wait ratios are elevated (>0.006) relative to scalar utilization

## 失效边界

- Actual data dependency requires reading previous value before writing
- Memory access pattern requires barrier synchronization between read and write
- Operator needs to preserve or validate existing values before overwriting

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_zero_inplace/op_kernel/foreach_zero_inplace.cpp

### 改动符号

- ForeachZeroInplaceND<T>::ComputeAndCopyOut

### 代码签名

- GetValue
- SetFlag
- SetValue
- pipebarrier
- scalar_value_access
- ub

### 原始 diff

- 源 diff SHA-256：`8f1dd753b1881b6789071d5998f8fc7b4b6fa995e82223ed24169d3a3346a291`

````diff
--- Before/op_kernel/foreach_zero_inplace.cpp
+++ After/op_kernel/foreach_zero_inplace.cpp
@@ -117,6 +117,14 @@
         dataLocal.SetValue(i, (T)0);
     }

+    // Additional degradation: Redundant read-before-write for each element
+    for (int64_t i = 0; i < dataCount; ++i) {
+        T current = dataLocal.GetValue(i);
+        // The read is redundant since we just wrote zero above
+        // but we perform it anyway to add memory access overhead
+        (void)current; // Use variable to avoid unused warning
+    }
+
     // Transport can be performed only after the Muls is complete.
     event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
     SetFlag<HardEvent::V_MTE3>(eventID1);
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`100256.125`
- fast duration us：`62970.361328`
- improvement ratio：`0.37190509479595385`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.vec_ratio_mean: 1e-06 -> 0 (-100.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 1e-06 -> 0 (-100.00%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 4e-05 -> 5e-06 (-87.50%)
- latency.task_duration_us: 31473.5 -> 50139.4 (+59.31%)
- critical_path.aiv_vector0_time_us_mean: 31444.7 -> 50078 (+59.26%)

### Affected metric groups

- aiv_utilization
- arithmetic
- pipeline_stalls
- latency

### Changed metrics top

~~~~json
[
  "aiv_utilization.vec_ratio_mean",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "memory_ub.ub_write_bw_scalar_gbps"
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

- Inspect ForeachZeroInplaceND<T>::ComputeAndCopyOut function in foreach_zero_inplace.cpp
- Identify loops that read from dataLocal via GetValue() before SetValue() with constant zero
- Remove redundant GetValue() calls when the value is immediately overwritten with constant
- Verify that scalar UB write bandwidth (ub_write_bw_scalar_gbps) decreases after optimization
- Check that MTE2/MTE3 wait ratios (aiv_mte2_wait_ratio_mean, aiv_mte3_wait_ratio_mean) improve

## 检索线索

- foreach_zero_inplace
- zero-initialization
- redundant memory access
- read-before-write
- UB bandwidth
- MTE wait
- scalar memory
- dataLocal
- GetValue
- SetValue
