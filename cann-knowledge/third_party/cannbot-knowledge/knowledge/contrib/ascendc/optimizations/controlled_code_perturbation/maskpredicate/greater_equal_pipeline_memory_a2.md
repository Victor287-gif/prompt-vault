---
type: Optimization
title: "greater_equal：流水与内存优化"
description: "greater_equal 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/docs/GreaterEqual.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/framework/tf_plugin/tensorflow_greater_equal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/op_host/greater_equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/op_host/greater_equal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/op_kernel/greater_equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/tests/st/GreaterEqual_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/tests/st/test_greater_equal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/greater_equal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# greater_equal：流水与内存优化

## 来源信息

- experience ID：`19c2f0003a63838c`
- 源 schema version：`v2`
- 算子：`greater_equal`
- source iteration：`round_013_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.buffer.reduction`
- canonical family：`pipeline.buffer.reduction`
- 源记录时间：`2026-04-20T21:20:51`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates double-buffering overhead
~~~~

### Original summary

~~~~text
Setting BUFFER_NUM=1 instead of BUFFER_NUM=2 in Ascend C kernels removes double-buffering overhead when the operator is scalar-bound rather than memory-bound. This optimization reduces pipeline stalls and improves vector unit utilization by preventing unnecessary overlap between data transfer and computation stages. Apply this optimization when profiling shows high MTE wait ratios and low vector utilization despite sufficient memory bandwidth.
~~~~

### Original root mechanism

~~~~text
Reduced BUFFER_NUM from 2 to 1, eliminating double-buffering and preventing overlap between data transfer and computation stages.
~~~~

### Original causal chain

~~~~text
Reduced BUFFER_NUM from 2 to 1 → Decreased pipeline stalls (aiv_vec_bankgroup_cflt_ratio_mean -66.3%, aiv_vec_total_cflt_ratio_mean -45.8%) → Improved MTE utilization (mte3_ratio_mean -48.0%, mte2_ratio_mean -39.7%) → Reduced task duration from 2167.5µs to 1916.7µs (11.6% improvement)
~~~~

## 适用条件

- Operator is scalar-bound (scalar_ratio_mean > 0.2)
- MTE wait ratios are high (aiv_mte2_wait_ratio_mean > 0.99, aiv_mte3_wait_ratio_mean > 0.99)
- Vector utilization is low (vec_ratio_mean < 0.5)
- Memory bandwidth usage is low (gm_to_ub_bw_usage_rate_pct < 15)
- Using BUFFER_NUM=2 in Ascend C kernel with constexpr buffer allocation

## 失效边界

- Operator is memory-bound with high GM bandwidth usage (> 80% of peak)
- Vector utilization is already high (vec_ratio_mean > 0.7)
- Pipeline shows no bank group conflicts (aiv_vec_bankgroup_cflt_ratio_mean < 0.01)
- Using asynchronous data transfer patterns that benefit from double-buffering

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- greater_equal/op_kernel/greater_equal.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- ub

### 原始 diff

- 源 diff SHA-256：`a00c73e1a4bcd2e368ccf761a469e58afdb6521f53d72df7dc1757ca17ff43a0`

````diff
--- Before/op_kernel/greater_equal.cpp
+++ After/op_kernel/greater_equal.cpp
@@ -8,7 +8,7 @@

 namespace GreaterEqualK{

-constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
+constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
 constexpr float NEGATIVE_ONE_FP32 = -1.0F;
 constexpr float POSITIVE_ONE_FP32 = 1.0F;
 constexpr int32_t NEGATIVE_ONE_I32 = -1;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2167.523438`
- fast duration us：`1916.698364`
- improvement ratio：`0.1157196594060544`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.074214 -> 0.024999 (-66.31%)
- aiv_utilization.mte3_ratio_mean: 0.454762 -> 0.236546 (-47.98%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.102925 -> 0.055768 (-45.82%)
- aiv_utilization.mte2_ratio_mean: 0.970693 -> 0.585787 (-39.65%)
- aiv_utilization.scalar_ratio_mean: 0.28318 -> 0.21931 (-22.55%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- memory_ub
- arithmetic

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "aiv_utilization.scalar_ratio_mean"
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

- Check greater_equal/op_kernel/greater_equal.cpp for BUFFER_NUM definition
- Change constexpr int32_t BUFFER_NUM = 2; to constexpr int32_t BUFFER_NUM = 1;
- Verify the operator uses compile-time buffer allocation with constexpr
- Profile before and after to confirm reduction in pipeline stalls and MTE wait ratios
- Ensure memory bandwidth usage remains below 80% of peak to avoid becoming memory-bound

## 检索线索

- BUFFER_NUM
- double-buffering
- pipeline stalls
- MTE wait
- scalar-bound
- Ascend C
- vector utilization
- bank group conflicts
- constexpr buffer
- greater_equal
