---
type: Optimization
title: "gelu：流水与内存优化"
description: "gelu 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/Gelu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/tensorflow_gelu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/Gelu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/test_gelu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu：流水与内存优化

## 来源信息

- experience ID：`22c6654936d219aa`
- 源 schema version：`v2`
- 算子：`gelu`
- source iteration：`round_004_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.buffer.reduction`
- canonical family：`pipeline.buffer.reduction`
- 源记录时间：`2026-04-20T19:35:58`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce double buffering to single buffer
~~~~

### Original summary

~~~~text
For memory-bound operators with small working sets, reducing double buffering to single buffering can improve performance by eliminating pipeline stalls. This optimization works when the operator's compute time is shorter than the memory transfer time, allowing the vector unit to operate more efficiently without waiting for buffer swaps. Apply this optimization to vector operators where aiv_vec_wait_ratio_mean is high and mte2/mte3 utilization ratios are low.
~~~~

### Original root mechanism

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates pipeline overlap between data transfer and computation stages, reducing wait stalls and improving vector unit utilization.
~~~~

### Original causal chain

~~~~text
Reducing BUFFER_NUM from 2 to 1 → aiv_vec_wait_ratio_mean decreases from 0.232542 to 0.001194 (-99.5%) → aiv_utilization.vec_ratio_mean increases from 0.777436 to 0.983007 (+26.4%) → latency.task_duration_us decreases from 1375.968μs to 1065.081μs (-22.6%)
~~~~

## 适用条件

- Operator has high aiv_vec_wait_ratio_mean (>0.1)
- Operator shows low mte2_ratio_mean (<0.5) and mte3_ratio_mean (<0.2)
- Memory bandwidth usage (gm_to_ub_bw_usage_rate_pct, ub_to_gm_bw_usage_rate_pct) is below 10%
- Working set fits within single buffer size
- Operator uses BUFFER_NUM template parameter with value 2

## 失效边界

- Operator requires overlapping computation with data transfer for optimal performance
- Memory bandwidth is fully utilized (>80% bandwidth usage)
- Working set exceeds single buffer capacity
- Operator has high icache_miss_rate_mean (>0.01)
- Vector unit utilization (vec_ratio_mean) is already high (>0.9)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- gelu/op_kernel/gelu.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- template
- ub

### 原始 diff

- 源 diff SHA-256：`fc6d6d9b719ce349ba3eb6f4557a5ac70086e510c878af2c5dc440350121c19a`

````diff
--- Before/op_kernel/gelu.cpp
+++ After/op_kernel/gelu.cpp
@@ -1,6 +1,6 @@
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;
+constexpr int32_t BUFFER_NUM = 1;

 template <typename T>
 class KernelGelu
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1375.967529`
- fast duration us：`1065.081299`
- improvement ratio：`0.2259400919336666`
- machine confidence：`0.95`

### Profile evidence

- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.001194 -> 0.232542 (+19375.88%)
- aiv_utilization.mte2_ratio_mean: 0.462171 -> 0.208212 (-54.95%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.017643 -> 0.025336 (+43.60%)
- aiv_utilization.mte3_ratio_mean: 0.158951 -> 0.098343 (-38.13%)
- latency.task_duration_us: 1065.08 -> 1375.97 (+29.19%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- latency
- critical_path

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.001194,
      "after": 0.232542,
      "delta": 0.231348,
      "rel": 193.75879396984925
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.462171,
      "after": 0.208212,
      "delta": -0.253959,
      "rel": -0.5494914220061405
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.017643,
      "after": 0.025336,
      "delta": 0.007693000000000002,
      "rel": 0.43603695516635504
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.158951,
      "after": 0.098343,
      "delta": -0.06060800000000001,
      "rel": -0.3812998974526741
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1065.081,
      "after": 1375.968,
      "delta": 310.88700000000017,
      "rel": 0.2918904759356333
    }
  }
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

- Check gelu.cpp for BUFFER_NUM template parameter
- Change constexpr int32_t BUFFER_NUM = 2 to constexpr int32_t BUFFER_NUM = 1
- Verify operator is memory-bound with low memory bandwidth usage
- Monitor aiv_vec_wait_ratio_mean reduction and vec_ratio_mean increase after change
- Test with small to medium tensor sizes where single buffering is sufficient

## 检索线索

- BUFFER_NUM
- double_buffering
- single_buffer
- pipeline_stalls
- aiv_vec_wait_ratio
- vector_utilization
- memory_bound
- AscendC
- gelu
- buffer_reduction
