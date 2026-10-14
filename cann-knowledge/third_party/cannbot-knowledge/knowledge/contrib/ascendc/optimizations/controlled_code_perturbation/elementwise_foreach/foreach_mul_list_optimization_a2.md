---
type: Optimization
title: "foreach_mul_list：代码结构优化"
description: "foreach_mul_list 算子的代码结构优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/docs/aclnnForeachMulList.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_host/foreach_mul_list.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_kernel/foreach_mul_list.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_mul_list：代码结构优化

## 来源信息

- experience ID：`23e9b5cc502c2437`
- 源 schema version：`v2`
- 算子：`foreach_mul_list`
- source iteration：`round_006_refactor`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`code_structure.compiler_optimization`
- canonical family：`untitled`
- 源记录时间：`2026-04-14T23:29:52`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Avoid oversimplification of template-heavy Ascend C kernel headers
~~~~

### Original summary

~~~~text
When refactoring Ascend C operators, preserving template helper macros and complex type definitions in tiling headers maintains compiler optimization opportunities. Over-simplifying header files by removing utility macros can reduce vector utilization by 24.4% and memory bandwidth by 24.4%, even when functional correctness is preserved. This pattern applies to operators using template-based tiling data structures with multiple helper macros.
~~~~

### Original root mechanism

~~~~text
Removing template helper macros from tiling headers disrupts compiler's ability to optimize vector operations and memory access patterns.
~~~~

### Original causal chain

~~~~text
Removing FOREACH_* macro definitions from foreach_mul_list_tiling.h → compiler generates less optimized vector instructions → vector utilization decreases from 0.03445 to 0.02605 → memory bandwidth drops from 13.09/6.51 Gbps to 9.90/4.92 Gbps → pipeline stalls increase → task duration increases by 32.0%
~~~~

## 适用条件

- Refactoring Ascend C operators with template-heavy tiling headers
- Simplifying complex macro definitions in kernel header files
- Modifying foreach operator implementations with multiple helper macros
- Changing code structure in op_host/foreach_mul_list_tiling.h

## 失效边界

- Removing FOREACH_OPDEF_BEGIN/FOREACH_OPDEF_END_HOST_CONFIG macros without understanding compiler optimization dependencies
- Simplifying type conversion macros like DtypeTensor2Scalar and DtypeScalarToTensor2
- Changing parameter macro definitions (FOREACH_UNARY_PARAM, FOREACH_BINARY_LIST_PARAM, etc.)
- Refactoring without preserving all template instantiation patterns

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异


## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`16588.531739000002`
- fast duration us：`14530.310547000001`
- improvement ratio：`0.12407494673932339`
- machine confidence：`0.85`

### Profile evidence

- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.022798 -> 0.004015 (-82.39%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 6.8e-05 -> 2.4e-05 (-64.71%)
- critical_path.aiv_vector0_time_us_mean: 6422.3 -> 8489.53 (+32.19%)
- latency.task_duration_us: 6446.27 -> 8509.05 (+32.00%)
- memory_gm.aiv_main_mem_write_bw_gbps: 6.51 -> 4.92 (-24.42%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.022798,
      "after": 0.004015,
      "delta": -0.018782999999999998,
      "rel": -0.8238880603561716
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 6.8e-05,
      "after": 2.4e-05,
      "delta": -4.4e-05,
      "rel": -0.6470588235294118
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 6422.301,
      "after": 8489.532,
      "delta": 2067.230999999999,
      "rel": 0.32188323157074056
    }
  },
  {
    "latency.task_duration_us": {
      "before": 6446.269,
      "after": 8509.05,
      "delta": 2062.780999999999,
      "rel": 0.31999610937737766
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 6.51,
      "after": 4.92,
      "delta": -1.5899999999999999,
      "rel": -0.24423963133640553
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

- Preserve template helper macros in tiling headers (foreach_mul_list_tiling.h)
- Maintain type conversion functions (DtypeTensor2Scalar, DtypeScalarToTensor2) in header files
- Keep parameter macro definitions (FOREACH_*_PARAM) for compiler optimization hints
- Test vector utilization metrics (aiv_utilization.vec_ratio_mean) after header simplifications
- Monitor memory bandwidth (aiv_main_mem_read_bw_gbps, aiv_main_mem_write_bw_gbps) when changing header structure

## 检索线索

- foreach_mul_list
- tiling_header
- template_macros
- vector_utilization
- memory_bandwidth
- compiler_optimization
- header_refactoring
- Ascend_C
- pipeline_stalls
- code_structure
