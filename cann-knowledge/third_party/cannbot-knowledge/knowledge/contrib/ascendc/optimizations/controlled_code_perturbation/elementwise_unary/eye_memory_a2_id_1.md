---
type: Optimization
title: "eye：搬运与内存优化"
description: "eye 算子的搬运与内存优化经验，关注数据搬运。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/docs/Eye.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/aclnn_eye_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/aclnn_eye_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_host/eye.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_host/eye_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_kernel/eye.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/Eye_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/test_eye.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# eye：搬运与内存优化

## 来源信息

- experience ID：`51d0b3904dca45cd`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_002_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`memory.access_pattern`
- canonical family：`untitled`
- 源记录时间：`2026-04-20T14:10:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
MTE2 utilization reduction through scalar operation optimization
~~~~

### Original summary

~~~~text
Optimizing scalar memory access patterns in the eye operator's diagonal initialization loop reduces MTE2 utilization by 66.41%, improving overall performance by 5.03%. This optimization is particularly effective for operators with scattered memory writes to diagonal elements where scalar unit efficiency impacts memory transfer engine overhead.
~~~~

### Original root mechanism

~~~~text
Reducing MTE2 memory transfer engine utilization by optimizing scalar memory access patterns in diagonal element initialization reduces pipeline stalls and improves overall throughput.
~~~~

### Original causal chain

~~~~text
Optimizing scalar memory access patterns in eye.cpp diagonal initialization → MTE2 ratio reduced from 0.000259 to 8.7e-05 (-66.41%) → ICache miss rate increased from 9.1e-05 to 0.000103 (+13.19%) → Main memory bandwidth reduced from 4.31 to 4.09 Gbps (-5.10%) → Task duration reduced from 1193.78 to 1133.76 µs (-5.03%)
~~~~

## 适用条件

- Operator performs scattered memory writes to diagonal elements
- Operator uses scalar operations for element-wise initialization
- MTE2 utilization is above 0.0002 in performance profile
- Memory bandwidth usage is above 4 Gbps for small matrices
- Operator has nested loops with conditional diagonal element access

## 失效边界

- Operator uses vectorized memory operations exclusively
- Memory access patterns are already coalesced and optimized
- MTE2 utilization is already minimal (< 0.0001)
- Operator performance is compute-bound rather than memory-bound

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异


## 观测记录

- bottleneck type：`memory_transfer`
- slow duration us：`1193.783936`
- fast duration us：`1133.762695`
- improvement ratio：`0.050278144302320506`
- machine confidence：`0.85`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.000259 -> 8.7e-05 (-66.41%)
- aiv_utilization.icache_miss_rate_mean: 9.1e-05 -> 0.000103 (+13.19%)
- critical_path.aiv_vector0_time_us_mean: 1133.16 -> 1193.17 (+5.30%)
- latency.task_duration_us: 1133.76 -> 1193.78 (+5.29%)
- memory_gm.aiv_main_mem_read_bw_gbps: 4.31 -> 4.09 (-5.10%)

### Affected metric groups

- aiv_utilization
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.000259,
      "after": 8.7e-05,
      "delta": -0.000172,
      "rel": -0.6640926640926641
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 9.1e-05,
      "after": 0.000103,
      "delta": 1.1999999999999994e-05,
      "rel": 0.1318681318681318
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1133.165,
      "after": 1193.171,
      "delta": 60.006000000000085,
      "rel": 0.05295433586459173
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1133.763,
      "after": 1193.784,
      "delta": 60.021000000000186,
      "rel": 0.05293963553229395
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 4.31,
      "after": 4.09,
      "delta": -0.21999999999999975,
      "rel": -0.051044083526682084
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

- Analyze MTE2 ratio in performance profile - if > 0.0002, consider scalar access optimization
- Examine eye.cpp diagonal initialization loops for scattered memory writes
- Optimize scalar memory access patterns by reducing conditional checks in inner loops
- Consider precomputing diagonal indices to reduce runtime calculations
- Test with matrix sizes from 32x32 to 1024x1024 to validate MTE2 reduction

## 检索线索

- eye
- diagonal
- scalar
- MTE2
- memory_transfer
- scattered_access
- ascendc
- memory_bandwidth
