---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/Triu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/tensorflow_triu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_kernel/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/Triu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/test_triu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# triu：冗余消除优化

## 来源信息

- experience ID：`b96c42c42eda5f76`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_009_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T00:39:50`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant tensor allocation and immediate free in Compute function
~~~~

### Original summary

~~~~text
Removing unnecessary tensor allocation and immediate free operations in the KernelTriu::Compute function reduces memory management overhead and improves performance. This optimization is applicable when tensor queue operations (AllocTensor/FreeTensor) are used within compute loops without actual data usage. The improvement comes from eliminating redundant memory management instructions that consume compute cycles.
~~~~

### Original root mechanism

~~~~text
The redundant tensor allocation and immediate free operations served no functional purpose and only added unnecessary memory management overhead. Removing them preserves identical computation while eliminating wasteful operations.
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor and FreeTensor calls -> reduces memory management overhead -> decreases aiv_vector0_time_us_mean by 346.20us (17.18%) and task_duration_us by 346.23us (17.18%) -> improves overall latency by 14.66%
~~~~

## 适用条件

- When using TQue::AllocTensor and TQue::FreeTensor operations in compute loops
- When tensor allocations are followed immediately by FreeTensor without actual data usage
- When working with Ascend C operators that use queue-based memory management
- When optimizing KernelTriu or similar math operators with tensor queue operations

## 失效边界

- When AllocTensor/FreeTensor pairs are necessary for actual data processing
- When tensor lifecycle management requires explicit allocation and deallocation
- When the allocated tensor is used for intermediate computations before being freed

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- triu/op_kernel/triu.cpp

### 改动符号

- KernelTriu::Compute

### 代码签名

- AllocTensor
- DeQue
- FreeTensor
- TQue
- queue_api

### 原始 diff

- 源 diff SHA-256：`6b995056970a730000e54e850539915cdafdc1dfa7f4e11e607baa7d60e0311a`

````diff
--- Before/op_kernel/triu.cpp
+++ After/op_kernel/triu.cpp
@@ -121,10 +121,6 @@
         auto xLocal = inQueueX.DeQue<DTYPE_X>();
         auto yLocal = outQueueY.AllocTensor<DTYPE_X>();

-        // Add redundant tensor allocation and immediate free to increase memory management overhead
-        auto redundantLocal = outQueueY.AllocTensor<DTYPE_X>();
-        outQueueY.FreeTensor(redundantLocal);
-
         uint32_t localOffset=0;
         uint32_t currLength=initLength;
         DTYPE_X scalarZero=0;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2361.387207`
- fast duration us：`2015.160278`
- improvement ratio：`0.1466201425897706`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 2014.61 -> 2360.81 (+17.18%)
- latency.task_duration_us: 2015.16 -> 2361.39 (+17.18%)
- aiv_utilization.icache_miss_rate_mean: 9.7e-05 -> 8.2e-05 (-15.46%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 31.82 -> 27.15 (-14.68%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.25 -> 30.08 (-14.67%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2014.613,
      "after": 2360.809,
      "delta": 346.19600000000014,
      "rel": 0.17184243326137583
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2015.16,
      "after": 2361.387,
      "delta": 346.2270000000001,
      "rel": 0.17181117132138396
    }
  },
  {
    "aiv_utilization.icache_miss_rate_mean": {
      "before": 9.7e-05,
      "after": 8.2e-05,
      "delta": -1.4999999999999999e-05,
      "rel": -0.15463917525773194
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 31.82,
      "after": 27.15,
      "delta": -4.670000000000002,
      "rel": -0.14676304211187938
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.25,
      "after": 30.08,
      "delta": -5.170000000000002,
      "rel": -0.14666666666666672
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

- Identify AllocTensor/FreeTensor pairs in KernelTriu::Compute function
- Check if the allocated tensor is used between allocation and free operations
- Remove the redundant AllocTensor and FreeTensor calls when no data operations occur between them
- Verify that the removal doesn't affect functional correctness by checking tensor dependencies
- Profile the operator to confirm latency improvements in aiv_vector0_time_us_mean and task_duration_us

## 检索线索

- triu
- AllocTensor
- FreeTensor
- TQue
- queue_api
- memory_management
- redundant_operations
- tensor_allocation
- KernelTriu
- Compute
- Ascend C
- performance_sensitivity
