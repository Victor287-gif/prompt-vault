---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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

- experience ID：`58d5d414005d05f1`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_010_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operation`
- canonical family：`remove.redundant.operation`
- 源记录时间：`2026-04-21T00:53:02`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating redundant tensor allocation and immediate free in Ascend C Compute functions
~~~~

### Original summary

~~~~text
Removing unnecessary tensor allocation and immediate free operations in Ascend C Compute functions reduces memory management overhead and improves overall performance. This optimization is particularly effective when tensor operations are performed within tight loops or high-frequency execution paths. The improvement comes from eliminating redundant queue API calls that consume compute cycles without contributing to actual computation.
~~~~

### Original root mechanism

~~~~text
Redundant tensor allocation and immediate free operations in Ascend C Compute functions introduce unnecessary memory management overhead that impacts pipeline efficiency and resource utilization
~~~~

### Original causal chain

~~~~text
Removing redundant tensor allocation and free -> reduced memory management overhead -> decreased aiv_vector0_time_us_mean from 2360.81μs to 1999.36μs (-15.3%) -> reduced task_duration_us from 2361.39μs to 1999.86μs (-15.3%)
~~~~

## 适用条件

- Ascend C operators using TQue::AllocTensor() and TQue::FreeTensor() API calls
- Compute functions with tensor allocations that are immediately freed without being used
- Operators with vector processing blocks where memory management overhead impacts performance
- Code patterns where temporary tensors are allocated but not utilized in computation

## 失效边界

- Tensor allocations that are actually used for intermediate computation results
- Memory management patterns that are required for correct operator behavior
- When tensor lifetime management is necessary for data flow between pipeline stages

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
- fast duration us：`1999.859985`
- improvement ratio：`0.15309950902092787`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1999.36 -> 2360.81 (+18.08%)
- latency.task_duration_us: 1999.86 -> 2361.39 (+18.08%)
- aiv_utilization.mte3_ratio_mean: 0.772612 -> 0.653308 (-15.44%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.52 -> 30.08 (-15.32%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 32.06 -> 27.15 (-15.32%)

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
      "before": 1999.356,
      "after": 2360.809,
      "delta": 361.4530000000002,
      "rel": 0.18078471267748225
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1999.86,
      "after": 2361.387,
      "delta": 361.52700000000027,
      "rel": 0.1807761543308033
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.772612,
      "after": 0.653308,
      "delta": -0.11930399999999997,
      "rel": -0.15441644706527982
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.52,
      "after": 30.08,
      "delta": -5.440000000000005,
      "rel": -0.15315315315315328
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 32.06,
      "after": 27.15,
      "delta": -4.910000000000004,
      "rel": -0.15315034310667508
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

- Review Compute functions in triu/op_kernel/triu.cpp for redundant AllocTensor/FreeTensor pairs
- Identify tensor allocations that are immediately freed without being used (e.g., auto redundantLocal = outQueueY.AllocTensor<DTYPE_X>(); outQueueY.FreeTensor(redundantLocal);)
- Remove unnecessary tensor management operations to reduce memory overhead
- Verify that removal doesn't affect actual computation by checking tensor usage patterns
- Test performance impact by measuring aiv_vector0_time_us_mean and task_duration_us metrics

## 检索线索

- triu
- ascendc
- tensor_allocation
- memory_management
- redundant_operation
- tque
- alloctensor
- freetensor
- compute_function
- performance_sensitivity
