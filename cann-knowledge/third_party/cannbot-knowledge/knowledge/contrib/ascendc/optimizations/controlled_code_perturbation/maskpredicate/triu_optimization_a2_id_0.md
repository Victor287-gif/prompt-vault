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

- experience ID：`196bff315bd9cfac`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operation`
- canonical family：`remove.redundant.operation`
- 源记录时间：`2026-04-21T02:32:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removed redundant tensor allocation and immediate free in Compute function
~~~~

### Original summary

~~~~text
Eliminating redundant tensor allocation and immediate free operations in Ascend C kernel Compute functions can improve performance by 15.6%. This optimization reduces memory management overhead and improves memory bandwidth utilization, particularly in vector operations. Apply this technique when you identify unnecessary AllocTensor/FreeTensor pairs in performance-critical loops.
~~~~

### Original root mechanism

~~~~text
Removing redundant tensor allocation and immediate free operations in the Compute function reduces memory management overhead, improving overall performance by eliminating unnecessary memory operations.
~~~~

### Original causal chain

~~~~text
Removed redundant AllocTensor/FreeTensor pair -> Reduced memory management overhead -> Improved vector unit utilization from 30.08% to 35.58% -> Reduced MTE3 wait ratio from 99.55% to 99.26% -> Decreased task duration from 2361.39μs to 1992.80μs
~~~~

## 适用条件

- Ascend C kernels use TQue API for tensor allocation and deallocation
- Compute functions contain redundant AllocTensor/FreeTensor pairs
- Performance profiling shows high memory management overhead
- Memory bandwidth utilization metrics indicate room for improvement
- Vector unit utilization is below optimal levels

## 失效边界

- Tensor allocation is necessary for actual data storage
- Memory operations are required for functional correctness
- AllocTensor/FreeTensor pairs serve a legitimate purpose in the algorithm
- Performance profiling shows no memory-related bottlenecks

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
- fast duration us：`1992.799805`
- improvement ratio：`0.15608935328665058`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1992.2 -> 2360.81 (+18.50%)
- latency.task_duration_us: 1992.8 -> 2361.39 (+18.50%)
- aiv_utilization.mte3_ratio_mean: 0.795048 -> 0.653308 (-17.83%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 32.18 -> 27.15 (-15.63%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.65 -> 30.08 (-15.62%)

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
      "before": 1992.198,
      "after": 2360.809,
      "delta": 368.6110000000001,
      "rel": 0.1850272914640011
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1992.8,
      "after": 2361.387,
      "delta": 368.5870000000002,
      "rel": 0.18495935367322372
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.795048,
      "after": 0.653308,
      "delta": -0.14173999999999998,
      "rel": -0.17827854418852696
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 32.18,
      "after": 27.15,
      "delta": -5.030000000000001,
      "rel": -0.15630826600372907
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.65,
      "after": 30.08,
      "delta": -5.57,
      "rel": -0.15624123422159888
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

- Review Compute function in triu.cpp for redundant tensor allocations
- Identify AllocTensor calls followed immediately by FreeTensor without data usage
- Remove the redundant allocation pair: 'auto redundantLocal = outQueueY.AllocTensor<DTYPE_X>(); outQueueY.FreeTensor(redundantLocal);'
- Verify functional correctness after removal
- Profile performance to confirm memory bandwidth improvement and reduced latency

## 检索线索

- triu
- math
- redundant
- allocation
- free
- memory
- overhead
- AllocTensor
- FreeTensor
- TQue
- vector
- bandwidth
- utilization
- MTE3
