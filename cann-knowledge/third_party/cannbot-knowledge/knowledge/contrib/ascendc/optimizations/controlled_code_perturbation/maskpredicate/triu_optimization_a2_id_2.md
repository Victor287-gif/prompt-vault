---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注内存管理开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`6123d949e3c9925b`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_014_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T01:39:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant tensor allocation and immediate free operations
~~~~

### Original summary

~~~~text
In Ascend C operators, avoid creating and immediately freeing tensors within the compute loop when they serve no functional purpose. These redundant memory operations introduce unnecessary overhead that reduces vector unit utilization from 35.3% to 30.1% and increases task duration by 17.4%. Apply this optimization when reviewing operator kernels for unnecessary temporary allocations that don't contribute to the computation.
~~~~

### Original root mechanism

~~~~text
Unnecessary tensor allocation and immediate deallocation in the compute loop adds memory management overhead and reduces vector unit utilization, even when the tensor serves no functional purpose.
~~~~

### Original causal chain

~~~~text
Removing redundant 'redundantLocal' tensor allocation and immediate free → Reduced memory management overhead → Improved vector unit utilization from 30.1% to 35.3% → Reduced pipeline wait ratios → Decreased task duration from 2361.39μs to 2010.98μs
~~~~

## 适用条件

- Operator uses TQue API with AllocTensor/FreeTensor calls
- Temporary tensors are allocated and freed within compute loops
- Vector unit utilization (vec_ratio_mean) is below 35%
- Memory bandwidth usage shows overhead from unnecessary allocations

## 失效边界

- Temporary tensors are actually needed for intermediate computation
- Memory allocations serve a functional purpose like data transformation
- Operator already has optimal vector unit utilization (>40%)

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

- bottleneck type：`memory_management_overhead`
- slow duration us：`2361.387207`
- fast duration us：`2010.980225`
- improvement ratio：`0.14839031098384373`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 2010.37 -> 2360.81 (+17.43%)
- latency.task_duration_us: 2010.98 -> 2361.39 (+17.42%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 31.89 -> 27.15 (-14.86%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.33 -> 30.08 (-14.86%)
- memory_gm.aiv_main_mem_read_bw_gbps: 62.18 -> 52.95 (-14.84%)

### Affected metric groups

- critical_path
- latency
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2010.365,
      "after": 2360.809,
      "delta": 350.4440000000002,
      "rel": 0.1743185938871798
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2010.98,
      "after": 2361.387,
      "delta": 350.40700000000015,
      "rel": 0.17424688460352672
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 31.89,
      "after": 27.15,
      "delta": -4.740000000000002,
      "rel": -0.14863593603010353
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.33,
      "after": 30.08,
      "delta": -5.25,
      "rel": -0.14859892442683273
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 62.18,
      "after": 52.95,
      "delta": -9.229999999999997,
      "rel": -0.14844001286587322
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

- Review KernelTriu::Compute function in triu/op_kernel/triu.cpp for unnecessary AllocTensor calls
- Check for tensor allocations followed by immediate FreeTensor without intervening computation
- Remove redundant 'auto redundantLocal = outQueueY.AllocTensor<DTYPE_X>(); outQueueY.FreeTensor(redundantLocal);' patterns
- Verify that all remaining tensor allocations have corresponding useful computations before freeing
- Profile the operator to confirm improved vector unit utilization (vec_ratio_mean) and reduced task duration

## 检索线索

- redundant tensor allocation
- memory management overhead
- AllocTensor FreeTensor
- vector unit utilization
- triu operator
- TQue API
- memory bandwidth
- pipeline stalls
- ascend c optimization
- remove redundant operations
