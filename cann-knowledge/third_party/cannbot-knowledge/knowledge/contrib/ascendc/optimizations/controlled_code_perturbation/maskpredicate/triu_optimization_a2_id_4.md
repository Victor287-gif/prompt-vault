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

- experience ID：`aeef05aee30ee3df`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_018_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.ops`
- canonical family：`remove.redundant.ops`
- 源记录时间：`2026-04-21T02:19:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Removing redundant tensor allocation and immediate free in Compute function
~~~~

### Original summary

~~~~text
In Ascend C operators, avoid unnecessary tensor allocations followed by immediate frees within the Compute function. These redundant operations introduce memory management overhead that impacts performance metrics including vector utilization, memory bandwidth usage, and pipeline efficiency. Apply this optimization when reviewing operator kernels for redundant memory operations that don't contribute to computation.
~~~~

### Original root mechanism

~~~~text
The redundant allocation and immediate free of 'redundantLocal' tensor does not affect the computation results and only adds unnecessary memory management overhead
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor/FreeTensor pair -> Reduced memory management overhead -> Improved vector utilization (vec_ratio_mean: 0.3008 -> 0.3548) and memory bandwidth efficiency -> 18.7% latency reduction (2361.39us -> 1989.14us)
~~~~

## 适用条件

- Ascend C operator has unnecessary AllocTensor/FreeTensor pairs in Compute function
- Tensor allocation is immediately followed by FreeTensor without actual usage
- Memory bandwidth utilization shows inefficiencies (gm_to_ub_bw_usage_rate_pct < 35%, ub_to_gm_bw_usage_rate_pct < 35%)
- Vector utilization is lower than expected (vec_ratio_mean < 0.35)

## 失效边界

- Tensor allocation is needed for intermediate computation results
- Memory operations are required for data movement between different memory hierarchies
- Allocation serves a purpose in the computation flow

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
- fast duration us：`1989.139771`
- improvement ratio：`0.15763930409063154`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1988.6 -> 2360.81 (+18.72%)
- latency.task_duration_us: 1989.14 -> 2361.39 (+18.71%)
- aiv_utilization.mte3_ratio_mean: 0.781085 -> 0.653308 (-16.36%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.72 -> 30.08 (-15.79%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 32.24 -> 27.15 (-15.79%)

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
      "before": 1988.596,
      "after": 2360.809,
      "delta": 372.2130000000002,
      "rel": 0.1871737648069292
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1989.14,
      "after": 2361.387,
      "delta": 372.24700000000007,
      "rel": 0.18713966839940882
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.781085,
      "after": 0.653308,
      "delta": -0.12777700000000003,
      "rel": -0.16358911002003626
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.72,
      "after": 30.08,
      "delta": -5.640000000000001,
      "rel": -0.15789473684210528
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 32.24,
      "after": 27.15,
      "delta": -5.090000000000003,
      "rel": -0.15787841191067006
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

- Examine the Compute function in triu/op_kernel/triu.cpp for redundant tensor allocations
- Look for patterns where AllocTensor is called but the tensor is not used before FreeTensor
- Remove the redundant allocation and free operations to reduce memory management overhead
- Verify that vector utilization (vec_ratio_mean) improves after removal
- Check that memory bandwidth metrics (gm_to_ub_bw_usage_rate_pct, ub_to_gm_bw_usage_rate_pct) show improved efficiency

## 检索线索

- triu
- Ascend C
- memory management
- redundant allocation
- AllocTensor
- FreeTensor
- vector utilization
- memory bandwidth
- Compute function
- op_kernel
