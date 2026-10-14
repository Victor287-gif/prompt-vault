---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.scalar"]
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

- experience ID：`76091055626b0a33`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_008_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T00:27:46`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor allocation and immediate free operations in Compute function
~~~~

### Original summary

~~~~text
Avoid unnecessary tensor allocation and immediate free operations within the Compute function's main loop. These operations introduce scalar instruction overhead that reduces vector unit utilization and memory bandwidth efficiency. Apply this optimization when profiling shows high aiv_vec_wait_ratio_mean and low memory bandwidth usage despite sufficient computational work.
~~~~

### Original root mechanism

~~~~text
Removing redundant AllocTensor/FreeTensor pairs eliminates unnecessary memory management overhead that introduces scalar instruction dependencies, reducing vector wait ratios and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor/FreeTensor calls reduces scalar instruction count → decreases aiv_vec_wait_ratio_mean from 0.725 to 0.621 → increases vector unit utilization from 0.300 to 0.438 → improves memory bandwidth utilization (GM read from 57.54 to 85.82 Gbps, UB read from 86.75 to 129.38 Gbps) → reduces task duration from 2173 to 1457 μs.
~~~~

## 适用条件

- Profiling shows high aiv_vec_wait_ratio_mean (>0.7) with low memory bandwidth utilization
- Operator uses TQue API with AllocTensor/FreeTensor calls in Compute function loops
- Vector unit utilization (aiv_utilization.vec_ratio_mean) is below 0.4 despite sufficient FLOPs
- Memory bandwidth usage (memory_gm.aiv_main_mem_read_bw_gbps) is below theoretical maximum

## 失效边界

- AllocTensor/FreeTensor operations are required for actual tensor lifecycle management
- Memory operations are already optimized and bandwidth utilization is near maximum
- Vector unit utilization is already high (>0.8) with low wait ratios

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

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2173.043457`
- fast duration us：`1457.209106`
- improvement ratio：`0.32941557091004836`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1456.66 -> 2172.43 (+49.14%)
- latency.task_duration_us: 1457.21 -> 2173.04 (+49.12%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 48.76 -> 32.69 (-32.96%)
- memory_gm.aiv_main_mem_read_bw_gbps: 85.82 -> 57.54 (-32.95%)
- memory_ub.ub_read_bw_vector_gbps: 129.38 -> 86.75 (-32.95%)

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
      "before": 1456.662,
      "after": 2172.428,
      "delta": 715.7659999999998,
      "rel": 0.4913741142420135
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1457.209,
      "after": 2173.043,
      "delta": 715.8340000000001,
      "rel": 0.49123632917446985
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 48.76,
      "after": 32.69,
      "delta": -16.07,
      "rel": -0.32957342083675145
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 85.82,
      "after": 57.54,
      "delta": -28.279999999999994,
      "rel": -0.3295269168026101
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 129.38,
      "after": 86.75,
      "delta": -42.629999999999995,
      "rel": -0.3294945122893801
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

- Inspect KernelTriu::Compute function in triu/op_kernel/triu.cpp for redundant AllocTensor/FreeTensor pairs
- Remove any AllocTensor calls that are immediately followed by FreeTensor without actual data usage
- Verify that tensor allocations are only performed when needed for actual data storage
- Profile after changes to confirm improvement in aiv_vec_wait_ratio_mean and memory bandwidth metrics

## 检索线索

- triu
- math
- tensor_allocation
- memory_management
- scalar_overhead
- vector_wait
- bandwidth_utilization
- TQue
- AllocTensor
- FreeTensor
- Compute_function
