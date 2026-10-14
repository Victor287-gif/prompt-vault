---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注内存管理开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
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

- experience ID：`dcf64914cfdc517a`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_015_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T01:46:10`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor allocation and immediate free operations
~~~~

### Original summary

~~~~text
Removing unnecessary tensor allocations and immediate frees in the Compute function reduces memory management overhead, improving vector unit utilization by 5.4 percentage points and decreasing pipeline stalls. This optimization is particularly effective for Ascend C operators with tight compute loops where memory operations compete with arithmetic operations for pipeline resources.
~~~~

### Original root mechanism

~~~~text
Redundant tensor allocation and immediate free operations introduce unnecessary memory management overhead that reduces vector unit utilization and increases pipeline stalls
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor/FreeTensor calls → reduces memory management overhead → decreases aiv_vec_wait_ratio_mean from 0.7838 to 0.7420 → improves vec_ratio_mean from 0.3008 to 0.3547 → reduces task_duration_us from 2361.39 to 1980.28
~~~~

## 适用条件

- Ascend C operators using TQue queue APIs with AllocTensor and FreeTensor calls
- Operators with tight compute loops where memory operations compete with arithmetic operations
- Vector operators with aiv_vec_wait_ratio_mean > 0.7 indicating pipeline stalls
- Operators showing low vec_ratio_mean (<0.35) despite sufficient arithmetic intensity

## 失效边界

- When tensor allocations are necessary for actual data storage or intermediate results
- When memory operations are intentionally overlapped with computation for hiding latency
- In scalar-dominated operators where vector unit utilization is not the bottleneck

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
- fast duration us：`1980.279663`
- improvement ratio：`0.16139138167186665`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1979.69 -> 2360.81 (+19.25%)
- latency.task_duration_us: 1980.28 -> 2361.39 (+19.25%)
- aiv_utilization.mte3_ratio_mean: 0.789751 -> 0.653308 (-17.28%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.88 -> 30.08 (-16.16%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 32.38 -> 27.15 (-16.15%)

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
      "before": 1979.694,
      "after": 2360.809,
      "delta": 381.11500000000024,
      "rel": 0.19251207509847493
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1980.28,
      "after": 2361.387,
      "delta": 381.1070000000002,
      "rel": 0.19245106752580454
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.789751,
      "after": 0.653308,
      "delta": -0.13644299999999998,
      "rel": -0.17276711267222197
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.88,
      "after": 30.08,
      "delta": -5.800000000000004,
      "rel": -0.16164994425864002
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 32.38,
      "after": 27.15,
      "delta": -5.230000000000004,
      "rel": -0.16151945645460172
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

- 1. Search for AllocTensor calls followed by FreeTensor in the same scope without intervening usage
- 2. In triu/op_kernel/triu.cpp, examine the KernelTriu::Compute function for redundant tensor allocations
- 3. Remove lines 124-126 that allocate and immediately free redundantLocal tensor
- 4. Verify that vec_ratio_mean improves by at least 0.05 and aiv_vec_wait_ratio_mean decreases
- 5. Check that memory bandwidth metrics (ub_read_bw_vector_gbps, ub_write_bw_vector_gbps) show corresponding improvements

## 检索线索

- tensor_allocation
- memory_management
- redundant_operations
- vector_utilization
- pipeline_stalls
- ascend_c
- tque_api
- alloc_tensor
- free_tensor
- triu
