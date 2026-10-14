---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
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

- experience ID：`fc2e89643de9a80a`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_011_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operation`
- canonical family：`remove.redundant.operation`
- 源记录时间：`2026-04-21T01:02:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove redundant tensor allocation and immediate free operations
~~~~

### Original summary

~~~~text
Removing unnecessary tensor allocation and immediate free operations in the KernelTriu::Compute function reduces memory management overhead and improves performance by 15.4%. This optimization applies when tensor operations allocate and free memory within the same scope without actual data usage, which occurs in Ascend C kernels using TQue APIs for tensor management.
~~~~

### Original root mechanism

~~~~text
Redundant tensor allocation and immediate free operations add unnecessary memory management overhead without affecting computation logic, impacting vector unit utilization and memory bandwidth efficiency
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor and FreeTensor calls → reduces memory management overhead → improves vector unit utilization (vec_ratio_mean increases from 0.301 to 0.355) → reduces pipeline stalls (aiv_vec_wait_ratio_mean decreases from 0.784 to 0.738) → improves memory bandwidth utilization (UB read/write bandwidth increases from 79.83 Gbps to 94.34 Gbps) → reduces task duration from 2361.39 μs to 1998.26 μs
~~~~

## 适用条件

- Ascend C kernels use TQue APIs for tensor allocation and deallocation
- Kernel functions contain temporary tensor allocations that are immediately freed
- Tensor operations allocate memory but don't actually use the allocated tensor
- Working with vector operators that have memory-intensive compute patterns

## 失效边界

- Tensor allocations are actually needed for intermediate computation results
- Memory reuse patterns require explicit allocation and deallocation
- Temporary tensors are used across multiple compute phases

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
- fast duration us：`1998.26001`
- improvement ratio：`0.15377706626154353`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1997.69 -> 2360.81 (+18.18%)
- latency.task_duration_us: 1998.26 -> 2361.39 (+18.17%)
- aiv_utilization.mte3_ratio_mean: 0.782112 -> 0.653308 (-16.47%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 32.09 -> 27.15 (-15.39%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.55 -> 30.08 (-15.39%)

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
      "before": 1997.687,
      "after": 2360.809,
      "delta": 363.1220000000003,
      "rel": 0.18177121841409605
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1998.26,
      "after": 2361.387,
      "delta": 363.1270000000002,
      "rel": 0.18172159779007746
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.782112,
      "after": 0.653308,
      "delta": -0.12880400000000003,
      "rel": -0.16468741049875213
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 32.09,
      "after": 27.15,
      "delta": -4.940000000000005,
      "rel": -0.1539420380180743
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 35.55,
      "after": 30.08,
      "delta": -5.469999999999999,
      "rel": -0.15386779184247537
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
- Look for patterns where tensors are allocated and freed in the same scope without data operations
- Check if the allocated tensor (e.g., redundantLocal) is used before being freed
- Remove the allocation and immediate free operations to reduce memory management overhead
- Verify performance improvement through vector unit utilization metrics (vec_ratio_mean) and memory bandwidth (ub_read_bw_vector_gbps)

## 检索线索

- triu
- math
- Ascend C
- TQue
- AllocTensor
- FreeTensor
- memory management
- redundant operation
- vector unit
- memory bandwidth
- KernelTriu
- Compute function
