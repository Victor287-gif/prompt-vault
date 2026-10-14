---
type: Optimization
title: "triu：冗余消除优化"
description: "triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
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

- experience ID：`e89a058c39294e94`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`remove.redundant.operations`
- canonical family：`remove.redundant.operations`
- 源记录时间：`2026-04-21T02:06:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating redundant tensor allocation and immediate free operations in Compute function
~~~~

### Original summary

~~~~text
In Ascend C operators, avoid creating temporary tensors that are immediately freed without being used. These redundant operations add memory management overhead and can cause pipeline stalls. Removing such operations improves performance by reducing vector wait ratios and increasing memory bandwidth utilization.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary tensor allocation and immediate free operations reduces memory management overhead and improves vector unit utilization by eliminating pipeline stalls caused by redundant memory operations.
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor/FreeTensor calls → reduces memory management overhead → decreases vector wait ratio from 0.7838 to 0.7401 → improves vector utilization from 0.3008 to 0.3529 → increases memory bandwidth from 52.95 to 62.39 Gbps → reduces task duration from 2361.39 to 2004.04 µs (15.1% improvement)
~~~~

## 适用条件

- When using TQue API with AllocTensor/FreeTensor operations in Compute functions
- When vector operations show high wait ratios (aiv_vec_wait_ratio_mean > 0.7)
- When memory bandwidth utilization is below expected levels
- When MTE3 utilization shows unexpected sensitivity to code changes

## 失效边界

- When tensor allocations are required for actual data processing
- When memory operations are intentionally inserted for synchronization
- When debugging or instrumentation requires additional tensor operations

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
- fast duration us：`2004.040039`
- improvement ratio：`0.15132934020337488`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 2003.44 -> 2360.81 (+17.84%)
- latency.task_duration_us: 2004.04 -> 2361.39 (+17.83%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 32 -> 27.15 (-15.16%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.45 -> 30.08 (-15.15%)
- aiv_utilization.mte3_ratio_mean: 0.769898 -> 0.653308 (-15.14%)

### Affected metric groups

- critical_path
- latency
- memory_ub
- aiv_utilization

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean: 2003.44 -> 2360.81 (+17.84%)",
  "latency.task_duration_us: 2004.04 -> 2361.39 (+17.83%)",
  "memory_ub.gm_to_ub_bw_usage_rate_pct: 32 -> 27.15 (-15.16%)",
  "memory_ub.ub_to_gm_bw_usage_rate_pct: 35.45 -> 30.08 (-15.15%)",
  "aiv_utilization.mte3_ratio_mean: 0.769898 -> 0.653308 (-15.14%)"
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

- 1. Search for AllocTensor calls followed by immediate FreeTensor in Compute functions (triu/op_kernel/triu.cpp:125-126)
- 2. Verify the allocated tensor is not used between allocation and free operations
- 3. Remove the redundant AllocTensor/FreeTensor pair to eliminate unnecessary memory management overhead
- 4. Profile the operator to confirm reduction in aiv_vec_wait_ratio_mean and improvement in memory bandwidth
- 5. Check that vector utilization (aiv_utilization.vec_ratio_mean) improves after removal

## 检索线索

- redundant tensor allocation
- memory management overhead
- AllocTensor
- FreeTensor
- TQue API
- vector wait ratio
- pipeline stalls
- memory bandwidth
- triu operator
- Compute function optimization
