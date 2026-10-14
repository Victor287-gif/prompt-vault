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

- experience ID：`bf01ef6826de9dc5`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_013_refactor`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`dead.code.elimination`
- canonical family：`dead.code.elimination`
- 源记录时间：`2026-04-21T01:27:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor allocation and immediate free in vector kernels
~~~~

### Original summary

~~~~text
Removing semantically redundant tensor allocation and immediate free operations in Ascend C vector kernels improves performance by 15.7%. These operations, while logically unnecessary, create memory management overhead that reduces vector unit utilization from 35.6% to 30.1% and decreases memory bandwidth usage by 15.7%. Apply this optimization when tensor allocation/free operations appear in Compute functions but don't contribute to the actual computation.
~~~~

### Original root mechanism

~~~~text
Redundant tensor allocation and immediate free operations in vector kernels create unnecessary memory management overhead that reduces vector unit utilization and memory bandwidth efficiency
~~~~

### Original causal chain

~~~~text
structural change to dead.code.elimination in triu/op_kernel/triu.cpp -> unexpected shift in critical_path.aiv_vector0_time_us_mean -> 15.7% latency impact
~~~~

## 适用条件

- Vector kernels using TQue APIs (AllocTensor/FreeTensor)
- Compute functions with semantically redundant tensor allocations
- Kernels showing low vector unit utilization (<40%)
- Operations with unnecessary memory management overhead

## 失效边界

- Tensor allocations are actually used in subsequent computations
- Memory operations are required for data movement patterns
- Kernel already has optimal vector utilization (>80%)

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
- fast duration us：`1991.059814`
- improvement ratio：`0.1568262044878607`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1990.52 -> 2360.81 (+18.60%)
- latency.task_duration_us: 1991.06 -> 2361.39 (+18.60%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 35.68 -> 30.08 (-15.70%)
- memory_gm.aiv_main_mem_read_bw_gbps: 62.8 -> 52.95 (-15.68%)
- memory_gm.aiv_main_mem_write_bw_gbps: 62.8 -> 52.95 (-15.68%)

### Affected metric groups

- critical_path
- latency
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean: 1990.52 → 2360.81 (+18.60%)",
  "latency.task_duration_us: 1991.06 → 2361.39 (+18.60%)",
  "memory_ub.ub_to_gm_bw_usage_rate_pct: 35.68 → 30.08 (-15.70%)",
  "memory_gm.aiv_main_mem_read_bw_gbps: 62.8 → 52.95 (-15.68%)",
  "memory_gm.aiv_main_mem_write_bw_gbps: 62.8 → 52.95 (-15.68%)"
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

- 1. Search for AllocTensor/FreeTensor pairs in Compute functions in op_kernel/ files
- 2. Verify the allocated tensor is not used between allocation and free
- 3. Remove the redundant allocation/free pair from KernelTriu::Compute or similar functions
- 4. Test performance impact by measuring vector unit utilization and memory bandwidth
- 5. Apply to other vector kernels showing similar patterns of low vector utilization

## 检索线索

- dead.code.elimination
- tensor.allocation
- memory.management
- vector.utilization
- bandwidth.efficiency
- TQue
- AllocTensor
- FreeTensor
- Compute.function
- triu
- math.kernel
