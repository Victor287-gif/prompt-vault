---
type: Optimization
title: "triu：流水与并行优化"
description: "triu 算子的流水与并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model"]
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
# triu：流水与并行优化

## 来源信息

- experience ID：`d597979baccd04dc`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism.single.buffer`
- canonical family：`pipeline.parallelism.single.buffer`
- 源记录时间：`2026-04-18T22:12:01`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce double buffering to single buffering
~~~~

### Original summary

~~~~text
Using single buffering (BUFFER_NUM=1) instead of double buffering improves performance for memory-bound vector operators by reducing pipeline synchronization overhead. This optimization is most effective when the operator has sufficient compute intensity to hide memory latency without pipelining, and when scalar overhead is the primary bottleneck rather than memory bandwidth.
~~~~

### Original root mechanism

~~~~text
Setting BUFFER_NUM to 1 eliminates pipeline bubbles caused by overlapping CopyIn, Compute, and CopyOut operations, allowing more efficient sequential execution.
~~~~

### Original causal chain

~~~~text
BUFFER_NUM changed from 2 to 1 → pipeline overlap eliminated → scalar_ratio_mean increased from 0.269 to 0.563 → vector unit utilization increased from 0.227 to 0.436 → memory bandwidth utilization doubled from ~45 Gbps to ~86 Gbps → task duration reduced from 2792μs to 1449μs
~~~~

## 适用条件

- Operator is memory-bound with scalar_ratio_mean < 0.3
- Memory bandwidth utilization is below 50% of peak
- Vector unit utilization (vec_ratio_mean) is below 0.3
- Operator uses AscendC pipeline with BUFFER_NUM > 1
- aiv_vec_wait_ratio_mean > 0.6 indicating pipeline stalls

## 失效边界

- Operator is compute-bound with high arithmetic intensity
- Memory bandwidth is already saturated (>80% utilization)
- Multiple blocks can execute in parallel with sufficient parallelism
- Operator requires overlapping memory transfers with computation to hide latency

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/triu.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- ub

### 原始 diff

- 源 diff SHA-256：`b228764570893b4a6b04f7bd1cb689461529b0e96d1f3843ec56c2941f40700b`

````diff
--- Before/op_kernel/triu.cpp
+++ After/op_kernel/triu.cpp
@@ -1,6 +1,6 @@
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
+constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue
 constexpr int32_t minNum = 1;
 constexpr int keyFour = 4;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2792.355957`
- fast duration us：`1449.588989`
- improvement ratio：`0.4808724205214214`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1448.99 -> 2791.74 (+92.67%)
- latency.task_duration_us: 1449.59 -> 2792.36 (+92.63%)
- aiv_utilization.scalar_ratio_mean: 0.562556 -> 0.268977 (-52.19%)
- memory_gm.aiv_main_mem_write_bw_gbps: 86.27 -> 44.77 (-48.10%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 49.02 -> 25.44 (-48.10%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- memory_gm

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "aiv_utilization.scalar_ratio_mean",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "memory_ub.ub_to_gm_bw_usage_rate_pct"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 1,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- In op_kernel/triu.cpp, change BUFFER_NUM from 2 to 1
- Verify scalar_ratio_mean increases and vec_ratio_mean improves
- Check that memory bandwidth utilization (aiv_main_mem_read_bw_gbps, aiv_main_mem_write_bw_gbps) increases proportionally
- Monitor aiv_vec_wait_ratio_mean reduction indicating fewer pipeline stalls
- Ensure task_duration_us decreases by approximately 2x for memory-bound operations

## 检索线索

- BUFFER_NUM
- single_buffer
- double_buffer
- pipeline_parallelism
- scalar_overhead
- memory_bound
- vector_operator
- triu
- AscendC
- pipeline_stalls
