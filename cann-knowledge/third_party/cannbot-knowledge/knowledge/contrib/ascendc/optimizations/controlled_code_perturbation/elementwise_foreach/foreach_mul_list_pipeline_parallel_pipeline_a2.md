---
type: Optimization
title: "foreach_mul_list：流水与并行优化"
description: "foreach_mul_list 算子的流水与并行优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/docs/aclnnForeachMulList.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_host/foreach_mul_list.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_kernel/foreach_mul_list.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_mul_list：流水与并行优化

## 来源信息

- experience ID：`a4467525bdce0cf0`
- 源 schema version：`v2`
- 算子：`foreach_mul_list`
- source iteration：`round_016_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism`
- canonical family：`untitled`
- 源记录时间：`2026-04-15T02:34:46`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove artificial data dependencies between independent Cast operations
~~~~

### Original summary

~~~~text
When performing consecutive Cast operations on independent data in Ascend C, avoid introducing artificial data dependencies between the operations. The original code inserted a redundant operation `float32Tensor[0] = float32Tensor[0] + 0.0f;` between two Cast calls, creating a false dependency that forced sequential execution. Removing this dependency allows the Cast operations to execute in parallel, reducing pipeline stalls by up to 95.59% for MTE2 wait and 52.14% for MTE3 wait.
~~~~

### Original root mechanism

~~~~text
Artificial data dependencies between independent Cast operations prevent pipeline parallelism, causing sequential execution and increased pipeline stalls.
~~~~

### Original causal chain

~~~~text
Removing artificial dependency between Cast operations → Reduced pipeline stall ratios (MTE2 wait: -95.59%, MTE3 wait: -52.14%) → Improved vector unit utilization (+31.6%) → Reduced overall latency by 12.35%
~~~~

## 适用条件

- Consecutive Cast operations on independent data buffers in vector kernels
- Operations separated by PipeBarrier<PIPE_V>() calls
- ForeachMulList operators with similar memory access patterns
- When pipeline_stalls.aiv_mte2_wait_ratio_mean or aiv_mte3_wait_ratio_mean are high (>0.01)

## 失效边界

- When there are actual data dependencies between operations
- When operations must execute sequentially for correctness
- When using scalar operations that cannot be pipelined

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/foreach_mul_list.cpp

### 代码签名

- PipeBarrier
- pipebarrier

### 原始 diff

- 源 diff SHA-256：`855c1a60481a536696fa6b672d5d30da0c826cb8d1531c36289e0105c8ccfe1c`

````diff
--- Before/op_kernel/foreach_mul_list.cpp
+++ After/op_kernel/foreach_mul_list.cpp
@@ -344,6 +344,9 @@
     {
         PipeBarrier<PIPE_V>();
         Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
+        // Artificial data dependency: use first cast result before second cast
+        // This forces sequential execution and breaks pipeline parallelism
+        float32Tensor[0] = float32Tensor[0] + 0.0f;  // Redundant operation creating dependency
         PipeBarrier<PIPE_V>();
         Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
         PipeBarrier<PIPE_V>();
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`16577.692383`
- fast duration us：`14530.310547000001`
- improvement ratio：`0.12350222146114487`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_mte2_wait_ratio_mean: 6.8e-05 -> 3e-06 (-95.59%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.022798 -> 0.010911 (-52.14%)
- critical_path.aiv_vector0_time_us_mean: 6422.3 -> 8476.22 (+31.98%)
- latency.task_duration_us: 6446.27 -> 8496.77 (+31.81%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 3.7 -> 2.8 (-24.32%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- memory_ub

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "aiv_utilization.vec_ratio_mean",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "memory_gm.aiv_main_mem_write_bw_gbps"
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

- Inspect code between PipeBarrier<PIPE_V>() calls for artificial dependencies like float32Tensor[0] = float32Tensor[0] + 0.0f
- Ensure independent Cast operations on different memory regions have no false dependencies
- Verify that consecutive memory operations use different tensor indices or offsets
- Check for redundant operations that read and write to same memory location between independent operations

## 检索线索

- pipeline_stalls
- Cast
- PipeBarrier
- data_dependency
- parallelism
- foreach_mul_list
- vector_kernel
- MTE2_wait
- MTE3_wait
- artificial_dependency
