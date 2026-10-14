---
type: Optimization
title: "cos：并行切核优化"
description: "cos 算子的并行切核优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/docs/Cos.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/framework/tf_plugin/tensorflow_cos_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/op_host/cos.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/op_host/cos_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/op_kernel/cos.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/Cos_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/st/test_cos.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/cos/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# cos：并行切核优化

## 来源信息

- experience ID：`9fb52b3e09727cd8`
- 源 schema version：`v2`
- 算子：`cos`
- source iteration：`round_003_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`core.parallelization.enable`
- canonical family：`core.parallelization.disable`
- 源记录时间：`2026-04-20T14:20:58`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable multi-core parallel execution for vector operations
~~~~

### Original summary

~~~~text
Applying multi-core parallelization to vector operations significantly reduces latency by distributing workload across available cores. For the Cos operator, enabling parallel execution reduced task duration from 184,388µs to 3,847µs (47.9x speedup). This optimization is most effective when processing large data volumes where per-core computation time dominates pipeline stalls.
~~~~

### Original root mechanism

~~~~text
Parallel execution across multiple cores reduces per-core workload and eliminates pipeline stalls from sequential execution bottlenecks.
~~~~

### Original causal chain

~~~~text
Enabling coreNum calculation based on data size → Increases block_count from 1 to 48 → Reduces per-core vector operations from 12.1B to 253M → Decreases pipeline stall ratio from 0.999722 to 0.998977 → Lowers task duration by 97.9%
~~~~

## 适用条件

- Vector operators with large input data volumes (>10MB)
- Operators using TilingFunc with coreNum parameter
- When block_count is 1 but data size suggests parallelization would be beneficial
- When aiv_mte3_wait_ratio_mean > 0.99 indicates pipeline stall bottlenecks

## 失效边界

- Small data volumes where parallelization overhead exceeds benefits
- When memory bandwidth is already saturated
- When coreNum calculation would result in <2 blocks per core
- Operators with data dependencies preventing parallel execution

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- cos/op_host/cos.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch
- ub

### 原始 diff

- 源 diff SHA-256：`3eee8975b61bf60d35b90911c333f0c00bfdf9a8ac3f876ab55f146c3a55db47`

````diff
--- Before/op_host/cos.cpp
+++ After/op_host/cos.cpp
@@ -50,15 +50,8 @@
     // Input data for 32B alignment
     uint64_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

-    if(ubPartDataNum >= inputDataNum)
-    {
-        coreNum=1;
-    }
-    else
-    {
-        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
-        coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
-    }
+    // Force sequential execution on single core for degradation
+    coreNum = 1;

     uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
     uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`184388.296875`
- fast duration us：`3846.857178`
- improvement ratio：`0.9791371944792795`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 11008.2 -> 528395 (+4700.01%)
- arithmetic.aiv_vec_fops: 2.52697e+08 -> 1.21294e+10 (+4699.97%)
- memory_ub.gm_to_ub_kb: 11008.4 -> 528395 (+4699.92%)
- critical_path.aiv_vector0_time_us_mean: 3844.74 -> 184384 (+4695.75%)
- latency.task_duration_us: 3846.86 -> 184388 (+4693.22%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_kb",
  "arithmetic.aiv_vec_fops",
  "memory_ub.gm_to_ub_kb",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "parallelism.active_block_count",
  "parallelism.block_count",
  "aiv_utilization.mte2_ratio_mean"
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

- Check TilingFunc implementation in op_host files for coreNum calculation logic
- Ensure coreNum = max(1, min(available_cores, inputLengthAlign32 / BLOCK_SIZE))
- Verify block_count increases proportionally to coreNum in performance metrics
- Monitor aiv_mte3_wait_ratio_mean reduction as parallelization reduces pipeline stalls
- Validate that memory_ub.gm_to_ub_kb and ub_to_gm_kb increase proportionally with block_count

## 检索线索

- parallelization
- coreNum
- block_count
- tiling
- vector_operations
- pipeline_stall
- multi_core
- Ascend_C
- performance_optimization
- TilingFunc
