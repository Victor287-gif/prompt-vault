---
type: Optimization
title: "reciprocal：并行优化"
description: "reciprocal 算子的并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/Reciprocal.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/tensorflow_reciprocal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_kernel/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/Reciprocal_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/test_reciprocal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reciprocal：并行优化

## 来源信息

- experience ID：`00064ce8b411c09c`
- 源 schema version：`v2`
- 算子：`reciprocal`
- source iteration：`round_009_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.reduction`
- canonical family：`parallelism.reduction`
- 源记录时间：`2026-04-21T00:02:50`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable multi-core parallel execution for vector operations
~~~~

### Original summary

~~~~text
For vector operations like Reciprocal, enabling parallel execution across multiple cores significantly reduces latency by distributing computation. The optimization involves configuring the core launch count based on input size and available hardware resources, moving from serial execution (1 core) to parallel execution (48 cores). This technique is particularly effective for memory-bound vector operations where data can be partitioned across cores.
~~~~

### Original root mechanism

~~~~text
Using multiple cores for parallel execution reduces latency by distributing workload across available hardware resources rather than forcing serial processing.
~~~~

### Original causal chain

~~~~text
apply parallelism.reduction in reciprocal/op_host/reciprocal.cpp -> improve arithmetic.aiv_vec_fops -> 88.3% latency improvement
~~~~

## 适用条件

- Vector operations with large input sizes (e.g., reciprocal, elementwise)
- Operations where input_length_align32 / BLOCK_SIZE suggests multiple cores could be utilized
- Memory-bound operations that benefit from parallel data movement
- When block_dim configuration can be dynamically adjusted based on input size

## 失效边界

- Small input sizes where parallel overhead exceeds benefits
- Operations with strong data dependencies between elements
- When hardware resources are limited or contended
- When memory bandwidth is already saturated with serial execution

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reciprocal/op_host/reciprocal.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`8850ce0c7d6429b461ca2683c4c078929e64bc82b11103013598d4c208f555a8`

````diff
--- Before/op_host/reciprocal.cpp
+++ After/op_host/reciprocal.cpp
@@ -39,7 +39,7 @@
     }
     else
     {
-        coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
+        coreNum = 1; // Force serial execution by using only 1 core
     }
     if(BLOCK_SIZE == 0 || coreNum == 0){
 		return ge::GRAPH_FAILED;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`13726.915039`
- fast duration us：`1602.072021`
- improvement ratio：`0.8832897255903239`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 3.64127e+07 -> 1.74781e+09 (+4699.98%)
- memory_ub.ub_to_gm_kb: 21873.9 -> 1.04994e+06 (+4699.98%)
- memory_ub.gm_to_ub_kb: 21874 -> 1.04994e+06 (+4699.96%)
- critical_path.aiv_vector0_time_us_mean: 1594.6 -> 13726.1 (+760.79%)
- latency.task_duration_us: 1602.07 -> 13726.9 (+756.82%)

### Affected metric groups

- arithmetic
- memory_ub
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "arithmetic.aiv_vec_fops: 36412736.0 -> 1747805472.0 (+4699.98%)",
  "memory_ub.ub_to_gm_kb: 21873.9 -> 1049943.1 (+4699.98%)",
  "memory_ub.gm_to_ub_kb: 21874.0 -> 1049943.2 (+4699.96%)",
  "critical_path.aiv_vector0_time_us_mean: 1594.596 -> 13726.078 (+760.79%)",
  "latency.task_duration_us: 1602.072 -> 13726.915 (+756.82%)"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 2,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- In the TilingFunc of reciprocal.cpp, calculate optimal core count as min(coreNum, inputLengthAlign32 / BLOCK_SIZE)
- Set block_dim to the calculated core count rather than forcing serial execution with coreNum = 1
- Ensure BLOCK_SIZE is properly configured for the operation type
- Verify that parallel execution doesn't cause resource contention by monitoring aiv_utilization metrics

## 检索线索

- parallelism
- coreNum
- block_dim
- vector
- reciprocal
- multi-core
- parallel execution
- TilingFunc
- inputLengthAlign32
- BLOCK_SIZE
