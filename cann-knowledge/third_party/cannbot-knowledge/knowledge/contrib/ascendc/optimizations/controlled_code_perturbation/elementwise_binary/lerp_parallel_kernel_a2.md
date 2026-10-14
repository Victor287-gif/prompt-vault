---
type: Optimization
title: "lerp：并行切核优化"
description: "lerp 算子的并行切核优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/Lerp.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/tensorflow_lerp_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_kernel/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/Lerp_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/test_lerp.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# lerp：并行切核优化

## 来源信息

- experience ID：`8b066a949a491431`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`core.parallelism.enable`
- canonical family：`core.parallelism.disable`
- 源记录时间：`2026-04-18T22:00:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable multi-core parallel execution for vector operations
~~~~

### Original summary

~~~~text
For vector operations like lerp, enabling multi-core parallelism by dynamically calculating coreNum based on data size improves performance by 94%. The optimization distributes the workload across available cores instead of forcing single-core execution, reducing task duration from 28.9ms to 1.7ms. Apply this when processing large data sets where each core can handle at least 32B of aligned data.
~~~~

### Original root mechanism

~~~~text
Properly distributing work across multiple cores reduces per-core workload and enables parallel execution, significantly reducing overall latency.
~~~~

### Original causal chain

~~~~text
Enabling multi-core parallelism -> Increases block_count from 1 to 48 -> Reduces per-core vector operations (aiv_vec_fops) from 1.89B to 39.4M -> Decreases memory transfers (gm_to_ub_kb from 1.58GB to 32.96KB) -> Reduces task_duration_us from 28.9ms to 1.7ms
~~~~

## 适用条件

- Processing large vector operations with inputLength > BLOCK_SIZE * available_cores
- When ubPartDataNum < inputDataNum indicating data can be distributed across cores
- For Ascend C operators with similar tiling logic in lerp/op_host/lerp.cpp
- When block_dim can be increased from 1 to utilize multiple cores

## 失效边界

- When input data size is smaller than BLOCK_SIZE * core_count
- When memory bandwidth is already saturated
- When scalar operations dominate vector operations
- For operations with high inter-core synchronization overhead

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- lerp/op_host/lerp.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch
- ub

### 原始 diff

- 源 diff SHA-256：`acd396c2bd83a46b71cc3dd8b82c76c41bf3edfec5da6e2b1692b5e594202952`

````diff
--- Before/op_host/lerp.cpp
+++ After/op_host/lerp.cpp
@@ -39,15 +39,20 @@
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
+    // DEGRADATION: Force single-core execution regardless of data size
+    // This eliminates multi-core parallelism, increasing execution time
+    coreNum = 1;
+
+    // Original logic commented out:
+    // if(ubPartDataNum >= inputDataNum)
+    // {
+    //     coreNum=1;
+    // }
+    // else
+    // {
+    //     // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
+    //     coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
+    // }

     uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
     uint64_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`28856.476562`
- fast duration us：`1731.234619`
- improvement ratio：`0.9400053358808262`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 10986.9 -> 527366 (+4699.95%)
- memory_ub.gm_to_ub_kb: 32960.8 -> 1.5821e+06 (+4699.94%)
- arithmetic.aiv_vec_fops: 3.94431e+07 -> 1.89323e+09 (+4699.90%)
- critical_path.aiv_vector0_time_us_mean: 1723.73 -> 28855.4 (+1574.01%)
- latency.task_duration_us: 1731.23 -> 28856.5 (+1566.81%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_kb",
  "memory_ub.gm_to_ub_kb",
  "arithmetic.aiv_vec_fops",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us"
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

- In TilingFunc, calculate coreNum based on inputLengthAlign32 and BLOCK_SIZE instead of forcing coreNum = 1
- Use condition: if(ubPartDataNum >= inputDataNum) { coreNum=1; } else { coreNum = min(coreNum, inputLengthAlign32 / BLOCK_SIZE); }
- Ensure each core processes at least 32B of aligned data (BLOCK_SIZE)
- Monitor block_count and active_block_count metrics to verify parallel execution
- Check aiv_vec_fops reduction as indicator of successful workload distribution

## 检索线索

- coreNum
- parallelism
- multi-core
- tiling
- vector
- lerp
- block_dim
- Ascend C
- performance optimization
- workload distribution
