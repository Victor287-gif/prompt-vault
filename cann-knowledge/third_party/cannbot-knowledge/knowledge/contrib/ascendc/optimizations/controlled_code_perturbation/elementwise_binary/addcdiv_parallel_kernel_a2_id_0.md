---
type: Optimization
title: "addcdiv：并行切核优化"
description: "addcdiv 算子的并行切核优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/docs/Addcdiv.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/aclnn_addcdiv_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/aclnn_addcdiv_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/framework/tf_plugin/tensorflow_addcdiv_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/op_host/addcdiv.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/op_host/addcdiv_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/op_kernel/addcdiv.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/Addcdiv_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/test_addcdiv.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# addcdiv：并行切核优化

## 来源信息

- experience ID：`2214f36de9502672`
- 源 schema version：`v2`
- 算子：`addcdiv`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`core.parallelism.enable`
- canonical family：`core.parallelism.disable`
- 源记录时间：`2026-04-18T21:47:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable multi-core parallelization for vector operations
~~~~

### Original summary

~~~~text
Applying multi-core parallelization to the addcdiv operator improves performance by distributing vector operations across 48 cores instead of serializing on a single core. This optimization reduces task duration from 14,339μs to 1,725μs (8.3x speedup) by increasing block_count from 1 to 48 and improving vector unit utilization from 63.6% to 11.1% through better workload distribution.
~~~~

### Original root mechanism

~~~~text
Parallelizing computation across multiple cores reduces latency by distributing work and increasing arithmetic throughput.
~~~~

### Original causal chain

~~~~text
Enabling multi-core execution (block_count: 1→48) reduces per-core workload, decreasing aiv_vector0_time_us_mean from 14,339μs to 1,718μs, which directly reduces task_duration_us from 14,340μs to 1,725μs.
~~~~

## 适用条件

- Vector operators with large input sizes where tileDataNum >= inputNum
- Operators using block-based tiling with BLOCK_SIZE granularity
- When op_context.block_dim shows potential for parallelism but is set to 1
- When aiv_vec_fops is high and vec_ratio_mean > 60% indicating underutilization

## 失效边界

- Operators with small input sizes where overhead of core coordination exceeds benefits
- When memory bandwidth is already saturated (aiv_main_mem_read_bw_gbps > 100 Gbps)
- When block_time_cv shows high variance indicating load imbalance
- When icache_miss_rate_mean is already high (>0.01)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/addcdiv.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`b821b8d056b4190d4c31f5c1538080757ac24eafba9984ddf761b5ed71303f80`

````diff
--- Before/op_host/addcdiv.cpp
+++ After/op_host/addcdiv.cpp
@@ -32,15 +32,9 @@
     uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
     // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits

-    if(tileDataNum >= inputNum)
-    {
-        coreNum=1;
-    }
-    else
-    {
-        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
-        coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
-    }
+    // DEGRADATION: Always use only 1 core to serialize computation
+    // This forces all work onto a single core even when parallelization would be beneficial
+    coreNum = 1;
     uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
     uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`14339.926758`
- fast duration us：`1725.134521`
- improvement ratio：`0.8796971176970917`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 10965.5 -> 526339 (+4699.95%)
- arithmetic.aiv_vec_fops: 2.80571e+07 -> 1.34673e+09 (+4699.95%)
- memory_ub.gm_to_ub_kb: 32896.6 -> 1.57902e+06 (+4699.94%)
- critical_path.aiv_vector0_time_us_mean: 1717.94 -> 14339.1 (+734.66%)
- latency.task_duration_us: 1725.13 -> 14339.9 (+731.24%)

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

- In op_host/addcdiv.cpp TilingFunc, ensure coreNum calculation uses: coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE
- Avoid forcing coreNum = 1 when tileDataNum >= inputNum
- Check block_count in parallelism metrics - should match available cores for optimal performance
- Monitor aiv_vec_wait_ratio_mean - should decrease with proper parallelization (0.527→0.915 in this case)

## 检索线索

- coreNum
- parallelism
- block_count
- tileDataNum
- inputLengthAlgin32
- BLOCK_SIZE
- vector
- addcdiv
- multi-core
- tiling
- scalar_overhead_bound
