---
type: Optimization
title: "equal：并行切核优化"
description: "equal 算子的并行切核优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/docs/Equal.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/framework/tf_plugin/tensorflow_equal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/op_host/equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/op_host/equal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/op_kernel/equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/st/Equal_case_fp_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/st/Equal_case_int_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/st/test_equal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/equal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# equal：并行切核优化

## 来源信息

- experience ID：`3513f1dc05da4e67`
- 源 schema version：`v2`
- 算子：`equal`
- source iteration：`round_017_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.core.parallelization`
- canonical family：`parallelism.core.serialization`
- 源记录时间：`2026-04-20T16:09:00`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable multi-core parallel execution for vector operations
~~~~

### Original summary

~~~~text
Applying multi-core parallel execution to vector operations significantly reduces latency by distributing the workload across available AI cores. This optimization is most effective for operators with large data volumes that can be partitioned across cores, reducing the time each core spends on vector operations and improving memory bandwidth usage. The technique works by adjusting the coreNum parameter to match the data size rather than forcing serial execution.
~~~~

### Original root mechanism

~~~~text
Distributing vector operations across multiple AI cores reduces per-core workload and improves memory bandwidth utilization by enabling concurrent data transfer.
~~~~

### Original causal chain

~~~~text
Increasing coreNum from 1 to 48 -> reduces aiv_vector0_time_us_mean from 16267.026 to 1268.273 (11.8x reduction) -> decreases task_duration_us from 16267.866 to 1273.465 (12.8x reduction)
~~~~

## 适用条件

- When implementing vector operations in Ascend C with large input data sizes
- When the operator uses BLOCK_SIZE parameter for data partitioning
- When coreNum calculation logic is present in TilingFunc functions
- When aiv_vector0_time_us_mean shows high values indicating vector operation bottlenecks

## 失效边界

- When data size is smaller than BLOCK_SIZE (tileDataNum >= inputNum)
- When memory bandwidth is already saturated
- When the operation has strong data dependencies requiring sequential processing

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- equal/op_host/equal.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`e00c20518b2402d6931d37eea8f41b027c2a05e2e44a5c1f637f58b5609bdd55`

````diff
--- Before/op_host/equal.cpp
+++ After/op_host/equal.cpp
@@ -49,15 +49,8 @@

         uint64_t inputLengthAlgin32 = (((inputNum + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

-        if(tileDataNum >= inputNum)
-        {
-            coreNum=1;
-        }
-        else
-        {
-            // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
-            coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
-        }
+        // Force serial execution by limiting to 1 core regardless of data size
+        coreNum = 1;

        if (BLOCK_SIZE == 0 || coreNum == 0)
        {
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`16267.866211`
- fast duration us：`1273.465454`
- improvement ratio：`0.9217189619411237`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 6630.5 -> 318264 (+4700.00%)
- memory_ub.gm_to_ub_kb: 26418.9 -> 1.2681e+06 (+4699.98%)
- critical_path.aiv_vector0_time_us_mean: 1268.27 -> 16267 (+1182.61%)
- latency.task_duration_us: 1273.46 -> 16267.9 (+1177.45%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 10.17 -> 38.13 (+274.93%)

### Affected metric groups

- memory_ub
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 6630.5,
      "after": 318263.9,
      "delta": 311633.4,
      "rel": 46.999984918181134
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 26418.9,
      "after": 1268101.9,
      "delta": 1241683.0,
      "rel": 46.999799386045595
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1268.273,
      "after": 16267.026,
      "delta": 14998.753,
      "rel": 11.826123397722732
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1273.465,
      "after": 16267.866,
      "delta": 14994.401,
      "rel": 11.774490072361628
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 10.17,
      "after": 38.13,
      "delta": 27.96,
      "rel": 2.7492625368731565
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

- Check the TilingFunc function in op_host/equal.cpp for coreNum calculation logic
- Replace hardcoded coreNum = 1 with dynamic calculation based on inputNum and BLOCK_SIZE
- Use the formula: coreNum = min(available_cores, inputLengthAlgin32 / BLOCK_SIZE)
- Ensure coreNum > 1 when inputNum > BLOCK_SIZE to enable parallel execution
- Verify performance improvement by monitoring aiv_vector0_time_us_mean and task_duration_us metrics

## 检索线索

- parallelism
- coreNum
- vector
- Ascend C
- BLOCK_SIZE
- TilingFunc
- multi-core
- memory_bandwidth
- aiv_vector0_time_us_mean
- task_duration_us
