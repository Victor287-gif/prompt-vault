---
type: Optimization
title: "gcd：并行优化"
description: "gcd 算子的并行优化经验，关注并行度。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/docs/Gcd.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/framework/tf_plugin/tensorflow_sqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_host/gcd_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/op_kernel/gcd_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gcd/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gcd：并行优化

## 来源信息

- experience ID：`667222ad002adaca`
- 源 schema version：`v2`
- 算子：`gcd`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.reduction`
- canonical family：`parallelism.reduction`
- 源记录时间：`2026-04-20T19:09:04`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase AI core parallelism by maximizing block dimension allocation
~~~~

### Original summary

~~~~text
For GCD vector operators, maximizing the block dimension by using all available AI cores (via ascendcPlatform.GetCoreNumAiv()) provides significant performance improvement. This optimization doubles the active block count from 24 to 48, reducing latency by approximately 50% while maintaining similar vector unit utilization and memory bandwidth patterns.
~~~~

### Original root mechanism

~~~~text
Increasing block dimension from 24 to 48 doubles AI core parallelism, allowing more concurrent execution of vector operations and reducing overall latency.
~~~~

### Original causal chain

~~~~text
Setting block dimension to full AI core count (48) -> Increases active block count from 24 to 48 -> Doubles parallel execution capacity -> Reduces task duration from 5719.9µs to 2863.3µs
~~~~

## 适用条件

- Optimizing vector operators with independent parallel work units
- When ascendcPlatform.GetCoreNumAiv() returns more cores than currently used
- When block dimension is artificially limited in tiling functions
- When performance profiling shows low active_block_count relative to available AI cores

## 失效边界

- When memory bandwidth is already saturated
- When vector unit utilization is already near 100%
- When operator has data dependencies preventing parallel execution
- When increasing parallelism causes resource contention or cache thrashing

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/gcd.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- GetCoreNum
- SetBlockDim
- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`a42d8d8d650659bd61d4559bce2fc0e4c2477cea7f420eb7f279529c68a9a260`

````diff
--- Before/op_host/gcd.cpp
+++ After/op_host/gcd.cpp
@@ -49,6 +49,9 @@
     tiling.set_broadcast_mask(broadcast_mask);

     uint32_t n_core = ascendcPlatform.GetCoreNumAiv();
+    // Artificially reduce available cores to decrease parallelism and degrade performance
+    n_core = n_core / 2;
+    if (n_core < 1) n_core = 1;
     context->SetBlockDim(n_core);
     tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
     context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
````

## 观测记录

- bottleneck type：`parallelism`
- slow duration us：`5719.93457`
- fast duration us：`2863.337158`
- improvement ratio：`0.49941085462451373`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 344 -> 688 (+100.00%)
- arithmetic.aiv_vec_fops: 1.10593e+08 -> 2.21183e+08 (+100.00%)
- memory_ub.gm_to_ub_kb: 688.1 -> 1376.1 (+99.99%)
- critical_path.aiv_vector0_time_us_mean: 2839.1 -> 5673.95 (+99.85%)
- latency.task_duration_us: 2863.34 -> 5719.94 (+99.76%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 344.0,
      "after": 688.0,
      "delta": 344.0,
      "rel": 1.0
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 110592736.0,
      "after": 221183232.0,
      "delta": 110590496.0,
      "rel": 0.9999797455051659
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 688.1,
      "after": 1376.1,
      "delta": 687.9999999999999,
      "rel": 0.9998546722860048
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2839.096,
      "after": 5673.954,
      "delta": 2834.8579999999997,
      "rel": 0.99850727132862
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2863.337,
      "after": 5719.935,
      "delta": 2856.5980000000004,
      "rel": 0.9976464523735769
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 0.05,
      "after": 0.02,
      "delta": -0.030000000000000002,
      "rel": -0.6
    }
  },
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 0.15,
      "after": 0.07,
      "delta": -0.07999999999999999,
      "rel": -0.5333333333333333
    }
  },
  {
    "parallelism.active_block_count": {
      "before": 48.0,
      "after": 24.0,
      "delta": -24.0,
      "rel": -0.5
    }
  },
  {
    "parallelism.block_count": {
      "before": 48.0,
      "after": 24.0,
      "delta": -24.0,
      "rel": -0.5
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 0.04,
      "after": 0.02,
      "delta": -0.02,
      "rel": -0.5
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

- In the TilingFunc implementation, call ascendcPlatform.GetCoreNumAiv() to get maximum AI core count
- Set block dimension using context->SetBlockDim(n_core) with the full core count
- Verify that the operator workload can be evenly distributed across all cores
- Monitor aiv_vec_fops and memory_ub metrics to ensure workload scales linearly
- Check that active_block_count equals block_count in performance metrics

## 检索线索

- parallelism
- block_dim
- SetBlockDim
- GetCoreNumAiv
- AI core
- vector operator
- gcd
- tiling
- performance scaling
