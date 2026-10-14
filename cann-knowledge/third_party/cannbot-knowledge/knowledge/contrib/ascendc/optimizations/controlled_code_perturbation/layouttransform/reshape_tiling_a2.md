---
type: Optimization
title: "reshape：Tiling优化"
description: "reshape 算子的Tiling优化经验，关注内存带宽。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.implementation.pipeline", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/docs/Reshape.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/tf_plugin/tensorflow_reshape_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_host/reshape.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_host/reshape_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_kernel/reshape.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reshape：Tiling优化

## 来源信息

- experience ID：`9baf6ad30a78358a`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_003_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.granularity.reduction`
- canonical family：`tiling.granularity.reduction`
- 源记录时间：`2026-04-20T23:07:20`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize tile size calculation by halving tile granularity for better memory efficiency
~~~~

### Original summary

~~~~text
Applying tile size halving in the tiling calculation improves performance by 20.9% for the reshape operator. This optimization reduces tile granularity from `ub_size / ub_size_per_it / mini_batch * mini_batch` to `(tileDiv > 1 ? tileDiv / 2 : 1) * mini_batch`, where `tileDiv = ub_size / ub_size_per_it / mini_batch`. The smaller tile size increases iteration count but reduces memory bandwidth pressure and improves cache efficiency, particularly beneficial for memory-bound operators with high MTE wait ratios.
~~~~

### Original root mechanism

~~~~text
Halving the tile size reduces data transfer per iteration, improving memory bandwidth utilization and reducing pipeline stalls by better aligning with memory subsystem characteristics.
~~~~

### Original causal chain

~~~~text
apply tiling.granularity.reduction in reshape/op_host/reshape.cpp -> improve aiv_utilization.scalar_ratio_mean -> 20.9% latency improvement
~~~~

## 适用条件

- Operator uses tiling with `SetTilingKey(1)` in Ascend C platform
- Memory bandwidth utilization shows high values (>20% for gm_to_ub_bw_usage_rate_pct and ub_to_gm_bw_usage_rate_pct)
- MTE wait ratios are high (>0.99 for aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean)
- Tile calculation uses `ub_size / ub_size_per_it / mini_batch * mini_batch` pattern
- Operator is memory-bound with significant data movement between GM and UB

## 失效边界

- Operator is compute-bound with low memory bandwidth utilization (<10%)
- Tile size is already optimal for the memory subsystem
- Increasing iteration count would exceed kernel launch overhead limits
- Data dependencies prevent effective parallelization of smaller tiles

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reshape/op_host/reshape.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`d564ee87fbe92be485bb014567fd849c06ba85ef19e75b29f3cba4983a431e1e`

````diff
--- Before/op_host/reshape.cpp
+++ After/op_host/reshape.cpp
@@ -74,7 +74,8 @@
         printf("sizeof(data)= %u, ", type_sz);
         if(ub_size_per_it != 0) { // known data types
             context->SetTilingKey(1);
-            uint32_t tileLength = ub_size / ub_size_per_it / mini_batch * mini_batch;
+            uint32_t tileDiv = ub_size / ub_size_per_it / mini_batch;
+            uint32_t tileLength = (tileDiv > 1 ? tileDiv / 2 : 1) * mini_batch;
             uint32_t tileNumber = dataLength / tileLength;
             uint32_t reminder = (dataLength % tileLength + mini_batch - 1) / mini_batch * mini_batch;
             SET(tileLength);
````

## 观测记录

- bottleneck type：`memory_bandwidth`
- slow duration us：`4545.390625`
- fast duration us：`3594.671875`
- improvement ratio：`0.20916106632749523`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.032322 -> 0.05087 (+57.39%)
- latency.task_duration_us: 3594.67 -> 4545.39 (+26.45%)
- critical_path.aiv_vector0_time_us_mean: 3590.94 -> 4539.78 (+26.42%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 27.9 -> 22.07 (-20.90%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 25.18 -> 19.92 (-20.89%)

### Affected metric groups

- aiv_utilization
- latency
- critical_path
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.032322,
      "after": 0.05087,
      "delta": 0.018548000000000002,
      "rel": 0.5738506280551947
    }
  },
  {
    "latency.task_duration_us": {
      "before": 3594.672,
      "after": 4545.391,
      "delta": 950.7189999999996,
      "rel": 0.2644800415726385
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 3590.944,
      "after": 4539.782,
      "delta": 948.8380000000002,
      "rel": 0.2642307983638843
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 27.9,
      "after": 22.07,
      "delta": -5.829999999999998,
      "rel": -0.20896057347670247
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 25.18,
      "after": 19.92,
      "delta": -5.259999999999998,
      "rel": -0.20889594916600468
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

- In the TilingFunc implementation (reshape/op_host/reshape.cpp:77-78), replace `uint32_t tileLength = ub_size / ub_size_per_it / mini_batch * mini_batch;` with `uint32_t tileDiv = ub_size / ub_size_per_it / mini_batch; uint32_t tileLength = (tileDiv > 1 ? tileDiv / 2 : 1) * mini_batch;`
- Verify the optimization by checking memory bandwidth metrics: aiv_main_mem_read_bw_gbps and aiv_main_mem_write_bw_gbps should decrease by ~20%
- Monitor UB bandwidth utilization metrics: gm_to_ub_bw_usage_rate_pct and ub_to_gm_bw_usage_rate_pct should decrease by ~20%
- Check that scalar ratio utilization (aiv_utilization.scalar_ratio_mean) increases as overhead from additional iterations is offset by better memory efficiency
- Ensure tileNumber calculation accounts for the halved tile size: `uint32_t tileNumber = dataLength / tileLength;`

## 检索线索

- tile
- tiling
- granularity
- reshape
- memory_bandwidth
- ub_size
- mini_batch
- SetTilingKey
- AscendC
- MTE_wait
