---
type: Optimization
title: "swish：内存与 Tiling优化"
description: "swish 算子的内存与 Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/docs/Swish.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/framework/tf_plugin/tensorflow_swish_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/op_host/swish.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/op_host/swish_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/op_kernel/swish.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/Swish_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/st/test_swish.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/swish/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# swish：内存与 Tiling优化

## 来源信息

- experience ID：`36864e70923e76c4`
- 源 schema version：`v2`
- 算子：`swish`
- source iteration：`round_018_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.buffer.misalignment`
- canonical family：`tiling.buffer.misalignment`
- 源记录时间：`2026-04-21T01:35:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Host-kernel buffer count mismatch amplification
~~~~

### Original summary

~~~~text
On `math/swish`, applying `Host-kernel buffer count mismatch amplification` yields 14.5% performance improvement (1415.7 vs 1656.2 us, 14.5% difference) by addressing `mixed_or_unknown`. Increased host BUFFER_NUM constant from 2 to 4 while kernel uses BUFFER_NUM=1, causing host to calculate smaller tile sizes than physically possible and increasing loop iteration count
~~~~

### Original root mechanism

~~~~text
Increased host BUFFER_NUM constant from 2 to 4 while kernel uses BUFFER_NUM=1, causing host to calculate smaller tile sizes than physically possible and increasing loop iteration count
~~~~

### Original causal chain

~~~~text
apply tiling.buffer.misalignment in op_host/swish.cpp -> improve pipeline_stalls.aiv_vec_bank_cflt_ratio_mean -> 14.5% latency improvement
~~~~

## 适用条件

- The target operator's profiling shows bottleneck in: pipeline_stalls, cache_l2, arithmetic
- The kernel code uses related APIs/patterns: BUFFER_NUM, tile, tile_shape
- The operator has a `mixed_or_unknown` bottleneck profile

## 失效边界

- The operator's bottleneck is unrelated to `mixed_or_unknown`
- The `tiling.buffer.misalignment` pattern is not present in the target code
- The same optimization is already applied in the current implementation

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/swish.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- BUFFER_NUM
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`56aa72386569b23443eaed7bd6a146f7df2b09331a7c19ad4e82ce189212a5f2`

````diff
--- Before/op_host/swish.cpp
+++ After/op_host/swish.cpp
@@ -8,7 +8,7 @@

 namespace optiling {
 const uint64_t BLOCK_SIZE = 32;
-const uint64_t BUFFER_NUM = 2;
+const uint64_t BUFFER_NUM = 4;

 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 {

````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1656.193115`
- fast duration us：`1415.728271`
- improvement ratio：`0.14519130759700088`
- machine confidence：`0.82`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.025433 -> 0.050993 (+100.50%)
- cache_l2.aiv_read_hit_rate_pct: 1.6 -> 3.19 (+99.37%)
- cache_l2.aiv_total_hit_rate_pct: 2.33 -> 4.59 (+97.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.002293 -> 0.004396 (+91.71%)
- cache_l2.aiv_write_hit_rate_pct: 5.08 -> 9.67 (+90.35%)

### Affected metric groups

- pipeline_stalls
- cache_l2
- arithmetic
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.025433,
      "after": 0.050993,
      "delta": 0.025559999999999996,
      "rel": 1.0049935123658238
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 1.6,
      "after": 3.19,
      "delta": 1.5899999999999999,
      "rel": 0.9937499999999999
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 2.33,
      "after": 4.59,
      "delta": 2.26,
      "rel": 0.9699570815450642
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.002293,
      "after": 0.004396,
      "delta": 0.002103,
      "rel": 0.9171391190580026
    }
  },
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 5.08,
      "after": 9.67,
      "delta": 4.59,
      "rel": 0.9035433070866141
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.110571,
      "after": 0.179827,
      "delta": 0.06925599999999998,
      "rel": 0.6263486809380396
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1415.728,
      "after": 1656.193,
      "delta": 240.46499999999992,
      "rel": 0.16985254229626023
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1409.066,
      "after": 1644.664,
      "delta": 235.59799999999996,
      "rel": 0.16720153633683585
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.102542,
      "after": 0.087738,
      "delta": -0.014803999999999998,
      "rel": -0.14437011175908407
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 165.45,
      "after": 141.94,
      "delta": -23.50999999999999,
      "rel": -0.14209731036566933
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

- Locate the `tiling.buffer.misalignment` code region in `op_host/swish.cpp` (key symbols: optiling::TilingFunc)
- Apply the optimization pattern shown in the code_diff
- After applying, verify that `pipeline_stalls.aiv_vec_bank_cflt_ratio_mean` moves in the expected direction

## 检索线索

- After
- BLOCK_SIZE
- BUFFER_NUM
- Before
- TilingContext
- TilingFunc
- aiv_utilization
- arithmetic
- cache_l2
- graphStatus
- mixed_or_unknown
- pipeline_stalls
- tile
- tile_shape
- tiling.buffer.misalignment
