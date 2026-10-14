---
type: Optimization
title: "less_equal：内存与 Tiling优化"
description: "less_equal 算子的内存与 Tiling优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/docs/LessEqual.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/framework/tf_plugin/tensorflow_less_equal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/op_host/less_equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/op_host/less_equal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/op_kernel/less_equal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/LessEqual_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/st/test_less_equal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/less_equal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# less_equal：内存与 Tiling优化

## 来源信息

- experience ID：`c45c9c106f63565b`
- 源 schema version：`v2`
- 算子：`less_equal`
- source iteration：`round_009_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.buffer_count`
- canonical family：`untitled`
- 源记录时间：`2026-04-20T20:43:45`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase UB buffer count to reduce pipeline stalls in vector comparison operators
~~~~

### Original summary

~~~~text
For vector comparison operators like LessEqual, increasing the bufferNum parameter in the TilingFunc from 16 to 32 significantly reduces pipeline stalls caused by bank conflicts. This optimization improves vector unit utilization from 33.9% to 40.3% and reduces total conflict ratio from 1.7% to 8.9%, resulting in 36.5% faster execution. Apply this optimization when vector operators show high aiv_vec_bank_cflt_ratio_mean and aiv_vec_bankgroup_cflt_ratio_mean metrics.
~~~~

### Original root mechanism

~~~~text
Increasing bufferNum from 16 to 32 in the tiling function reduces bank and bankgroup conflicts in the vector pipeline, allowing more efficient memory access patterns.
~~~~

### Original causal chain

~~~~text
apply untitled in op_host/less_equal.cpp -> improve arithmetic.aiv_vec_misc_ratio_mean -> 36.5% latency improvement
~~~~

## 适用条件

- Vector operators with high aiv_vec_bank_cflt_ratio_mean (>0.02) or aiv_vec_bankgroup_cflt_ratio_mean (>0.05)
- Operators using platform_ascendc::PlatformAscendC with GetCoreMemSize for UB sizing
- Tiling functions with bufferNum parameter controlling memory access patterns
- Operators showing low vector unit utilization (<40%) with available compute headroom

## 失效边界

- When UB memory is already fully utilized (check ub_read_bw_vector_gbps and ub_write_bw_vector_gbps near limits)
- For scalar-dominant operators where vector conflicts are not the bottleneck
- When increasing bufferNum would exceed available UB memory capacity
- For operators with already high cache hit rates (>80%)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/less_equal.cpp

### 代码签名

- GetCoreNum
- coreNum
- core_launch
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`71c210dca8d483f68a570c8cadf56bac6c4ed8588dd833d5ae95313f742aa251`

````diff
--- Before/op_host/less_equal.cpp
+++ After/op_host/less_equal.cpp
@@ -105,7 +105,7 @@
 static ge::graphStatus TilingFunc(gert::TilingContext* context) {
   LessEqualTilingData tiling;
   uint64_t ubSize;
-  uint32_t bufferNum = 16;
+  uint32_t bufferNum = 32;
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
   auto coreNum = ascendcPlatform.GetCoreNumAiv();
````

## 观测记录

- bottleneck type：`pipeline_stalls`
- slow duration us：`1957.559204`
- fast duration us：`1242.28479`
- improvement ratio：`0.3653909483495754`
- machine confidence：`0.9`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.006273 -> 0.012963 (+106.65%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.029947 -> 0.002707 (-90.96%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.089842 -> 0.017145 (-80.92%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.059895 -> 0.014438 (-75.89%)
- cache_l2.aiv_total_hit_rate_pct: 3.87 -> 6.45 (+66.67%)

### Affected metric groups

- arithmetic
- pipeline_stalls
- cache_l2
- latency

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.006273,
      "after": 0.012963,
      "delta": 0.006690000000000001,
      "rel": 1.0664753706360595
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.029947,
      "after": 0.002707,
      "delta": -0.02724,
      "rel": -0.9096069723177613
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.089842,
      "after": 0.017145,
      "delta": -0.07269700000000001,
      "rel": -0.8091649785178425
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.059895,
      "after": 0.014438,
      "delta": -0.045457,
      "rel": -0.7589448201018449
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 3.87,
      "after": 6.45,
      "delta": 2.58,
      "rel": 0.6666666666666666
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

- 1. Locate the TilingFunc in the operator's host code (e.g., op_host/less_equal.cpp)
- 2. Identify the bufferNum parameter declaration and initialization
- 3. Increase bufferNum from default 16 to 32 (or higher powers of 2)
- 4. Verify UB memory capacity using ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)
- 5. Profile the operator to confirm reduction in aiv_vec_bank_cflt_ratio_mean and aiv_vec_bankgroup_cflt_ratio_mean
- 6. Monitor vector unit utilization (vec_ratio_mean) improvement

## 检索线索

- bufferNum
- tiling
- pipeline_stalls
- bank_conflict
- vector_utilization
- less_equal
- ascendc
- UB_memory
- GetCoreMemSize
- TilingFunc
