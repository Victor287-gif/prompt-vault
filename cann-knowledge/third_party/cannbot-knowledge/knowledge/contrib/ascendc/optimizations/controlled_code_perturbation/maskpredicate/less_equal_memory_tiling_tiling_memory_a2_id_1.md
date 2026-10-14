---
type: Optimization
title: "less_equal：内存与 Tiling优化"
description: "less_equal 算子的内存与 Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.tiling"]
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

- experience ID：`fd21767dbb9e2ffc`
- 源 schema version：`v2`
- 算子：`less_equal`
- source iteration：`round_012_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.buffer.overpartition`
- canonical family：`tiling.buffer.overpartition`
- 源记录时间：`2026-04-20T21:26:51`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase buffer allocation count to reduce tile size
~~~~

### Original summary

~~~~text
Increasing bufferNum parameter from 32 to 64 in tiling calculations reduces maximum tile size while increasing tile count per block. This optimization improves cache hit rates (L2 read hit rate increased from 3.33% to 11.72%) and reduces vector unit wait ratios, leading to 41% performance improvement. Apply this technique when tiling calculations show high vector wait ratios and low cache hit rates despite sufficient memory bandwidth.
~~~~

### Original root mechanism

~~~~text
Doubling bufferNum from 32 to 64 reduces maximum tile size and increases tile count per block, improving data locality and reducing pipeline stalls
~~~~

### Original causal chain

~~~~text
Increased bufferNum from 32 to 64 → Reduced maximum tile size → Increased number of tiles per block → Improved L2 cache hit rates (3.33% to 11.72%) → Reduced vector unit wait ratios → Decreased aiv_vector0_time_us_mean from 3289.09μs to 1923.99μs → Reduced task_duration_us from 3324.61μs to 1957.56μs
~~~~

## 适用条件

- TilingFunc implementations with bufferNum parameter controlling tile size calculations
- Operators showing low L2 cache hit rates (<15%) despite sufficient memory bandwidth
- Vector units showing high wait ratios (>60%) indicating memory access bottlenecks
- Performance profiles with high aiv_vector0_time_us_mean relative to total task duration
- Operators using platform_ascendc::PlatformAscendC.GetCoreMemSize() for UB size calculations

## 失效边界

- When UB memory is already fully utilized (ub_read_bw_vector_gbps > 80 Gbps)
- When increasing tile count would exceed coreNum * bufferNum limitations
- When arithmetic intensity is already high (aiv_vec_fp32_ratio_mean > 0.5)
- When L2 cache hit rates are already high (>30%)
- When vector unit utilization is already optimal (vec_ratio_mean > 0.5)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/less_equal.cpp

### 改动符号

- TilingFunc

### 代码签名

- GetCoreNum
- coreNum
- core_launch
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`812da6ad447eb4ed9992253f17572b676dde6f365b336ff6c348024274fde7c3`

````diff
--- Before/op_host/less_equal.cpp
+++ After/op_host/less_equal.cpp
@@ -105,7 +105,7 @@
 static ge::graphStatus TilingFunc(gert::TilingContext* context) {
   LessEqualTilingData tiling;
   uint64_t ubSize;
-  uint32_t bufferNum = 32;
+  uint32_t bufferNum = 64;
   auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
   ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
   auto coreNum = ascendcPlatform.GetCoreNumAiv();
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`3324.606689`
- fast duration us：`1957.559204`
- improvement ratio：`0.41119074010261075`
- machine confidence：`0.9`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 3.33 -> 11.72 (+251.95%)
- cache_l2.aiv_total_hit_rate_pct: 6.45 -> 18.14 (+181.24%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.012963 -> 0.022252 (+71.66%)
- critical_path.aiv_vector0_time_us_mean: 1923.99 -> 3289.09 (+70.95%)
- latency.task_duration_us: 1957.56 -> 3324.61 (+69.83%)

### Affected metric groups

- cache_l2
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "memory_gm.aiv_main_mem_read_bw_gbps",
  "memory_ub.ub_write_bw_vector_gbps",
  "cache_l2.aiv_write_hit_rate_pct"
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

- 1. In TilingFunc implementations, locate the bufferNum variable declaration (typically uint32_t bufferNum = default_value)
- 2. Check if current bufferNum value is less than 64 for Ascend C platforms
- 3. Increase bufferNum to 64 or higher while ensuring total tile count doesn't exceed coreNum * bufferNum
- 4. Verify UB memory usage doesn't exceed available UB size using GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)
- 5. Re-profile to confirm L2 cache hit rate improvement and vector wait ratio reduction
- 6. Monitor aiv_vector0_time_us_mean and task_duration_us for expected 30-50% reduction

## 检索线索

- bufferNum
- tiling
- overpartition
- cache_hit_rate
- vector_wait
- L2_cache
- tile_size
- ascendc
- TilingFunc
- GetCoreMemSize
- UB
- coreNum
- platform_ascendc
