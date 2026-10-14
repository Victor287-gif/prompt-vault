---
type: Optimization
title: "is_inf：计算与内存优化"
description: "is_inf 算子的计算与内存优化经验，关注UB 内存。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_kernel/is_inf.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_kernel/is_inf.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# is_inf：计算与内存优化

## 来源信息

- experience ID：`2b334d411a26530d`
- 源 schema version：`v2`
- 算子：`is_inf`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.division.inefficiency`
- canonical family：`memory.division.inefficiency`
- 源记录时间：`2026-04-18T21:59:40`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce UB memory division factor to increase usable UB allocation
~~~~

### Original summary

~~~~text
In Ascend C operators, the UB_DIVIDER_FOR_TEMP_CASTING constant controls how UB memory is partitioned for temporary data casting. Reducing this value from 20 to 10 increases usable UB memory allocation by 2x, enabling larger data transfers per iteration and improving memory bandwidth utilization. Apply this optimization when UB memory bandwidth metrics show low utilization despite sufficient compute resources.
~~~~

### Original root mechanism

~~~~text
Decreasing the UB_DIVIDER_FOR_TEMP_CASTING constant reduces memory division overhead, allowing larger data chunks per iteration and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Reduced UB_DIVIDER_FOR_TEMP_CASTING from 20 to 10 → Increased usable UB memory allocation → Improved UB read/write bandwidth (202.27→168.48 Gbps read, 173.05→158.52 Gbps write) → Reduced pipeline stalls (aiv_vec_total_cflt_ratio_mean 0.115841→0.066416) → Lowered latency from 1798.276μs to 1644.853μs.
~~~~

## 适用条件

- Operator uses UB memory for temporary data casting with UB_DIVIDER_FOR_TEMP_CASTING constant
- UB memory bandwidth metrics show low utilization (ub_read_bw_vector_gbps < 200 Gbps)
- Pipeline stalls show high aiv_vec_total_cflt_ratio_mean (>0.05)
- Cache hit rates are low (aiv_total_hit_rate_pct < 2%)

## 失效边界

- Operator already achieves near-peak UB memory bandwidth (>250 Gbps)
- UB memory is constrained by other factors like total available UB size
- Operator has minimal memory access patterns (compute-bound rather than memory-bound)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/is_inf.cpp

### 改动符号

- UB_DIVIDER_FOR_TEMP_CASTING

### 代码签名

- compile_time_specialization
- constexpr
- ub

### 原始 diff

- 源 diff SHA-256：`f5da6a1ae6764a893938de9a16b1137e2b316a66d1ab9c55dfcea7bdc115f512`

````diff
--- Before/op_host/is_inf.cpp
+++ After/op_host/is_inf.cpp
@@ -32,7 +32,7 @@
 constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
 constexpr uint32_t BYTE_LEN_4 = 4;

-constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;
+constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 20;


 class IsInfTiling {
````

## 观测记录

- bottleneck type：`memory_ub`
- slow duration us：`1798.276001`
- fast duration us：`1644.852905`
- improvement ratio：`0.0853167677901964`
- machine confidence：`0.9`

### Profile evidence

- cache_l2.aiv_total_hit_rate_pct: 0.31 -> 1.53 (+393.55%)
- cache_l2.aiv_write_hit_rate_pct: 5 -> 20.83 (+316.60%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.026232 -> 0.066416 (+153.19%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.089609 -> 0 (-100.00%)
- aiv_utilization.scalar_ratio_mean: 0.228425 -> 0.415815 (+82.04%)

### Affected metric groups

- cache_l2
- pipeline_stalls
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean",
  "aiv_utilization.scalar_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.865493,
  "overall_deviation": 0.346197,
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

- 1. Locate UB_DIVIDER_FOR_TEMP_CASTING constant in op_host/is_inf.cpp or similar host files
- 2. Reduce the constant value (e.g., from 20 to 10) to increase usable UB memory allocation
- 3. Verify UB memory bandwidth metrics improve (ub_read_bw_vector_gbps, ub_write_bw_vector_gbps)
- 4. Check for reduced pipeline stalls (aiv_vec_total_cflt_ratio_mean, aiv_vec_bank_cflt_ratio_mean)
- 5. Monitor cache hit rates (aiv_total_hit_rate_pct, aiv_write_hit_rate_pct) for improvements

## 检索线索

- UB_DIVIDER_FOR_TEMP_CASTING
- memory division
- UB memory allocation
- Ascend C
- is_inf
- temporary casting
- memory bandwidth
- pipeline stalls
- cache hit rate
