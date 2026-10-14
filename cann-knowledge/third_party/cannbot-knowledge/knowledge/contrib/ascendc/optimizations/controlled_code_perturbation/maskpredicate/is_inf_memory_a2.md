---
type: Optimization
title: "is_inf：内存优化"
description: "is_inf 算子的内存优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.tiling", "topic.data_movement.memory_access"]
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
# is_inf：内存优化

## 来源信息

- experience ID：`b6a8df86b349bf6d`
- 源 schema version：`v2`
- 算子：`is_inf`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.buffer.size.reduction`
- canonical family：`memory.buffer.size.reduction`
- 源记录时间：`2026-04-20T19:02:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase UB_DIVIDER_FOR_TEMP_CASTING to reduce usable UB memory
~~~~

### Original summary

~~~~text
For Ascend C operators using unified buffer (UB) memory, increasing the UB_DIVIDER_FOR_TEMP_CASTING constant from 10 to 20 reduces the usable UB memory by half, forcing the operator to process data in smaller chunks. This optimization improves L2 cache hit rates by 393.55% (from 0.31% to 1.53%) and reduces pipeline stalls by decreasing bank group conflicts while maintaining similar vector utilization. Apply this technique when UB memory usage shows high bandwidth but poor cache efficiency.
~~~~

### Original root mechanism

~~~~text
Increasing UB_DIVIDER_FOR_TEMP_CASTING reduces the usable UB memory allocation, forcing smaller chunk processing which improves cache hit rates and reduces pipeline stalls
~~~~

### Original causal chain

~~~~text
Increased UB_DIVIDER_FOR_TEMP_CASTING from 10 to 20 → Reduced usable UB memory allocation → Smaller data chunks processed per iteration → Improved L2 cache hit rates (cache_l2.aiv_total_hit_rate_pct: 0.31 → 1.53) → Reduced pipeline stalls (aiv_vec_total_cflt_ratio_mean: 0.116 → 0.066) → Decreased task duration from 1818μs to 1632μs
~~~~

## 适用条件

- Operator uses unified buffer (UB) memory with UB_DIVIDER_FOR_TEMP_CASTING constant
- Profile shows low L2 cache hit rates (<2%) despite high UB bandwidth usage
- Pipeline stalls show high aiv_vec_bankgroup_cflt_ratio_mean (>0.05)
- Operator processes large data sets that exceed available cache capacity
- UB memory bandwidth utilization is high but cache efficiency is poor

## 失效边界

- Operator already has high L2 cache hit rates (>10%)
- UB memory bandwidth is already the limiting factor
- Operator has strict memory size requirements that cannot be reduced
- Data access patterns are already optimal for cache locality
- Changing UB_DIVIDER_FOR_TEMP_CASTING would cause compilation errors

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/is_inf.cpp

### 改动符号

- UB_DIVIDER_FOR_TEMP_CASTING
- GetUsableUbMemory

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

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1818.416382`
- fast duration us：`1632.152588`
- improvement ratio：`0.10243187195395613`
- machine confidence：`0.85`

### Profile evidence

- cache_l2.aiv_total_hit_rate_pct: 0.31 -> 1.53 (+393.55%)
- cache_l2.aiv_write_hit_rate_pct: 5 -> 20.83 (+316.60%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.026309 -> 0.066078 (+151.16%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.089872 -> 0 (-100.00%)
- aiv_utilization.scalar_ratio_mean: 0.228912 -> 0.41286 (+80.36%)

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

- Locate UB_DIVIDER_FOR_TEMP_CASTING constant in operator source files (e.g., op_host/is_inf.cpp:35)
- Increase the divider value (e.g., from 10 to 20) to reduce usable UB memory
- Verify GetUsableUbMemory function uses this constant for memory allocation
- Profile the operator to confirm improved cache_l2.aiv_total_hit_rate_pct
- Monitor pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean for reduction
- Check that UB bandwidth (ub_read_bw_vector_gbps) remains sufficient for workload

## 检索线索

- UB_DIVIDER_FOR_TEMP_CASTING
- unified_buffer
- cache_hit_rate
- pipeline_stalls
- memory_buffer_size
- ascend_c
- is_inf
- math_operator
- cache_efficiency
- bank_conflict
