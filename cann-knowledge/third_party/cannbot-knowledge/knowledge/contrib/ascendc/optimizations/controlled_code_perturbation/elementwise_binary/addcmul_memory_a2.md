---
type: Optimization
title: "addcmul：内存优化"
description: "addcmul 算子的内存优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/docs/Addcmul.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/framework/tf_plugin/tensorflow_addcmul_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/op_host/addcmul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/op_host/addcmul_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/op_kernel/addcmul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/Addcmul_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/test_addcmul.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# addcmul：内存优化

## 来源信息

- experience ID：`59ca5d2cbd1531d9`
- 源 schema version：`v2`
- 算子：`addcmul`
- source iteration：`round_013_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.alignment.power_of_two`
- canonical family：`memory.alignment.inefficient.non.power.of.two`
- 源记录时间：`2026-04-20T15:09:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Power-of-two block size alignment for memory efficiency
~~~~

### Original summary

~~~~text
Using power-of-two block sizes (32, 64, 128) instead of non-power-of-two values (31, 63, 127) improves memory access efficiency in Ascend C operators. This optimization reduces memory subsystem bottlenecks by ensuring aligned cache line accesses, which decreases memory transfer overhead and improves overall throughput. Apply this technique when BLOCK_SIZE constants control memory access patterns in vector operations.
~~~~

### Original root mechanism

~~~~text
Using power-of-two block sizes (e.g., 32) ensures efficient memory alignment and cache line access patterns compared to non-power-of-two values (e.g., 31) that cause misaligned accesses.
~~~~

### Original causal chain

~~~~text
Changing BLOCK_SIZE from 31 to 32 (power of two) → reduces misaligned memory accesses → decreases memory transfer overhead (ub_to_gm_kb by 19.1%, ub_to_gm_bw_usage_rate_pct by 18.4%) → improves cache hit rates (L2 write hit rate by 15.4%, total hit rate by 12.6%) → reduces pipeline stalls → decreases critical path time by 5.2% → reduces overall latency by 6.6%
~~~~

## 适用条件

- BLOCK_SIZE constant controls memory access patterns in Ascend C operators
- Memory-bound operations with vector loads/stores
- UB to GM memory transfers show high bandwidth usage (>3%)
- Cache hit rates are suboptimal (<10%)
- MTE3 utilization shows significant pipeline stalls

## 失效边界

- Block size is determined by data layout constraints requiring non-power-of-two values
- Memory access patterns are already optimally aligned
- Performance is compute-bound rather than memory-bound

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- addcmul/op_host/addcmul.cpp

### 改动符号

- BLOCK_SIZE

### 代码签名

- BUFFER_NUM

### 原始 diff

- 源 diff SHA-256：`f70957aa008ab6bdde3c94e429aec0c9dd3f75d9147c27c29ef21f1412631055`

````diff
--- Before/op_host/addcmul.cpp
+++ After/op_host/addcmul.cpp
@@ -5,7 +5,7 @@

 namespace optiling
 {
-    const uint32_t BLOCK_SIZE = 32;
+    const uint32_t BLOCK_SIZE = 31;
     const uint32_t BUFFER_NUM = 2;
     static ge::graphStatus TilingFunc(gert::TilingContext *context)
     {
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1839.796753`
- fast duration us：`1725.814575`
- improvement ratio：`0.06195367929318221`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.mte3_ratio_mean: 0.284086 -> 0.144414 (-49.17%)
- memory_ub.ub_to_gm_kb: 10965.5 -> 13062 (+19.12%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 3.48 -> 4.12 (+18.39%)
- cache_l2.aiv_write_hit_rate_pct: 5.07 -> 5.85 (+15.38%)
- cache_l2.aiv_total_hit_rate_pct: 1.9 -> 2.14 (+12.63%)

### Affected metric groups

- aiv_utilization
- memory_ub
- cache_l2
- latency

### Changed metrics top

~~~~json
[
  "aiv_utilization.mte3_ratio_mean",
  "memory_ub.ub_to_gm_kb",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.320909,
  "overall_deviation": 0.128363,
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

- Inspect BLOCK_SIZE constants in op_host/*.cpp files for non-power-of-two values
- Change BLOCK_SIZE from prime numbers (e.g., 31, 61) to nearest power-of-two (e.g., 32, 64)
- Verify memory alignment by checking that (BLOCK_SIZE * element_size) % cache_line_size == 0
- Profile before/after changes focusing on memory_ub and cache_l2 metrics
- Ensure BUFFER_NUM remains compatible with the new BLOCK_SIZE for tiling calculations

## 检索线索

- BLOCK_SIZE
- memory alignment
- power of two
- cache line
- memory_ub
- cache_l2
- Ascend C
- vector operations
- tiling
- memory efficiency
