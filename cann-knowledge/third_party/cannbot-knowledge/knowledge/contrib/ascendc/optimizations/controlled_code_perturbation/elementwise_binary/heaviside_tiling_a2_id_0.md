---
type: Optimization
title: "heaviside：Tiling优化"
description: "heaviside 算子的Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/docs/Heaviside.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_host/heaviside_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/op_kernel/heaviside.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/heaviside/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# heaviside：Tiling优化

## 来源信息

- experience ID：`b3f841578585427e`
- 源 schema version：`v2`
- 算子：`heaviside`
- source iteration：`round_011_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.misalignment`
- canonical family：`tiling.misalignment`
- 源记录时间：`2026-04-20T20:49:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Memory block size misalignment for inefficient work distribution
~~~~

### Original summary

~~~~text
Applying aligned memory block sizes (512 bytes) instead of misaligned sizes (511 bytes) improves performance by 9.75% for vector operators. This optimization reduces cache misses and improves memory bandwidth utilization by ensuring efficient memory access patterns and avoiding integer division truncation of fractional element counts.
~~~~

### Original root mechanism

~~~~text
Using a blockSize of 511 bytes causes misaligned memory access and fractional element count per block that truncates in integer division, reducing performance
~~~~

### Original causal chain

~~~~text
apply tiling.misalignment in op_host/heaviside.cpp -> improve cache_l2.aiv_write_hit_rate_pct -> 9.8% latency improvement
~~~~

## 适用条件

- When implementing vector operators with block-based memory access patterns
- When using constexpr uint32_t blockSize for tiling calculations
- When blockNum calculation uses integer division: (size * DataTypeSize + blockSize - 1) / blockSize
- When working with 2-byte data types (fp16) requiring aligned memory access

## 失效边界

- Avoid using block sizes that are not powers of two or create misaligned memory access
- Avoid block sizes that cause fractional element counts per block after integer division truncation
- When blockSize is close to but not equal to memory alignment boundaries (e.g., 511 vs 512)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/heaviside.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- compile_time_specialization
- constexpr

### 原始 diff

- 源 diff SHA-256：`7aff02e9354e011567f4f73fe5f3aa5937e7e0ecf8b3935eceb9bac5c434ada4`

````diff
--- Before/op_host/heaviside.cpp
+++ After/op_host/heaviside.cpp
@@ -14,7 +14,7 @@
     } else {
         DataTypeSize = 2;
     }
-    constexpr uint32_t blockSize = 512;
+    constexpr uint32_t blockSize = 511;
     uint32_t blockNum = (size * DataTypeSize + blockSize - 1) / blockSize;
     aivNum = std::min(aivNum, blockNum);
     if (aivNum == 0) {
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1622.632446`
- fast duration us：`1464.369263`
- improvement ratio：`0.09753483198868564`
- machine confidence：`0.9`

### Profile evidence

- cache_l2.aiv_write_hit_rate_pct: 0 -> 5.38 (+100.00%)
- cache_l2.aiv_total_hit_rate_pct: 0 -> 2.06 (+100.00%)
- cache_l2.aiv_read_hit_rate_pct: 0 -> 1.62 (+100.00%)
- aiv_utilization.mte3_ratio_mean: 0.210919 -> 0.284268 (+34.78%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 18.22 -> 20.77 (+14.00%)

### Affected metric groups

- cache_l2
- aiv_utilization
- memory_ub
- latency

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_write_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_read_hit_rate_pct",
  "aiv_utilization.mte3_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.scalar_ratio_mean",
  "memory_ub.ub_to_gm_bw_usage_rate_pct"
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

- Check op_host/heaviside.cpp for blockSize definition in TilingFunc
- Ensure blockSize is a power of two that aligns with memory access patterns
- Verify blockNum calculation: (size * DataTypeSize + blockSize - 1) / blockSize doesn't truncate fractional elements
- Use 512-byte blocks instead of 511-byte blocks for 2-byte data types
- Monitor cache_l2 hit rates and aiv_utilization.mte3_ratio_mean to validate alignment improvements

## 检索线索

- blockSize
- tiling
- misalignment
- memory alignment
- cache hit rate
- MTE3 utilization
- vector operator
- heaviside
- constexpr
- integer division truncation
