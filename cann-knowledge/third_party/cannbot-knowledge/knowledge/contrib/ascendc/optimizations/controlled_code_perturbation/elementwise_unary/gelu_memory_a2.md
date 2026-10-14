---
type: Optimization
title: "gelu：内存优化"
description: "gelu 算子的内存优化经验，关注缓冲区利用率。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/docs/Gelu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/framework/tf_plugin/tensorflow_gelu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_host/gelu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel/gelu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/op_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/Gelu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/st/test_gelu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/gelu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# gelu：内存优化

## 来源信息

- experience ID：`c09e3a6e574ea668`
- 源 schema version：`v2`
- 算子：`gelu`
- source iteration：`round_010_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.buffer.allocation.tiling`
- canonical family：`memory.buffer.allocation`
- 源记录时间：`2026-04-20T20:20:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize TBuffer reservation ratio for better computation buffer utilization
~~~~

### Original summary

~~~~text
For Ascend C vector operators like gelu, adjusting the TBuffer reservation ratio from 1/2 to 1/3 of available UB space improves performance by 8.9%. This optimization reduces the computation buffer size but increases cache hit rates (L2 read hit +50.94%, L2 total hit +50.79%) and decreases vector wait ratio. Apply this pattern when profiling shows high vector wait ratios and low cache hit rates in vector operators with memory-bound characteristics.
~~~~

### Original root mechanism

~~~~text
Increasing TBuffer reservation from 1/2 to 1/3 of UB space reduces available computation buffer size but improves overall performance by better balancing memory and computation resources
~~~~

### Original causal chain

~~~~text
Changed UB memory allocation ratio from 1/2 to 1/3 for TBuffer reservation → Reduced computation buffer size → Increased tile count → Improved L2 cache hit rates (read hit +50.94%, total hit +50.79%) → Reduced vector wait ratio from 0.26218 to 0.232542 → Decreased task duration from 1510.21μs to 1375.97μs
~~~~

## 适用条件

- Vector operators using TBuffer for memory operations
- Profiling shows low L2 cache hit rates (<1% total hit rate)
- High vector wait ratios (>0.25) with memory-bound characteristics
- UB memory bandwidth usage is below peak (<300 Gbps for vector reads)
- Operator has significant GM to UB data movement

## 失效边界

- Computation-bound operators where vector utilization is already high (>0.85)
- Operators with minimal GM memory access
- When UB memory bandwidth is already saturated (>350 Gbps)
- Scalar-dominant operators with low vector usage

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- gelu/op_host/gelu.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- TBuf
- tbuf
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`51f0d71f0a549bb314b61981198964307ade80fb28979c24707d80f5fe34e82f`

````diff
--- Before/op_host/gelu.cpp
+++ After/op_host/gelu.cpp
@@ -36,7 +36,7 @@
         }

         uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
-        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
+        uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 3) / NUM;
         tiling_size = tiling_size <= BYTESIZE ? tiling_size : tiling_size / BYTESIZE * BYTESIZE;

         uint32_t block_size = tiling_size * ALIGN_NUM;
````

## 观测记录

- bottleneck type：`memory_buffer_utilization`
- slow duration us：`1510.210205`
- fast duration us：`1375.967529`
- improvement ratio：`0.08889006017543105`
- machine confidence：`0.85`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 0.53 -> 0.8 (+50.94%)
- cache_l2.aiv_total_hit_rate_pct: 0.63 -> 0.95 (+50.79%)
- cache_l2.aiv_write_hit_rate_pct: 1.04 -> 1.56 (+50.00%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.025336 -> 0.037003 (+46.05%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.001819 -> 0.002614 (+43.71%)

### Affected metric groups

- cache_l2
- pipeline_stalls
- arithmetic
- aiv_utilization

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "arithmetic.aiv_vec_misc_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.951248,
  "overall_deviation": 0.380499,
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

- 1. In the tiling function (optiling::TilingFunc in gelu/op_host/gelu.cpp), locate the TBuffer reservation calculation
- 2. Change the TBuffer reservation ratio from 1/2 to 1/3 of available UB space: uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 3) / NUM;
- 3. Verify the change reduces computation buffer size but maintains sufficient buffer for data movement
- 4. Profile to confirm improved L2 cache hit rates and reduced vector wait ratios
- 5. Monitor UB bandwidth usage to ensure it doesn't become the new bottleneck

## 检索线索

- TBuffer
- UB memory allocation
- tiling ratio
- cache hit rate
- vector wait ratio
- memory buffer optimization
- Ascend C vector operator
- gelu
- memory.buffer.allocation
