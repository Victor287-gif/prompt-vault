---
type: Optimization
title: "reshape：循环计算优化"
description: "reshape 算子的循环计算优化经验，关注数据搬运。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.data_movement.memory_access"]
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
# reshape：循环计算优化

## 来源信息

- experience ID：`37918f18f0e21f7b`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_015_refactor`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.redundancy.elimination`
- canonical family：`loop.redundancy.elimination`
- 源记录时间：`2026-04-21T01:31:10`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant memory copy loops in Ascend C reshape kernels
~~~~

### Original summary

~~~~text
In Ascend C reshape kernels, avoid performing identical memory copy operations multiple times within the same execution loop. Each redundant CopyIn/CopyOut pair doubles the GM-to-UB and UB-to-GM data transfers, increasing memory bandwidth usage by 100% and latency by 59.8%. This optimization applies to any operator performing tile-based memory transfers where verification and actual work use identical copy patterns.
~~~~

### Original root mechanism

~~~~text
Redundant memory copy operations double the GM-to-UB and UB-to-GM data transfers, causing unnecessary memory bandwidth consumption and increased latency.
~~~~

### Original causal chain

~~~~text
Removing redundant CopyIn/CopyOut pairs → GM-to-UB transfers reduced from 924,768 KB to 462,384 KB (-50%) → UB-to-GM transfers reduced from 924,768 KB to 462,384 KB (-50%) → Memory bandwidth usage reduced from 2.77 Gbps to 2.21 Gbps (-20%) → Task duration reduced from 318,904 μs to 199,536 μs (-37.4%).
~~~~

## 适用条件

- Operator uses tile-based memory transfers with CopyIn/CopyOut patterns in Ascend C
- Kernel contains loops that process the same data multiple times with identical operations
- Performance profiling shows memory_ub.gm_to_ub_kb and memory_ub.ub_to_gm_kb metrics are unexpectedly high
- Memory bandwidth usage (aiv_main_mem_read_bw_gbps, aiv_main_mem_write_bw_gbps) exceeds expected levels

## 失效边界

- Data dependencies require multiple passes over the same data with different processing
- Memory transfers serve different purposes (e.g., different UB buffers, different data transformations)
- Operator already uses optimal single-pass memory transfer patterns

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- reshape/op_kernel/reshape.cpp

### 改动符号

- MyReshape<DATA, SHAPE, 1>::exec()

### 代码签名

- InitBuffer
- TBuf
- queue_api
- tbuf
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`e9461f013fda5e4d00926235b1b8138028ff9105ca5e00ae5621ea6141e1182c`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -57,20 +57,13 @@
         p->InitBuffer(q_data, 1, this->ti.tileLength * sizeof(DATA));
     }
     __aicore__ inline void exec() {
-        // Degrade: process each tile twice - once for "verification" and once for "actual" work
-        // This adds redundant computation without changing functional behavior
+        // Process each tile once (removed redundant verification pass)
         for(uint32_t i = 0; i < TI.tileNumber; i++) {
-            // First pass: "verification" copy (redundant)
-            CopyIn(i * TI.tileLength, TI.tileLength);
-            CopyOut(i * TI.tileLength, TI.tileLength);
-            // Second pass: "actual" copy (functional)
             CopyIn(i * TI.tileLength, TI.tileLength);
             CopyOut(i * TI.tileLength, TI.tileLength);
         }
-        // Handle reminder with same redundant pattern
+        // Handle reminder
         if (TI.reminder > 0) {
-            CopyIn(TI.tileNumber * TI.tileLength, TI.reminder);
-            CopyOut(TI.tileNumber * TI.tileLength, TI.reminder);
             CopyIn(TI.tileNumber * TI.tileLength, TI.reminder);
             CopyOut(TI.tileNumber * TI.tileLength, TI.reminder);
         }
````

## 观测记录

- bottleneck type：`memory_transfer_bound`
- slow duration us：`318903.53125`
- fast duration us：`199535.59375`
- improvement ratio：`0.3743073556825031`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 462384 -> 924768 (+100.00%)
- memory_ub.gm_to_ub_kb: 462384 -> 924768 (+100.00%)
- latency.task_duration_us: 199536 -> 318904 (+59.82%)
- critical_path.aiv_vector0_time_us_mean: 199508 -> 318849 (+59.82%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 1.13 -> 1.42 (+25.66%)

### Affected metric groups

- memory_ub
- latency
- critical_path
- aiv_utilization

### Changed metrics top

~~~~json
[
  "memory_ub.ub_to_gm_kb: 462383.8 → 924767.5 (+100.00%)",
  "memory_ub.gm_to_ub_kb: 462383.9 → 924767.6 (+100.00%)",
  "latency.task_duration_us: 199535.594 → 318903.531 (+59.82%)",
  "critical_path.aiv_vector0_time_us_mean: 199507.535 → 318848.548 (+59.82%)",
  "memory_ub.gm_to_ub_bw_usage_rate_pct: 1.13 → 1.42 (+25.66%)"
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

- Inspect the exec() method in reshape/op_kernel/reshape.cpp for redundant CopyIn/CopyOut pairs
- Look for patterns where the same memory region is copied multiple times without data transformation
- Remove duplicate copy operations that serve no functional purpose
- Verify that scalar_ratio_mean decreases (from 0.211 to 0.168) and mte3_ratio_mean increases (from 0.666 to 0.684) after optimization
- Monitor memory_ub.gm_to_ub_kb and memory_ub.ub_to_gm_kb metrics - they should halve with this optimization

## 检索线索

- reshape
- Ascend C
- memory copy
- redundant loops
- CopyIn
- CopyOut
- GM-to-UB
- UB-to-GM
- memory bandwidth
- tile processing
