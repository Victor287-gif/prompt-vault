---
type: Optimization
title: "clip_by_value_v2：循环计算优化"
description: "clip_by_value_v2 算子的循环计算优化经验，关注内存带宽。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/docs/ClipByValueV2.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/framework/tf_plugin/tensorflow_clip_by_value_v2_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/op_host/clip_by_value_v2.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/op_host/clip_by_value_v2_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/op_kernel/clip_by_value_v2.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/atk/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/atk/generate_reduce.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/atk/op_clip.yaml
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/ClipByValueV2_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/st/test_clip_by_value_v2.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/clip_by_value_v2/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# clip_by_value_v2：循环计算优化

## 来源信息

- experience ID：`43937d65d5ac6348`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_014_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.restructure`
- canonical family：`loop.restructure`
- 源记录时间：`2026-04-20T15:44:51`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Single-loop with conditional processDataNum assignment
~~~~

### Original summary

~~~~text
Replace separate last-iteration handling with a single loop using conditional processDataNum assignment. This optimization reduces memory bandwidth usage by 8.33% on GM read/write and 7.14-8.33% on UB bandwidth, while decreasing vector stall ratios by 5.7-6.7%. Apply when processing tile-based operations with tail handling in Ascend C vector kernels.
~~~~

### Original root mechanism

~~~~text
Consolidating separate last-iteration handling into a single loop with ternary operator reduces memory bandwidth contention and improves pipeline efficiency by eliminating redundant loop control flow.
~~~~

### Original causal chain

~~~~text
Changed from separate loop for last iteration to single loop with ternary operator for processDataNum assignment → Reduced memory bandwidth contention (GM read/write -8.33%, UB bandwidth -7.14-8.33%) → Decreased vector stall ratios (aiv_vec_total_cflt_ratio_mean -6.7%, aiv_vec_bankgroup_cflt_ratio_mean -6.65%) → Improved overall latency by 6.64% (from 93.6ms to 87.4ms).
~~~~

## 适用条件

- Ascend C vector kernels with tile-based processing (tileNum > 1)
- Operations requiring separate handling for tail data (tailDataNum != ubPartDataNum)
- Kernels using loopCount-1 pattern for main iterations with separate last iteration
- When memory bandwidth is a bottleneck (GM/UB bandwidth usage > 0.1 Gbps)

## 失效边界

- Single-tile operations (tileNum == 1)
- When tailDataNum equals ubPartDataNum (no special handling needed)
- When compiler can optimize the separate loop pattern effectively
- When loopCount is small (< 4) and loop overhead dominates

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- clip_by_value_v2/op_kernel/clip_by_value_v2.cpp

### 改动符号

- KernelClipByValueV2::Process
- KernelClipByValueV2<bfloat16_t,bfloat16_t,bfloat16_t,bfloat16_t,IsExistBigCore>::Process

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`e3bf4d840f4f82486e3d54da95898d1d2cb7dd27fafb39440a77f3a0f4d52c74`

````diff
--- Before/op_kernel/clip_by_value_v2.cpp
+++ After/op_kernel/clip_by_value_v2.cpp
@@ -58,17 +58,13 @@
     }
     __aicore__ inline void Process() {
         uint64_t loopCount = this->tileNum;
-        this->processDataNum = this->ubPartDataNum;
-        for (uint64_t i = 0; i < loopCount-1; i++)
-        {
+        for (uint64_t i = 0; i < loopCount; i++)
+        {
+            this->processDataNum = (i == loopCount - 1) ? this->tailDataNum : this->ubPartDataNum;
             CopyIn(i);
             Compute(i);
             CopyOut(i);
         }
-        this->processDataNum = this->tailDataNum;
-        CopyIn(loopCount-1);
-        Compute(loopCount-1);
-        CopyOut(loopCount-1);
     }
 private:
     __aicore__ inline void CopyIn(uint64_t progress) {
@@ -179,17 +175,13 @@
     }
     __aicore__ inline void Process() {
         uint64_t loopCount = this->tileNum;
-        this->processDataNum = this->ubPartDataNum;
-        for (uint64_t i = 0; i < loopCount-1; i++)
-        {
+        for (uint64_t i = 0; i < loopCount; i++)
+        {
+            this->processDataNum = (i == loopCount - 1) ? this->tailDataNum : this->ubPartDataNum;
             CopyIn(i);
             Compute(i);
             CopyOut(i);
         }
-        this->processDataNum = this->tailDataNum;
-        CopyIn(loopCount-1);
-        Compute(loopCount-1);
-        CopyOut(loopCount-1);
     }
 private:
     __aicore__ inline void CopyIn(uint64_t progress) {
````

## 观测记录

- bottleneck type：`memory_bandwidth`
- slow duration us：`93604.664062`
- fast duration us：`87387.40625`
- improvement ratio：`0.0664203848633219`
- machine confidence：`0.85`

### Profile evidence

- memory_gm.aiv_main_mem_read_bw_gbps: 0.24 -> 0.22 (-8.33%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.24 -> 0.22 (-8.33%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.12 -> 0.11 (-8.33%)
- memory_ub.ub_read_bw_vector_gbps: 0.96 -> 0.89 (-7.29%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 0.14 -> 0.13 (-7.14%)

### Affected metric groups

- memory_gm
- memory_ub
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_gm.aiv_main_mem_read_bw_gbps",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "memory_ub.ub_read_bw_vector_gbps",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
  "aiv_utilization.vec_ratio_mean",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean"
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

- In KernelClipByValueV2::Process() function, replace separate last-iteration handling with single loop
- Move processDataNum assignment inside loop: this->processDataNum = (i == loopCount - 1) ? this->tailDataNum : this->ubPartDataNum
- Remove separate CopyIn/Compute/CopyOut calls for loopCount-1 iteration
- Ensure ternary operator doesn't introduce branch misprediction penalties (use conditional move if available)
- Verify memory access patterns remain consistent with UB/GM bandwidth improvements

## 检索线索

- loop_restructure
- tile_processing
- tail_handling
- memory_bandwidth
- vector_kernel
- clip_by_value_v2
- Ascend_C
- processDataNum
- ubPartDataNum
- tailDataNum
