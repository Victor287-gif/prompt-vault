---
type: Optimization
title: "clip_by_value_v2：循环计算优化"
description: "clip_by_value_v2 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise"]
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

- experience ID：`e56e66b3a3158177`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_006_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.restructure`
- canonical family：`loop.restructure`
- 源记录时间：`2026-04-20T14:41:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Consolidated loop structure with conditional tail handling
~~~~

### Original summary

~~~~text
Applying a unified loop structure with conditional tail handling reduces pipeline stalls and improves vector utilization in Ascend C kernels. This optimization consolidates separate tail processing into the main loop using a conditional check (i == loopCount - 1) ? tailDataNum : ubPartDataNum, which reduces control flow overhead and improves instruction cache efficiency. Use this pattern when processing tiled data with varying tail sizes to minimize pipeline bubbles and improve memory bandwidth utilization.
~~~~

### Original root mechanism

~~~~text
Moved tail processing into main loop with conditional check instead of separate tail processing after loop
~~~~

### Original causal chain

~~~~text
Loop restructuring from separate tail processing to conditional tail handling within main loop → reduced pipeline stalls (aiv_vec_bank_cflt_ratio_mean decreased by 28.57%, aiv_mte3_wait_ratio_mean decreased by 27.00%) → improved vector utilization (aiv_utilization.vec_ratio_mean increased by 24.88%) → reduced latency from 74956.7µs to 56310.7µs (24.88% improvement)
~~~~

## 适用条件

- Processing tiled data with separate head/tail loops in Ascend C kernels
- Kernel uses template-based tiling with tileNum, ubPartDataNum, and tailDataNum variables
- Vector operations show low utilization (vec_ratio_mean < 0.005)
- Pipeline stalls indicate control flow overhead (aiv_vec_bank_cflt_ratio_mean > 1e-06)
- Memory bandwidth utilization is suboptimal (gm_to_ub_bw_usage_rate_pct < 0.2)

## 失效边界

- Loop count is small (loopCount <= 2) where overhead outweighs benefits
- Tail processing requires significantly different operations than main loop
- Memory access patterns differ substantially between head and tail tiles
- Compiler already optimizes the loop structure automatically

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- clip_by_value_v2/op_kernel/clip_by_value_v2.cpp

### 改动符号

- KernelClipByValueV2::Process()

### 代码签名

- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`953660c05efbcff774a654a6177492f1526250f2fcec79b761d15660bccc8aaf`

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
@@ -175,17 +171,13 @@
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

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`74956.734375`
- fast duration us：`56310.726562`
- improvement ratio：`0.248756939165948`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 56301.3 -> 74946.3 (+33.12%)
- latency.task_duration_us: 56310.7 -> 74956.7 (+33.11%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 7e-06 -> 5e-06 (-28.57%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.003341 -> 0.002439 (-27.00%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.19 -> 0.14 (-26.32%)

### Affected metric groups

- critical_path
- latency
- pipeline_stalls
- memory_ub

### Changed metrics top

~~~~json
[
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.mte3_ratio_mean",
  "memory_ub.ub_read_bw_vector_gbps",
  "pipeline_stalls.aiv_vec_total_cflt_ratio_mean",
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

- 1. Identify loops in clip_by_value_v2.cpp:58-76 and clip_by_value_v2.cpp:175-193 where tail processing occurs after main loop
- 2. Replace separate tail processing with conditional assignment: this->processDataNum = (i == loopCount - 1) ? this->tailDataNum : this->ubPartDataNum;
- 3. Remove the separate tail processing block after the main loop
- 4. Ensure the loop runs for full loopCount iterations (i < loopCount) instead of loopCount-1
- 5. Verify that all memory operations (CopyIn, Compute, CopyOut) handle variable processDataNum correctly
- 6. Profile the kernel to confirm reduction in pipeline stalls (aiv_vec_bank_cflt_ratio_mean) and improved vector utilization (vec_ratio_mean)

## 检索线索

- loop.restructure
- tail processing
- pipeline stalls
- vector utilization
- Ascend C
- clip_by_value_v2
- tileNum
- ubPartDataNum
- tailDataNum
- processDataNum
- conditional assignment
- control flow
- memory bandwidth
- aiv_vec_bank_cflt_ratio_mean
- aiv_mte3_wait_ratio_mean
- gm_to_ub_bw_usage_rate_pct
