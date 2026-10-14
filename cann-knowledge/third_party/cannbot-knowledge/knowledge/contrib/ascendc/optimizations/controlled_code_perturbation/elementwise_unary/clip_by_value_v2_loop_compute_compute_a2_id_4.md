---
type: Optimization
title: "clip_by_value_v2：循环计算优化"
description: "clip_by_value_v2 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline"]
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

- experience ID：`566c8d8507be0f48`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_003_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.restructure`
- canonical family：`loop.restructure`
- 源记录时间：`2026-04-20T14:16:31`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Unified loop with conditional tail handling improves pipeline efficiency
~~~~

### Original summary

~~~~text
Consolidating separate main and tail loops into a single unified loop with conditional tail handling improves pipeline efficiency in Ascend C kernels. This optimization reduces pipeline stalls and improves memory bandwidth utilization by maintaining consistent loop structure throughout all iterations. Apply this pattern when processing tile-based workloads with separate main and tail loops.
~~~~

### Original root mechanism

~~~~text
Moving tail iteration handling inside the main loop reduces pipeline bubbles caused by separate tail processing
~~~~

### Original causal chain

~~~~text
Changed from separate main loop and tail iteration to unified loop with conditional check for last iteration -> Reduced pipeline stall ratios (aiv_vec_bank_cflt_ratio_mean: -28.57%, aiv_mte3_wait_ratio_mean: -27.00%) -> Improved memory bandwidth utilization (gm_to_ub_bw_usage_rate_pct: -26.32%, ub_read_bw_vector_gbps: -25.00%) -> Decreased latency by 24.87%
~~~~

## 适用条件

- Processing tile-based workloads with separate main and tail loops
- Using Ascend C kernel patterns with ubPartDataNum and tailDataNum variables
- When pipeline stalls (aiv_vec_bank_cflt_ratio_mean, aiv_mte3_wait_ratio_mean) are significant
- When memory bandwidth utilization (gm_to_ub_bw_usage_rate_pct) is suboptimal

## 失效边界

- When loop count is very small (1-2 iterations)
- When tail processing requires significantly different operations than main loop
- When conditional branching inside loop would cause excessive overhead

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
- fast duration us：`56311.683594`
- improvement ratio：`0.24874417137386126`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 56300.3 -> 74946.3 (+33.12%)
- latency.task_duration_us: 56311.7 -> 74956.7 (+33.11%)
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
  "memory_ub.gm_to_ub_bw_usage_rate_pct"
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

- Identify loops in clip_by_value_v2/op_kernel/clip_by_value_v2.cpp that separate main and tail processing
- Replace separate loop structures with unified loop using conditional: this->processDataNum = (i == loopCount - 1) ? this->tailDataNum : this->ubPartDataNum
- Apply to KernelClipByValueV2::Process and KernelClipByValueV2<bfloat16_t,bfloat16_t,bfloat16_t,bfloat16_t,IsExistBigCore>::Process methods
- Verify pipeline stall metrics (aiv_vec_bank_cflt_ratio_mean, aiv_mte3_wait_ratio_mean) improve
- Check memory bandwidth utilization (gm_to_ub_bw_usage_rate_pct, ub_read_bw_vector_gbps) increases

## 检索线索

- loop.restructure
- pipeline_stall_bound
- tile_processing
- main_tail_loop
- clip_by_value_v2
- Ascend_C
- vector_kernel
- ub_partition
- conditional_tail
- pipeline_efficiency
