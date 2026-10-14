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

- experience ID：`c31a27a4f3980c15`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_005_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.restructure`
- canonical family：`loop.restructure`
- 源记录时间：`2026-04-20T14:35:24`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Refactored loop to use conditional inside loop instead of separate tail iteration
~~~~

### Original summary

~~~~text
When implementing tile-based loops in Ascend C kernels, avoid handling the final iteration separately outside the main loop. Instead, use a conditional check inside the loop body to handle tail data. This optimization reduces pipeline stalls by maintaining uniform loop structure and improving instruction cache efficiency. Apply this pattern when processing data in tiles where the last tile may have fewer elements.
~~~~

### Original root mechanism

~~~~text
Moving the tail iteration handling from outside the loop to a conditional inside the loop body reduces pipeline stalls by eliminating the separate code path for the final iteration.
~~~~

### Original causal chain

~~~~text
Changed loop structure from N-1 iterations with ubPartDataNum plus separate tail iteration to single loop with conditional data size selection -> Reduced pipeline stall ratios (aiv_vec_bank_cflt_ratio_mean decreased by 28.6%, aiv_mte3_wait_ratio_mean decreased by 27.0%) -> Improved vector unit utilization (vec_ratio_mean increased by 32.9%) -> Reduced task duration by 24.9% (from 74957us to 56310us)
~~~~

## 适用条件

- Implementing tile-based loops in Ascend C kernels with variable-sized final tile
- Using ubPartDataNum and tailDataNum pattern for data processing
- Processing data in KernelClipByValueV2::Process() or similar kernel functions
- When loopCount > 1 and tailDataNum differs from ubPartDataNum

## 失效边界

- When loopCount is 1 (single iteration)
- When tailDataNum equals ubPartDataNum (uniform tile sizes)
- When compiler can optimize the separate tail iteration effectively
- When the conditional check inside loop would add significant overhead

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- clip_by_value_v2/op_kernel/clip_by_value_v2.cpp

### 改动符号

- KernelClipByValueV2::Process

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
- fast duration us：`56310.347656`
- improvement ratio：`0.24876199416204892`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 56299.2 -> 74946.3 (+33.12%)
- latency.task_duration_us: 56310.3 -> 74956.7 (+33.11%)
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
  {
    "metric": "critical_path.aiv_vector0_time_us_mean",
    "before": 56299.22,
    "after": 74946.272,
    "delta": 18647.051999999996,
    "rel": 0.3312133276446813
  },
  {
    "metric": "latency.task_duration_us",
    "before": 56310.348,
    "after": 74956.734,
    "delta": 18646.386,
    "rel": 0.33113604625565446
  },
  {
    "metric": "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
    "before": 7e-06,
    "after": 5e-06,
    "delta": -1.9999999999999995e-06,
    "rel": -0.28571428571428564
  },
  {
    "metric": "pipeline_stalls.aiv_mte3_wait_ratio_mean",
    "before": 0.003341,
    "after": 0.002439,
    "delta": -0.0009019999999999996,
    "rel": -0.2699790481891648
  },
  {
    "metric": "memory_ub.gm_to_ub_bw_usage_rate_pct",
    "before": 0.19,
    "after": 0.14,
    "delta": -0.04999999999999999,
    "rel": -0.26315789473684204
  }
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

- In KernelClipByValueV2::Process() method, replace separate tail iteration with conditional inside loop
- Change: for (uint64_t i = 0; i < loopCount-1; i++) { processDataNum = ubPartDataNum; ... } followed by separate tail iteration
- To: for (uint64_t i = 0; i < loopCount; i++) { processDataNum = (i == loopCount - 1) ? tailDataNum : ubPartDataNum; ... }
- Apply to both vector and scalar processing paths in clip_by_value_v2.cpp
- Ensure the conditional check is placed before CopyIn(), Compute(), and CopyOut() calls

## 检索线索

- loop restructuring
- tail iteration optimization
- pipeline stall reduction
- Ascend C tile processing
- ubPartDataNum
- tailDataNum
- clip_by_value_v2
- KernelClipByValueV2::Process
- vector processing
- scalar processing
