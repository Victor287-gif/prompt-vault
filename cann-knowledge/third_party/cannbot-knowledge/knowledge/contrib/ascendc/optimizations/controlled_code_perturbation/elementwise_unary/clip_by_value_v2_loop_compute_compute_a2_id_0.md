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

- experience ID：`0477de01d3fb9dae`
- 源 schema version：`v2`
- 算子：`clip_by_value_v2`
- source iteration：`round_012_refactor`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.restructure`
- canonical family：`loop.restructure`
- 源记录时间：`2026-04-20T15:23:21`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Consolidated tail processing into main loop with conditional
~~~~

### Original summary

~~~~text
Applying loop consolidation with conditional processDataNum assignment improves performance by reducing pipeline stalls and memory bandwidth usage. This optimization restructures the main processing loop to handle both regular and tail iterations within a single loop using a conditional assignment, eliminating the separate tail processing block that causes pipeline bubbles.
~~~~

### Original root mechanism

~~~~text
Moving processDataNum assignment inside loop with ternary conditional reduces pipeline stalls by eliminating separate tail processing block
~~~~

### Original causal chain

~~~~text
Loop restructuring with conditional processDataNum assignment -> Reduced aiv_vec_bank_cflt_ratio_mean by 28.57% and aiv_mte3_wait_ratio_mean by 27.00% -> Decreased critical_path.aiv_vector0_time_us_mean from 74946.3us to 56299.6us (24.88% improvement)
~~~~

## 适用条件

- When implementing Ascend C operators with tail processing logic
- When processing loops have separate tail handling blocks outside main loop
- When pipeline_stalls.aiv_vec_bank_cflt_ratio_mean > 5e-06
- When aiv_mte3_wait_ratio_mean > 0.002
- When memory_ub.gm_to_ub_bw_usage_rate_pct > 0.15

## 失效边界

- When loop count is small (loopCount <= 2)
- When tailDataNum equals ubPartDataNum
- When conditional branching overhead outweighs pipeline stall reduction
- When compiler can automatically optimize tail processing

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
- fast duration us：`56310.164062`
- improvement ratio：`0.24876444349500781`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 56299.6 -> 74946.3 (+33.12%)
- latency.task_duration_us: 56310.2 -> 74956.7 (+33.11%)
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
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 56299.553,
      "after": 74946.272,
      "delta": 18646.718999999997,
      "rel": 0.3312054537981855
    }
  },
  {
    "latency.task_duration_us": {
      "before": 56310.164,
      "after": 74956.734,
      "delta": 18646.57,
      "rel": 0.33114039589726646
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 7e-06,
      "after": 5e-06,
      "delta": -1.9999999999999995e-06,
      "rel": -0.28571428571428564
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.003341,
      "after": 0.002439,
      "delta": -0.0009019999999999996,
      "rel": -0.2699790481891648
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 0.19,
      "after": 0.14,
      "delta": -0.04999999999999999,
      "rel": -0.26315789473684204
    }
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

- In clip_by_value_v2.cpp:175-181, replace separate tail processing with conditional assignment inside main loop
- Change: for (uint64_t i = 0; i < loopCount-1; i++) { processDataNum = ubPartDataNum; ... } then tail processing -> for (uint64_t i = 0; i < loopCount; i++) { processDataNum = (i == loopCount - 1) ? tailDataNum : ubPartDataNum; ... }
- Apply to KernelClipByValueV2::Process function in both template instantiations
- Ensure processDataNum is declared as a member variable accessible within the loop

## 检索线索

- loop_restructure
- tail_processing
- pipeline_stalls
- conditional_assignment
- clip_by_value_v2
- processDataNum
- ubPartDataNum
- tailDataNum
- Ascend_C
- vector_operator
