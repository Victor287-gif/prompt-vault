---
type: Optimization
title: "flash_attention_score_with_large_head_dim：向量化计算优化"
description: "flash_attention_score_with_large_head_dim 算子的向量化计算优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/docs/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/.keep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/framework/tf_plugin/tensorflow_flash_attention_score_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/tiling_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_host/tiling_type.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/op_kernel/flash_attention_score_with_large_head_dim.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/FlashAttentionScore_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/st/test_flash_attention_score.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/flash_attention_score_with_large_head_dim/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# flash_attention_score_with_large_head_dim：向量化计算优化

## 来源信息

- experience ID：`7688466eef526b44`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_014_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vector.processing.base.reduction`
- canonical family：`vector.processing.base.reduction`
- 源记录时间：`2026-04-20T22:12:17`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Halve vector processing base size for large head dimensions in ComputeBmm1Tail
~~~~

### Original summary

~~~~text
When head dimension (dSizeAlign16) exceeds 64, halving the vector processing base size from 8192 to 4096 improves performance by 7.67% by reducing vector unit wait ratios and increasing ProcessVec2 loop iterations. This optimization applies to FlashAttentionScoreWithLargeHeadDim operators where large head dimensions cause inefficient vector instruction scheduling.
~~~~

### Original root mechanism

~~~~text
Reducing the vec2S1BaseSize calculation from 8192 (64*128) to 4096 (32*128) for dSizeAlign16 > 64 improves vector instruction scheduling and reduces pipeline stalls
~~~~

### Original causal chain

~~~~text
Reduce vec2S1BaseSize numerator from 64*128 to 32*128 → Decreases aiv_vec_wait_ratio_mean from 0.1685 to 0.1300 (-22.9%) → Reduces aiv_mte2_wait_ratio_mean from 0.3956 to 0.3587 (-9.3%) → Improves overall latency from 2212 μs to 2042 μs
~~~~

## 适用条件

- FlashAttentionScoreWithLargeHeadDim operators with dSizeAlign16 > 64
- Vector processing operators using vec2S1BaseSize calculation with large head dimensions
- Operators showing high aiv_vec_wait_ratio_mean (>0.15)
- Operators with ProcessVec2 loop iterations limited by large vec2S1BaseSize

## 失效边界

- Head dimension (dSizeAlign16) ≤ 64
- Vector units already fully utilized (vec_ratio_mean > 0.9)
- Memory bandwidth is the primary bottleneck
- Small batch sizes where vector processing base size already matches s1RealSize

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- flash_attention_score_with_large_head_dim/op_kernel/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1::ComputeBmm1Tail

### 代码签名

- bool
- bool_control
- contiguous_access

### 原始 diff

- 源 diff SHA-256：`1429f07ad8c551393ec6ec865eb115200619e2ffaa212eb3f9ee645efe04f82a`

````diff
--- Before/op_kernel/flash_attention_score_with_large_head_dim.cpp
+++ After/op_kernel/flash_attention_score_with_large_head_dim.cpp
@@ -453,14 +453,14 @@

      bool notSecondLast = true;
      bool notLast = true;
-     multiCoreInnerLimit += 2;
+      multiCoreInnerLimit += 2;
      for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
              multiCoreInnerIdx++) {
-         if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
-             notSecondLast = false;
-         } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
-             notLast = false;
-         }
+          if (multiCoreInnerIdx == multiCoreInnerLimit - 2) {
+              notSecondLast = false;
+          } else if (multiCoreInnerIdx == multiCoreInnerLimit - 1) {
+              notLast = false;
+          }

          int64_t s2LoopLimit;
          bool notLastTwoLoop = notSecondLast && notLast;
@@ -558,14 +558,14 @@
          extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
      }

-     extraInfo.vec1S1BaseSize = Min(s2BaseNratioSize / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize);
+      extraInfo.vec1S1BaseSize = Min(s2BaseNratioSize / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize);
      extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

-     if (dSizeAlign16 > 64) {
-         extraInfo.vec2S1BaseSize = 64 * 128 / dSizeAlign16;
-     } else {
-         extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
-     }
+      if (dSizeAlign16 > 64) {
+          extraInfo.vec2S1BaseSize = 32 * 128 / dSizeAlign16;
+      } else {
+          extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
+      }
      return;
  }
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2212.104248`
- fast duration us：`2042.400879`
- improvement ratio：`0.07671580991421706`
- machine confidence：`0.85`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0003 -> 0.0002 (-33.33%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.130045 -> 0.168514 (+29.58%)
- arithmetic.aiv_vec_misc_ratio_mean: 0.017123 -> 0.019838 (+15.86%)
- aiv_utilization.mte2_ratio_mean: 0.138635 -> 0.16034 (+15.66%)
- aiv_utilization.icache_miss_rate_mean: 0.009509 -> 0.01061 (+11.58%)

### Affected metric groups

- memory_ub
- pipeline_stalls
- arithmetic
- aiv_utilization

### Changed metrics top

~~~~json
[
  "memory_ub.ub_read_bw_scalar_gbps",
  "pipeline_stalls.aiv_vec_wait_ratio_mean",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "aiv_utilization.icache_miss_rate_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.661803,
  "overall_deviation": 0.264721,
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

- In ComputeBmm1Tail function, locate vec2S1BaseSize calculation for dSizeAlign16 > 64
- Replace 'extraInfo.vec2S1BaseSize = 64 * 128 / dSizeAlign16;' with 'extraInfo.vec2S1BaseSize = 32 * 128 / dSizeAlign16;'
- Ensure the reduction maintains integer division and proper alignment
- Verify that s1RealSize remains the fallback for dSizeAlign16 ≤ 64
- Profile aiv_vec_wait_ratio_mean and aiv_mte2_wait_ratio_mean to confirm improvement

## 检索线索

- vec2S1BaseSize
- vector processing base size
- FlashAttentionScoreWithLargeHeadDim
- ComputeBmm1Tail
- dSizeAlign16
- head dimension
- vector wait ratio
- ProcessVec2
- vector unit scheduling
- Ascend C optimization
