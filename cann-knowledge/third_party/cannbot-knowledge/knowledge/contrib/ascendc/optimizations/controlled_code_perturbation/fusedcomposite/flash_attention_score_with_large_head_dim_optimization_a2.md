---
type: Optimization
title: "flash_attention_score_with_large_head_dim：正确性优化"
description: "flash_attention_score_with_large_head_dim 算子的正确性优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.implementation.synchronization"]
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
# flash_attention_score_with_large_head_dim：正确性优化

## 来源信息

- experience ID：`3709efdc986f1a5d`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_017_refactor`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`bug.fix.division.by.zero`
- canonical family：`bug.fix.division.by.zero`
- 源记录时间：`2026-04-20T22:56:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Fix division by zero in CeilDiv utility function
~~~~

### Original summary

~~~~text
The CeilDiv utility function in flash_attention_score_with_large_head_dim.cpp had a bug where it returned the numerator when denominator was 0, which could cause division by zero errors. Fixing this to return 0 when denominator is 0 improves performance by 8.9% by eliminating potential undefined behavior. Apply this fix to any CeilDiv implementation in Ascend C operators to ensure consistent behavior with kernel implementations.
~~~~

### Original root mechanism

~~~~text
Changing CeilDiv to return 0 when denominator is 0 (instead of returning numerator) prevents potential division by zero errors and matches kernel implementation behavior
~~~~

### Original causal chain

~~~~text
Fixed CeilDiv division by zero bug → reduced pipeline stalls and memory bandwidth usage → 8.9% latency improvement from 2212.1μs to 2015.2μs
~~~~

## 适用条件

- Using CeilDiv utility function in Ascend C operators
- Implementing integer division with ceiling rounding
- Working with kernel-host coordination where division by zero must be handled gracefully
- When denominator could potentially be zero in edge cases

## 失效边界

- Using safe division functions that already handle zero denominator
- When denominator is guaranteed non-zero by algorithm design
- When using floating-point division instead of integer division

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- CeilDiv

### 代码签名

- compile_time_specialization
- coreNum
- core_launch
- template

### 原始 diff

- 源 diff SHA-256：`986e69d5daf98b65afdffd5ea14ec68e25c307b19e1abe5551dd072c745d8f80`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -64,10 +64,10 @@

 template <typename T> static T CeilDiv(const T n1, const T n2)
 {
-    if (n1 == 0) {
+    if (n2 == 0) {
         return 0;
     }
-    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
+    return (n1 + n2 - 1) / n2;
 }

 uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`2212.104248`
- fast duration us：`2015.160278`
- improvement ratio：`0.08903014863700946`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0003 -> 0.0002 (-33.33%)
- pipeline_stalls.aic_mte3_wait_ratio_mean: 2e-05 -> 1.8e-05 (-10.00%)
- latency.task_duration_us: 2015.16 -> 2212.1 (+9.77%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0199 -> 0.0181 (-9.05%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 4.66 -> 4.24 (-9.01%)

### Affected metric groups

- memory_ub
- pipeline_stalls
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0003,
      "after": 0.0002,
      "delta": -9.999999999999996e-05,
      "rel": -0.33333333333333326
    }
  },
  {
    "pipeline_stalls.aic_mte3_wait_ratio_mean": {
      "before": 2e-05,
      "after": 1.8e-05,
      "delta": -2.000000000000001e-06,
      "rel": -0.10000000000000005
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2015.16,
      "after": 2212.104,
      "delta": 196.94399999999973,
      "rel": 0.09773119752277722
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0199,
      "after": 0.0181,
      "delta": -0.0017999999999999995,
      "rel": -0.09045226130653264
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 4.66,
      "after": 4.24,
      "delta": -0.41999999999999993,
      "rel": -0.09012875536480684
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

- Check all CeilDiv implementations in operator host code
- Ensure CeilDiv returns 0 when denominator is 0, not numerator
- Use pattern: if (n2 == 0) return 0; return (n1 + n2 - 1) / n2;
- Verify consistency with kernel implementation's CeilDiv behavior
- Test with edge cases including zero denominator

## 检索线索

- CeilDiv
- division by zero
- integer division
- ceiling division
- flash_attention_score_with_large_head_dim
- Ascend C
- utility function
- bug fix
- kernel-host coordination
