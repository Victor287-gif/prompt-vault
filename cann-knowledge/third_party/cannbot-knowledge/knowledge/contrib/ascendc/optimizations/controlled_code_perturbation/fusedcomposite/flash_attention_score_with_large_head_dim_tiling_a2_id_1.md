---
type: Optimization
title: "flash_attention_score_with_large_head_dim：Tiling优化"
description: "flash_attention_score_with_large_head_dim 算子的Tiling优化经验，关注内存受限。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.tiling"]
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
# flash_attention_score_with_large_head_dim：Tiling优化

## 来源信息

- experience ID：`5784daff8717fe76`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.block.size.increase`
- canonical family：`tiling.block.size.reduction`
- 源记录时间：`2026-04-18T22:00:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase computational tile sizes to improve data reuse efficiency
~~~~

### Original summary

~~~~text
Increasing computational tile sizes in the MatchTemplate function reduces loop iteration overhead and improves data reuse. This optimization applies when smaller tile sizes cause excessive loop iterations and poor cache utilization. Larger tiles increase data locality and reduce memory bandwidth pressure.
~~~~

### Original root mechanism

~~~~text
Larger computational tile sizes (s1BasicBlock from 32→64, s2BasicBlock from 64→128, dBasicBlock from 64→128) in MatchTemplate function reduce loop iterations and improve data reuse efficiency.
~~~~

### Original causal chain

~~~~text
Increasing tile sizes (s1BasicBlock 32→64, s2BasicBlock 64→128, dBasicBlock 64→128) → reduces loop iterations by 50% → decreases AIC cube instructions from 2184 to 1456 → reduces AIC cube wait ratio from 0.7496 to 0.5464 → improves task duration from 1774μs to 1278μs
~~~~

## 适用条件

- Operator uses MatchTemplate function with tile-based computation
- s1BasicBlock, s2BasicBlock, or dBasicBlock parameters are set below hardware-optimal values
- Performance shows high loop iteration overhead (high AIC cube wait ratios >0.5)
- UB bandwidth utilization is low while memory access patterns show poor locality

## 失效边界

- Tile sizes already approach or exceed available UB capacity (16KB)
- Increasing tiles would cause UB overflow or exceed hardware limits
- Problem dimensions are small enough that loop overhead is negligible
- Memory bandwidth is already saturated with current tile sizes

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/flash_attention_score_with_large_head_dim.cpp

### 改动符号

- MatchTemplate

### 代码签名

- contiguous_access
- template
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`89f357804f2ca3eb38e3725b284e7d2975576f8deb2fd7fb9845bc027a6ced3a`

````diff
--- Before/op_host/flash_attention_score_with_large_head_dim.cpp
+++ After/op_host/flash_attention_score_with_large_head_dim.cpp
@@ -311,12 +311,12 @@
     // s1.i: 默认64，当按64切分时，如果核外BNGS1out超vector核数时，S1.i设置为128
     // s2.i: 1024
     // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
-    s1BasicBlock = std::min(64L, alignedS1);
+    s1BasicBlock = std::min(32L, alignedS1); // DEGRADATION: Reduced from 64 to 32
     if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
-        s1BasicBlock = std::min(128L, alignedS1);
-    }
-    s2BasicBlock = std::min(128L, alignedS2);
-    dBasicBlock = std::min(128L, alignedD);
+        s1BasicBlock = std::min(64L, alignedS1); // DEGRADATION: Reduced from 128 to 64
+    }
+    s2BasicBlock = std::min(64L, alignedS2); // DEGRADATION: Reduced from 128 to 64
+    dBasicBlock = std::min(64L, alignedD); // DEGRADATION: Reduced from 128 to 64
     apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * s1BasicBlock * s2BasicBlock * sizeof(float);
     LOG_PRINT("[%s]final basic block: [%ld, %ld, %ld].\n", templateName, s1BasicBlock,
                 s2BasicBlock, dBasicBlock);
````

## 观测记录

- bottleneck type：`memory_bound`
- slow duration us：`1774.355469`
- fast duration us：`1278.345581`
- improvement ratio：`0.2795436972274466`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_write_bw_scalar_gbps: 0.004 -> 0.0114 (+185.00%)
- critical_path.aiv_vector0_time_us_mean: 711.013 -> 1345.2 (+89.20%)
- critical_path.aic_time_us_mean: 732.978 -> 1358.12 (+85.29%)
- parallelism.block_time_mean_us: 732.978 -> 1358.12 (+85.29%)
- critical_path.aiv_vector1_time_us_mean: 731.715 -> 1337.41 (+82.78%)

### Affected metric groups

- memory_ub
- critical_path
- parallelism
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.004,
      "after": 0.0114,
      "delta": 0.0074,
      "rel": 1.85
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 711.013,
      "after": 1345.204,
      "delta": 634.1909999999999,
      "rel": 0.8919541555498983
    }
  },
  {
    "critical_path.aic_time_us_mean": {
      "before": 732.978,
      "after": 1358.125,
      "delta": 625.147,
      "rel": 0.8528864440678985
    }
  },
  {
    "parallelism.block_time_mean_us": {
      "before": 732.978,
      "after": 1358.125,
      "delta": 625.147,
      "rel": 0.8528864440678985
    }
  },
  {
    "critical_path.aiv_vector1_time_us_mean": {
      "before": 731.715,
      "after": 1337.414,
      "delta": 605.699,
      "rel": 0.827779941643946
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

- Identify tile size parameters in MatchTemplate function (s1BasicBlock, s2BasicBlock, dBasicBlock)
- Check if current values are below typical optimal ranges (64-128 for s1BasicBlock, 128 for s2BasicBlock/dBasicBlock)
- Increase tile sizes incrementally while monitoring UB usage
- Verify UB capacity constraints: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp <= UB_SIZE
- Test with s1BasicBlock=64, s2BasicBlock=128, dBasicBlock=128 as starting point
- Monitor AIC cube wait ratio and memory bandwidth metrics for improvement

## 检索线索

- tile_size
- block_size
- MatchTemplate
- flash_attention
- data_reuse
- loop_iterations
- UB_capacity
- s1BasicBlock
- s2BasicBlock
- dBasicBlock
- memory_bound
- cube_wait
