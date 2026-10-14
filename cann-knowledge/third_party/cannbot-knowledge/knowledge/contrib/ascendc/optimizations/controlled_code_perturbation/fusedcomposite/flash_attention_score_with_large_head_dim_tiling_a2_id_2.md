---
type: Optimization
title: "flash_attention_score_with_large_head_dim：Tiling优化"
description: "flash_attention_score_with_large_head_dim 算子的Tiling优化经验，关注内存带宽。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model"]
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

- experience ID：`e75747d2efa756b7`
- 源 schema version：`v2`
- 算子：`flash_attention_score_with_large_head_dim`
- source iteration：`round_016_refactor`
- 源 category：`math`
- 计算分区：`fusedcomposite`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`tiling.vectorization`
- canonical family：`untitled`
- 源记录时间：`2026-04-20T22:41:48`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce vectorization ratio and block sizes for improved memory efficiency
~~~~

### Original summary

~~~~text
In FlashAttentionScoreWithLargeHeadDim, reducing the nRatio parameter from 8 to 4 and s2BasicBlock from 128 to 64 improves performance by 7.4%. This optimization reduces memory bandwidth pressure on the vector unit (AIV) while maintaining sufficient parallelism. The change decreases memory subsystem contention, particularly in UB (Unified Buffer) bandwidth usage, leading to better overall throughput.
~~~~

### Original root mechanism

~~~~text
Reducing vectorization ratio from 8 to 4 and block sizes from 128 to 64 decreases memory bandwidth pressure while maintaining sufficient parallelism, improving overall performance by reducing memory subsystem contention.
~~~~

### Original causal chain

~~~~text
Reduced nRatio from 8 to 4 and s2BasicBlock from 128 to 64 -> decreased memory bandwidth pressure on AIV vector unit -> reduced UB read/write bandwidth usage by 7.3-7.7% -> improved AIV vector utilization from 0.235 to 0.253 -> reduced task duration from 2212.1µs to 2049.1µs
~~~~

## 适用条件

- FlashAttention operators with large head dimensions where AIV vector unit memory bandwidth is a bottleneck
- Operators showing high UB (Unified Buffer) bandwidth usage (>40 Gbps)
- When aic_cube_wait_ratio_mean > 0.7 indicating memory subsystem contention
- When aiv_vec_ratio_mean < 0.25 indicating underutilized vector units

## 失效边界

- Operators already limited by compute-bound bottlenecks with cube_ratio_mean > 0.4
- When memory bandwidth usage is already low (<20 Gbps)
- Small tensor sizes where block size reductions would hurt parallelism
- When aiv_num (available vector units) is already fully utilized

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 原始 diff

- 源 diff SHA-256：`5c4ac56d309358459dda29b2fe8bae6c425162d5ca2310210cf4d97959d2cb13`

````diff
--- original_flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim.cpp	2026-04-11 14:18:48.409809000 +0800
+++ flash_attention_score_with_large_head_dim/op_host/flash_attention_score_with_large_head_dim.cpp	2026-04-20 21:39:59.677635392 +0800
@@ -201,7 +201,7 @@
     int64_t s1BasicBlock = std::numeric_limits<int64_t>::max();
     int64_t s2BasicBlock = std::numeric_limits<int64_t>::max();
     int64_t dBasicBlock = std::numeric_limits<int64_t>::max();
-    int64_t nRatio = 8L;
+    int64_t nRatio = 4L;

     const char *templateName = "FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1";
     int64_t s2sizeLimitMin = 1024;
@@ -315,7 +315,7 @@
     if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) > aivNum) {
         s1BasicBlock = std::min(128L, alignedS1);
     }
-    s2BasicBlock = std::min(128L, alignedS2);
+    s2BasicBlock = std::min(64L, alignedS2);
     dBasicBlock = std::min(128L, alignedD);
     apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * s1BasicBlock * s2BasicBlock * sizeof(float);
     LOG_PRINT("[%s]final basic block: [%ld, %ld, %ld].\n", templateName, s1BasicBlock,
@@ -336,7 +336,7 @@
 {
     auto &multiCoreParams = tilingData.multiCoreParams;
     int64_t totalSize = bSize * n2Size * gSize * tilingData.coreParams.get_s1OuterSize();
-    actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
+    actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum / 2));
     multiCoreParams.set_totalSize(totalSize);
     multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
 }
--- original_flash_attention_score_with_large_head_dim/op_kernel/flash_attention_score_with_large_head_dim.cpp	2026-04-11 14:18:48.411612000 +0800
+++ flash_attention_score_with_large_head_dim/op_kernel/flash_attention_score_with_large_head_dim.cpp	2026-04-20 22:05:24.645726041 +0800
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

- bottleneck type：`memory_bandwidth`
- slow duration us：`2212.104248`
- fast duration us：`2049.061035`
- improvement ratio：`0.07370503137336767`
- machine confidence：`0.85`

### Profile evidence

- memory_ub.ub_read_bw_scalar_gbps: 0.0003 -> 0.0002 (-33.33%)
- latency.task_duration_us: 2049.06 -> 2212.1 (+7.96%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0196 -> 0.0181 (-7.65%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 4.58 -> 4.24 (-7.42%)
- memory_gm.aiv_main_mem_read_bw_gbps: 12.67 -> 11.73 (-7.42%)

### Affected metric groups

- memory_ub
- latency
- memory_gm
- pipeline_stalls

### Changed metrics top

~~~~json
[
  "memory_ub.ub_read_bw_scalar_gbps",
  "latency.task_duration_us",
  "memory_ub.ub_write_bw_scalar_gbps",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "memory_gm.aiv_main_mem_read_bw_gbps"
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

- Examine op_host/flash_attention_score_with_large_head_dim.cpp:201 and reduce nRatio from 8 to 4
- Modify op_host/flash_attention_score_with_large_head_dim.cpp:315 to reduce s2BasicBlock from std::min(128L, alignedS2) to std::min(64L, alignedS2)
- Adjust op_host/flash_attention_score_with_large_head_dim.cpp:336 to use aivNum/2 instead of full aivNum for actualUsedAivNum calculation
- Update op_kernel/flash_attention_score_with_large_head_dim.cpp:558 to reduce vec2S1BaseSize from 64*128/dSizeAlign16 to 32*128/dSizeAlign16
- Verify performance improvement by monitoring aiv_vec_ratio_mean and memory_ub bandwidth metrics

## 检索线索

- flash_attention
- vectorization_ratio
- memory_bandwidth
- ub_bandwidth
- aiv_utilization
- block_size
- nratio
- s2basicblock
- memory_subsystem
- bandwidth_bottleneck
