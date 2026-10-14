---
type: Optimization
title: "upsample_bilinear2d_grad：流水优化"
description: "upsample_bilinear2d_grad 算子的流水优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.synchronization", "topic.implementation.pipeline"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/docs/aclnnUpsampleBilinear2dBackwardV2.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/aclnn_upsample_bilinear2d_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/aclnn_upsample_bilinear2d_grad.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/resize_bilinear_v2_grad_l0.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/resize_bilinear_v2_grad_l0.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_bilinear2d_grad_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_bilinear2d_grad_l0.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_bilinear2d_grad_l0.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_bilinear2d_grad_tiling.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_bilinear2d_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_util.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_host/upsample_util.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_kernel/upsample_bilinear2d_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/op_kernel/upsample_bilinear2d_grad.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bilinear2d_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# upsample_bilinear2d_grad：流水优化

## 来源信息

- experience ID：`6eef2f97b48527f9`
- 源 schema version：`v2`
- 算子：`upsample_bilinear2d_grad`
- source iteration：`round_019_degrade`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.function.call.overhead`
- canonical family：`pipeline.function.call.overhead`
- 源记录时间：`2026-04-15T05:24:21`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant matmulH.End() calls in batch processing loops
~~~~

### Original summary

~~~~text
Remove unnecessary matmulH.End() calls within batch processing loops to reduce function call overhead and pipeline synchronization costs. This optimization improves performance by decreasing AIC wait ratios (cube_wait_ratio_mean from 0.626585 to 0.542029, mte1_wait_ratio_mean from 0.588209 to 0.507101, mte2_wait_ratio_mean from 0.516259 to 0.444429) and reducing block time standard deviation from 20.688us to 8.35us. Apply this technique when batch processing loops contain redundant function calls that don't contribute to computation.
~~~~

### Original root mechanism

~~~~text
Redundant matmulH.End() calls inside batch processing loops increase function call overhead and pipeline synchronization costs, causing AIC wait ratios to rise and reducing overall throughput.
~~~~

### Original causal chain

~~~~text
Eliminating redundant matmulH.End() calls → reduces function call overhead → decreases AIC wait ratios (cube_wait_ratio_mean: -13.5%, mte1_wait_ratio_mean: -13.8%, mte2_wait_ratio_mean: -13.9%) → reduces block time standard deviation from 20.688us to 8.35us → improves task duration from 2162.223us to 1864.517us (13.8% improvement).
~~~~

## 适用条件

- Batch processing loops contain redundant function calls like matmulH.End()
- AIC wait ratios (aic_cube_wait_ratio_mean, aic_mte1_wait_ratio_mean, aic_mte2_wait_ratio_mean) are high (>0.5)
- Block time standard deviation is high (>10us) indicating load imbalance
- Operator uses Ascend C APIs with frequent synchronization calls in loops

## 失效边界

- Function calls are necessary for correct synchronization between compute units
- Batch processing requires explicit synchronization between iterations
- matmulH.End() calls serve a functional purpose beyond pipeline synchronization

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/upsample_bilinear2d_grad.cpp

### 改动符号

- UpSampleBilinear2dGradND::calculateHeightExtension

### 代码签名

- ub

### 原始 diff

- 源 diff SHA-256：`ac018298518468d0cc3ea999dccdd4a46ea5d14d939ad12607870efff96416a9`

````diff
--- Before/op_kernel/upsample_bilinear2d_grad.cpp
+++ After/op_kernel/upsample_bilinear2d_grad.cpp
@@ -699,6 +699,9 @@
         xIndex += middleSize;
         tensorCIndexWithOffset += cubeSize;
         matmulH.End();
+        // Redundant End() call to increase function call overhead
+        matmulH.End();
+        matmulH.End();
     }
 }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`2162.223145`
- fast duration us：`1864.517334`
- improvement ratio：`0.13768505424078237`
- machine confidence：`0.95`

### Profile evidence

- parallelism.block_time_std_us: 8.35 -> 20.688 (+147.76%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.00079 -> 0.001863 (+135.82%)
- parallelism.block_time_cv: 0.0045 -> 0.0097 (+115.56%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.02 -> 0.04 (+100.00%)
- memory_ub.ub_to_gm_kb: 94.1 -> 179.6 (+90.86%)

### Affected metric groups

- parallelism
- pipeline_stalls
- memory_gm
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "parallelism.block_time_std_us": {
      "before": 8.35,
      "after": 20.688,
      "delta": 12.338,
      "rel": 1.4776047904191616
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.00079,
      "after": 0.001863,
      "delta": 0.0010730000000000002,
      "rel": 1.358227848101266
    }
  },
  {
    "parallelism.block_time_cv": {
      "before": 0.0045,
      "after": 0.0097,
      "delta": 0.005200000000000001,
      "rel": 1.155555555555556
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.02,
      "after": 0.04,
      "delta": 0.02,
      "rel": 1.0
    }
  },
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 94.1,
      "after": 179.6,
      "delta": 85.5,
      "rel": 0.9086078639744952
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

- 1. Search for batch processing loops in Ascend C kernels (e.g., in op_kernel/upsample_bilinear2d_grad.cpp)
- 2. Identify redundant function calls like matmulH.End() inside loops
- 3. Remove duplicate calls that don't contribute to computation
- 4. Verify that only one matmulH.End() call remains per iteration for proper synchronization
- 5. Profile before/after to confirm reduction in AIC wait ratios and block time variance

## 检索线索

- matmulH.End
- batch processing loop
- function call overhead
- pipeline synchronization
- AIC wait ratios
- block time std
- Ascend C
- upsample_bilinear2d_grad
- redundant calls
- pipeline stalls
