---
type: Optimization
title: "reshape：内存与 Tiling优化"
description: "reshape 算子的内存与 Tiling优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.tiling"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/docs/Reshape.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/framework/tf_plugin/tensorflow_reshape_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_host/reshape.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_host/reshape_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/op_kernel/reshape.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reshape/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reshape：内存与 Tiling优化

## 来源信息

- experience ID：`c63f9063db6af2f6`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_011_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.buffer.overestimation`
- canonical family：`tiling.buffer.overestimation`
- 源记录时间：`2026-04-21T00:57:52`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Unified buffer per-iteration requirement overestimation
~~~~

### Original summary

~~~~text
When tiling memory-bound operators like Reshape, overestimating the unified buffer size per iteration (ub_size_per_it) reduces tile size and increases loop iterations, causing performance degradation. The optimal ub_size_per_it should be calculated as 2×type_size for 64-bit types, 2×type_size for 32-bit types, 2×type_size for 16-bit types, and 2×type_size for 8-bit types to maximize tile size while fitting within UB capacity. This optimization applies to operators with simple memory access patterns where tiling calculations use ub_size_per_it parameter.
~~~~

### Original root mechanism

~~~~text
Doubling the ub_size_per_it parameter in tiling calculation reduces effective tile size by half, forcing more loop iterations and increasing kernel launch overhead.
~~~~

### Original causal chain

~~~~text
Reducing ub_size_per_it from 32 to 16 for 64-bit types → increases tile size by 2× → reduces loop iterations by 2× → decreases pipeline stalls (aiv_mte2_wait_ratio_mean from 0.082228 to 0.040669, aiv_mte3_wait_ratio_mean from 0.034964 to 0.019165) → reduces scalar overhead (scalar_ratio_mean from 0.211257 to 0.28522) → decreases task duration from 318903.531μs to 230772.828μs (27.6% improvement).
~~~~

## 适用条件

- Operator uses unified buffer (UB) for data transfer
- Tiling calculation includes ub_size_per_it parameter
- Operator is memory-bound with simple copy/reshape patterns
- Performance shows high pipeline stall ratios (aiv_mte2_wait_ratio_mean > 0.04, aiv_mte3_wait_ratio_mean > 0.02)
- Scalar utilization ratio is below 0.3 (scalar_ratio_mean < 0.3)

## 失效边界

- Operator is compute-bound with high vector utilization
- UB capacity is fully utilized with current tile size
- Memory bandwidth is already saturated (gm_to_ub_bw_usage_rate_pct > 90%)
- Operator has complex data dependencies requiring smaller tiles

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reshape/op_host/reshape.cpp

### 改动符号

- optiling::TilingFunc

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`c15c8924a3c2107acac9701d8cc52945fdf1b6a14843bc91ba448be79ca42c5f`

````diff
--- Before/op_host/reshape.cpp
+++ After/op_host/reshape.cpp
@@ -53,21 +53,21 @@
         switch(context->GetInputTensor(0)->GetDataType()) {
             case DT_UINT64:
             case DT_INT64:
-                ub_size_per_it = 16;
+                ub_size_per_it = 32;  // Degrade: increased from 16 to 32 (4 * type_sz instead of 2 * type_sz)
                 break;
             case DT_UINT32:
             case DT_INT32:
             case DT_FLOAT:
-                ub_size_per_it = 8;
+                ub_size_per_it = 16;  // Degrade: increased from 8 to 16
                 break;
             case DT_UINT16:
             case DT_INT16:
             case DT_FLOAT16:
-                ub_size_per_it = 4;
+                ub_size_per_it = 8;   // Degrade: increased from 4 to 8
                 break;
             case DT_UINT8:
             case DT_INT8:
-                ub_size_per_it = 2;
+                ub_size_per_it = 4;   // Degrade: increased from 2 to 4
                 break;
             default:break;
         }
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`318903.53125`
- fast duration us：`230772.828125`
- improvement ratio：`0.276355369222648`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.040669 -> 0.082228 (+102.19%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.019165 -> 0.034964 (+82.44%)
- latency.task_duration_us: 230773 -> 318904 (+38.19%)
- critical_path.aiv_vector0_time_us_mean: 230742 -> 318849 (+38.18%)
- aiv_utilization.scalar_ratio_mean: 0.28522 -> 0.211257 (-25.93%)

### Affected metric groups

- pipeline_stalls
- latency
- critical_path
- aiv_utilization

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "latency.task_duration_us",
  "critical_path.aiv_vector0_time_us_mean",
  "aiv_utilization.scalar_ratio_mean"
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

- 1. In op_host/reshape.cpp, locate the TilingFunc implementation
- 2. Identify ub_size_per_it calculation based on data type
- 3. Set ub_size_per_it = 2 × type_size (e.g., 16 for DT_UINT64/DT_INT64, 8 for DT_FLOAT/DT_INT32, 4 for DT_FLOAT16/DT_INT16, 2 for DT_UINT8/DT_INT8)
- 4. Verify tile size increases by checking tiling calculation logic
- 5. Profile to confirm reduction in pipeline stalls (aiv_mte2_wait_ratio_mean, aiv_mte3_wait_ratio_mean) and increase in scalar utilization (scalar_ratio_mean)

## 检索线索

- ub_size_per_it
- tiling
- buffer
- overestimation
- reshape
- memory-bound
- pipeline_stalls
- scalar_overhead
- tile_size
- loop_iterations
