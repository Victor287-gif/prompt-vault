---
type: Optimization
title: "motion_compensation：向量化计算优化"
description: "motion_compensation 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/docs/MotionCompensation.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/tf_plugin/tensorflow_motion_compensation_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_host/motion_compensation.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_host/motion_compensation_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_kernel/motion_compensation.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/opp_kernel_aicpu/.gitkeep
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# motion_compensation：向量化计算优化

## 来源信息

- experience ID：`8c65d229830739c5`
- 源 schema version：`v2`
- 算子：`motion_compensation`
- source iteration：`round_002_degrade`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`vectorization.break.scalar.loop`
- canonical family：`vectorization.break.scalar.loop`
- 源记录时间：`2026-04-15T00:12:00`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Using vectorized DataCopy for contiguous memory transfers in motion compensation
~~~~

### Original summary

~~~~text
Using vectorized DataCopy operations for contiguous memory transfers between GM and UB memory significantly improves performance by enabling parallel memory operations. This optimization reduces pipeline stalls and improves L2 cache hit rates. Apply this technique when copying contiguous data blocks between global memory and unified buffer in Ascend C operators.
~~~~

### Original root mechanism

~~~~text
Vectorized DataCopy operations enable parallel memory transfers and reduce pipeline stalls by leveraging hardware memory bandwidth optimization.
~~~~

### Original causal chain

~~~~text
Replacing scalar loops with vectorized DataCopy -> reduces pipeline stalls (aiv_mte2_wait_ratio_mean from 0.001044 to 0) -> improves L2 cache hit rates (aiv_read_hit_rate_pct from 2.66% to 40.3%) -> reduces critical path time (aiv_vector0_time_us_mean from 12618.089us to 6985.028us) -> decreases overall latency (task_duration_us from 13184.684us to 7587.272us).
~~~~

## 适用条件

- Operator performs contiguous memory transfers between GM and UB memory
- Data transfers involve tensor operations with DataCopy API
- Memory access patterns show low L2 cache hit rates (<10%)
- Pipeline stall metrics show MTE wait ratios >0.001
- Operator uses TQue APIs for tensor queue management

## 失效边界

- Memory transfers involve non-contiguous or strided access patterns
- Data blocks are smaller than vector register width
- Memory bandwidth is already saturated (>80% utilization)
- Operator has high arithmetic intensity that dominates memory operations

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- motion_compensation/op_kernel/motion_compensation.cpp

### 改动符号

- KernelMotionCompensation::OnlyCopy

### 代码签名

- AllocTensor
- DataCopy
- DeQue
- EnQue
- FreeTensor
- GetValue
- SetValue
- TQue
- contiguous_access
- datacopy
- queue_api
- scalar_value_access
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`6d0ad10f66abfa8292aa27201ad2e7577f8182c82e59ee0276bc83427d1343d4`

````diff
--- Before/op_kernel/motion_compensation.cpp
+++ After/op_kernel/motion_compensation.cpp
@@ -172,15 +172,24 @@
     {
         int offset = N + N + N + globalBufferIndex2;
         auto inLocal = inQueueOther.AllocTensor<float>();
-        DataCopy(inLocal, pointsGm[offset + progress * this->tileDataNum2], this->processDataNum2);
+        // Degradation: Replace vectorized DataCopy with scalar loop
+        for (int32_t i = 0; i < this->processDataNum2; i++) {
+            inLocal.SetValue(i, pointsGm.GetValue(offset + progress * this->tileDataNum2 + i));
+        }
         inQueueOther.EnQue(inLocal);
         auto outLocal = outQueueOther.AllocTensor<float>();
         inLocal = inQueueOther.DeQue<float>();
-        DataCopy(outLocal, inLocal, this->processDataNum2);
+        // Degradation: Replace vectorized DataCopy with scalar loop
+        for (int32_t i = 0; i < this->processDataNum2; i++) {
+            outLocal.SetValue(i, inLocal.GetValue(i));
+        }
         inQueueOther.FreeTensor(inLocal);
         outQueueOther.EnQue(outLocal);
         outLocal = outQueueOther.DeQue<float>();
-        DataCopy(outGm[offset + progress * this->tileDataNum2], outLocal, this->processDataNum2);
+        // Degradation: Replace vectorized DataCopy with scalar loop
+        for (int32_t i = 0; i < this->processDataNum2; i++) {
+            outGm.SetValue(offset + progress * this->tileDataNum2 + i, outLocal.GetValue(i));
+        }
         outQueueOther.FreeTensor(outLocal);
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`13184.683594`
- fast duration us：`7587.271973`
- improvement ratio：`0.4245389418026864`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 2.66 -> 40.3 (+1415.04%)
- cache_l2.aiv_total_hit_rate_pct: 3.84 -> 48.35 (+1159.11%)
- cache_l2.aiv_write_hit_rate_pct: 8.8 -> 73.83 (+738.98%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.001044 -> 0 (-100.00%)
- critical_path.aiv_vector0_time_us_mean: 6985.03 -> 12618.1 (+80.64%)

### Affected metric groups

- cache_l2
- pipeline_stalls
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "memory_ub.ub_write_bw_scalar_gbps",
  "memory_ub.ub_read_bw_scalar_gbps",
  "memory_ub.ub_to_gm_bw_usage_rate_pct",
  "aiv_utilization.mte2_ratio_mean"
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

- Replace scalar loops with DataCopy API for contiguous memory transfers in motion_compensation.cpp:172-192
- Use DataCopy(inLocal, pointsGm[offset + progress * tileDataNum2], processDataNum2) instead of per-element SetValue/GetValue loops
- Apply to KernelMotionCompensation::OnlyCopy function for GM↔UB transfers
- Ensure tensor dimensions (processDataNum2) are aligned with vector width
- Verify DataCopy usage with TQue.AllocTensor/DeQue/FreeTensor patterns

## 检索线索

- DataCopy
- vectorization
- memory_transfer
- GM_UB
- pipeline_stall
- cache_hit_rate
- motion_compensation
- TQue
- AllocTensor
- contiguous_access
