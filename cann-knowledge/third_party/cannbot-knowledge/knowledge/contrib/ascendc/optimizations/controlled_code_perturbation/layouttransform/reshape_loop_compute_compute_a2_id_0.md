---
type: Optimization
title: "reshape：循环计算优化"
description: "reshape 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.data_movement.memory_access"]
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
# reshape：循环计算优化

## 来源信息

- experience ID：`259bacdc5b63f879`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_014_refactor`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.unrolling`
- canonical family：`constant.change`
- 源记录时间：`2026-04-21T01:25:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase chunk size from 16 to 32 in memory copy loops
~~~~

### Original summary

~~~~text
Increasing the CHUNK_SIZE constant from 16 to 32 in CopyIn and CopyOut methods reduces loop iteration overhead and improves memory bandwidth utilization. This optimization is effective for memory-bound operators using GM-to-UB and UB-to-GM transfers where pipeline stalls from memory wait cycles are limiting performance.
~~~~

### Original root mechanism

~~~~text
Larger chunk sizes reduce loop iteration overhead and memory transaction setup costs, improving memory bandwidth utilization by allowing more efficient burst transfers.
~~~~

### Original causal chain

~~~~text
Increased CHUNK_SIZE from 16 to 32 → Reduced loop iterations by half → Lowered aiv_mte2_wait_ratio_mean by 43.9% and aiv_mte3_wait_ratio_mean by 49.16% → Improved memory bandwidth utilization from 2.77 Gbps to 4.11 Gbps → Reduced task_duration_us from 318903μs to 150298μs (2.12x speedup).
~~~~

## 适用条件

- Operator uses GM-to-UB or UB-to-GM memory transfers with chunked copy loops
- Memory bandwidth utilization is below theoretical maximum (e.g., < 5 Gbps)
- Pipeline stall metrics show high aiv_mte2_wait_ratio_mean or aiv_mte3_wait_ratio_mean (> 0.1)
- Operator has memory-bound characteristics with gm_to_ub_kb and ub_to_gm_kb metrics showing high transfer volumes
- Copy loops use while(remaining > 0) pattern with fixed CHUNK_SIZE constant

## 失效边界

- Operator is compute-bound with high vector utilization
- Memory bandwidth is already saturated (> 80% of theoretical maximum)
- Cache hit rates are low (< 90%) indicating different bottleneck
- CHUNK_SIZE approaches or exceeds hardware burst transfer limits
- Data access patterns are non-contiguous or strided

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- reshape/op_kernel/reshape.cpp

### 改动符号

- MyReshape::CopyIn
- MyReshape::CopyOut

### 代码签名

- AllocTensor
- DeQue
- queue_api

### 原始 diff

- 源 diff SHA-256：`d7ddd0c26789c4cdd5981452416bb023d8a3a65c7aefed07f200df0977ef67d6`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -24,7 +24,7 @@
     __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.AllocTensor<DATA>();
         // Degrade: split single copy into multiple smaller chunks
-        const uint32_t CHUNK_SIZE = 16;  // Small chunk size to increase overhead
+        const uint32_t CHUNK_SIZE = 32;  // Small chunk size to increase overhead
         uint32_t remaining = length;
         uint32_t current_offset = 0;
         while (remaining > 0) {
@@ -38,7 +38,7 @@
     __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.DeQue<DATA>();
         // Degrade: split single copy into multiple smaller chunks
-        const uint32_t CHUNK_SIZE = 16;  // Small chunk size to increase overhead
+        const uint32_t CHUNK_SIZE = 32;  // Small chunk size to increase overhead
         uint32_t remaining = length;
         uint32_t current_offset = 0;
         while (remaining > 0) {
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`318903.53125`
- fast duration us：`150297.765625`
- improvement ratio：`0.5287046053209861`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 150268 -> 318849 (+112.19%)
- latency.task_duration_us: 150298 -> 318904 (+112.18%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.068775 -> 0.034964 (-49.16%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.146568 -> 0.082228 (-43.90%)
- memory_ub.ub_to_gm_kb: 647184 -> 924768 (+42.89%)

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
  "pipeline_stalls.aiv_mte3_wait_ratio_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "memory_ub.ub_to_gm_kb"
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

- Identify copy loops using while(remaining > 0) pattern in CopyIn/CopyOut methods
- Check current CHUNK_SIZE value (typically defined as const uint32_t CHUNK_SIZE)
- Increase CHUNK_SIZE by powers of 2 (16→32→64→128) while monitoring performance
- Verify CHUNK_SIZE doesn't exceed hardware burst transfer limits (check Ascend C documentation)
- Profile aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean metrics to confirm reduction
- Monitor memory_ub.gm_to_ub_bw_usage_rate_pct and memory_ub.ub_to_gm_bw_usage_rate_pct for improvement

## 检索线索

- CHUNK_SIZE
- CopyIn
- CopyOut
- memory copy
- loop unrolling
- pipeline stalls
- mte2_wait
- mte3_wait
- memory bandwidth
- GM-to-UB
- UB-to-GM
- reshape
- Ascend C
- queue_api
