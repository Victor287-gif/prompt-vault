---
type: Optimization
title: "motion_compensation：内存与搬运优化"
description: "motion_compensation 算子的内存与搬运优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.synchronization", "topic.data_movement.memory_access"]
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
# motion_compensation：内存与搬运优化

## 来源信息

- experience ID：`9d3c82fc9b7b6978`
- 源 schema version：`v2`
- 算子：`motion_compensation`
- source iteration：`round_000_degrade`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`cache.over.sync`
- canonical family：`cache.over.sync`
- 源记录时间：`2026-04-14T21:10:51`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating per-element cache synchronization in bulk DataCopy operations
~~~~

### Original summary

~~~~text
Avoid inserting per-element cache synchronization operations (DataCacheCleanAndInvalid) after bulk DataCopy operations. These redundant cache operations serialize memory access and reduce vector unit utilization from 83% to 33%. Use DataCopy's built-in cache management instead of manual per-element synchronization.
~~~~

### Original root mechanism

~~~~text
Replacing per-element cache synchronization calls with efficient bulk DataCopy operations reduces serialized memory access
~~~~

### Original causal chain

~~~~text
Removing per-element DataCacheCleanAndInvalid calls → reduces scalar unit utilization from 77% to 23% → reduces MTE2/MTE3 wait ratios by 68%/61% → improves vector unit utilization from 33% to 83% → reduces critical path time from 1281µs to 544µs → reduces total latency from 1853µs to 1167µs
~~~~

## 适用条件

- When using DataCopy operations with Ascend C vector kernels
- When writing data from UB to GM memory that will be read by subsequent operations
- When performance profiles show high scalar unit utilization (>70%) with low vector unit utilization (<40%)
- When pipeline stalls show high MTE2/MTE3 wait ratios (>0.1) despite adequate memory bandwidth

## 失效边界

- When explicit cache synchronization is required for data consistency between different hardware units
- When using non-coherent memory regions that require manual cache management
- When DataCopy operations involve non-contiguous memory patterns requiring per-element synchronization

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- motion_compensation/op_kernel/motion_compensation.cpp

### 改动符号

- KernelMotionCompensation::CopyOut

### 代码签名

- DataCopy
- FreeTensor
- TQue
- datacopy
- queue_api

### 原始 diff

- 源 diff SHA-256：`b81695c0c93eb590d08dc99c7188a124ac087376209c78003a5260e63c95edac`

````diff
--- Before/op_kernel/motion_compensation.cpp
+++ After/op_kernel/motion_compensation.cpp
@@ -152,8 +152,18 @@
             }
         }else{
             DataCopy(outGm[offset], cLocal, this->processDataNum);
+            // Degradation: Add unnecessary cache operations for each element
+            for(int32_t i = 0; i < this->processDataNum; i++) {
+                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(outGm[offset + i]);
+            }
             DataCopy(outGm[N + offset], cLocal[processDataNum], this->processDataNum);
+            for(int32_t i = 0; i < this->processDataNum; i++) {
+                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(outGm[N + offset + i]);
+            }
             DataCopy(outGm[N + N + offset], cLocal[processDataNum + processDataNum], this->processDataNum);
+            for(int32_t i = 0; i < this->processDataNum; i++) {
+                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE>(outGm[N + N + offset + i]);
+            }
         }
         outQueue.FreeTensor(cLocal);
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`1852.99707`
- fast duration us：`1166.543335`
- improvement ratio：`0.370455920364731`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.226662 -> 0.770589 (+239.97%)
- critical_path.aiv_vector0_time_us_mean: 543.736 -> 1280.99 (+135.59%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.019219 -> 0.0061 (-68.26%)
- aiv_utilization.mte2_ratio_mean: 0.096866 -> 0.036199 (-62.63%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 3.15 -> 1.19 (-62.22%)

### Affected metric groups

- aiv_utilization
- critical_path
- pipeline_stalls
- memory_ub

### Changed metrics top

~~~~json
[
  "aiv_utilization.scalar_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "aiv_utilization.mte2_ratio_mean",
  "memory_ub.gm_to_ub_bw_usage_rate_pct"
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

- In KernelMotionCompensation::CopyOut function in motion_compensation/op_kernel/motion_compensation.cpp, remove per-element DataCacheCleanAndInvalid calls after DataCopy operations
- Replace loops of DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE> with DataCopy's implicit cache management
- Ensure DataCopy operations use appropriate cache hints instead of manual synchronization
- Verify that removing cache operations doesn't break data consistency by checking subsequent memory access patterns

## 检索线索

- DataCopy
- DataCacheCleanAndInvalid
- cache synchronization
- vector utilization
- scalar utilization
- MTE wait
- pipeline stall
- memory bandwidth
- Ascend C
- motion_compensation
