---
type: Optimization
title: "upsample_bicubic2d_aa_grad：并行与 Tiling优化"
description: "upsample_bicubic2d_aa_grad 算子的并行与 Tiling优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/aclnn_upsample_bicubic2d_aa_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/aclnn_upsample_bicubic2d_aa_grad.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_l0.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_l0.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_tiling.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_kernel/upsample_bicubic2d_aa_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/op_kernel/upsample_bicubic2d_aa_grad.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/upsample_bicubic2d_aa_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# upsample_bicubic2d_aa_grad：并行与 Tiling优化

## 来源信息

- experience ID：`997609286fe06217`
- 源 schema version：`v2`
- 算子：`upsample_bicubic2d_aa_grad`
- source iteration：`round_012_degrade`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.core.limitation.const`
- canonical family：`tiling.core.limitation.const`
- 源记录时间：`2026-04-15T02:46:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase MAX_CORE_CONT constant in tiling data structure
~~~~

### Original summary

~~~~text
Setting MAX_CORE_CONT to a higher value (50) in tiling data structures improves performance by reducing memory access overhead and increasing parallel core utilization. This optimization applies to Ascend C operators using compile-time constant arrays for core assignment where the constant value artificially limits parallel execution. The improvement comes from reduced memory bandwidth pressure and better cache utilization.
~~~~

### Original root mechanism

~~~~text
Increasing MAX_CORE_CONT from 16 to 50 reduces tiling data structure size constraints, allowing better parallel core utilization and reducing memory access overhead.
~~~~

### Original causal chain

~~~~text
Increased MAX_CORE_CONT from 16 to 50 → Reduced memory bandwidth usage (GM read/write bandwidth increased by ~55%) → Improved cache hit rates (L2 write hit rate increased 13.4x) → Reduced pipeline stalls (cube wait ratio increased 2.65x) → Latency improved 13.5x (18091μs to 1342μs)
~~~~

## 适用条件

- Operator uses compile-time constant arrays for core assignment (e.g., MAX_CORE_CONT in tiling headers)
- Performance profiling shows high memory bandwidth usage with low cache hit rates
- Block time standard deviation is high (>1.0) indicating load imbalance
- Memory access patterns show excessive GM to L1 transfers
- Cube utilization is low (<0.1) with high pipeline stall ratios

## 失效边界

- MAX_CORE_CONT value already matches or exceeds physical core count
- Memory bandwidth is aazalready saturated at hardware limits
- Operator has memory-bound bottlenecks unrelated to core assignment
- Increasing array sizes would exceed available on-chip memory

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/upsample_bicubic2d_aa_grad_tiling.h

### 改动符号

- MAX_CORE_CONT

### 代码签名

- compile_time_specialization
- constexpr
- coreNum
- core_launch
- ub

### 原始 diff

- 源 diff SHA-256：`e8e7dbce0c8276bf5921249299984bc300fb34b9c26826c5ef0a284c13b2e153`

````diff
--- Before/op_host/upsample_bicubic2d_aa_grad_tiling.h
+++ After/op_host/upsample_bicubic2d_aa_grad_tiling.h
@@ -6,7 +6,7 @@

 namespace optiling {

-constexpr uint16_t MAX_CORE_CONT = 50;
+constexpr uint16_t MAX_CORE_CONT = 16;

 struct UpsampleBicubic2dAAGradCompileInfo {
     uint32_t coreNum;
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`18091.341797`
- fast duration us：`1342.146851`
- improvement ratio：`0.9258127525277001`
- machine confidence：`0.95`

### Profile evidence

- parallelism.block_time_std_us: 196.845 -> 7550.63 (+3735.82%)
- memory_gm.read_main_memory_kb: 23531.6 -> 383116 (+1528.09%)
- memory_ub.ub_to_gm_kb: 80.2 -> 1257.2 (+1467.58%)
- arithmetic.aic_cube_fops: 9.43718e+06 -> 1.47849e+08 (+1466.67%)
- arithmetic.aic_cube_fp_instr: 1152 -> 18048 (+1466.67%)

### Affected metric groups

- parallelism
- memory_gm
- memory_ub
- arithmetic

### Changed metrics top

~~~~json
[
  "parallelism.block_time_std_us",
  "memory_gm.read_main_memory_kb",
  "memory_ub.ub_to_gm_kb",
  "arithmetic.aic_cube_fops",
  "arithmetic.aic_cube_fp_instr",
  "arithmetic.aic_cube_total_instr",
  "memory_gm.write_main_memory_kb",
  "cache_l2.aic_write_hit_rate_pct",
  "latency.task_duration_us",
  "arithmetic.aiv_vec_fops"
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

- Check op_host/*_tiling.h files for MAX_CORE_CONT or similar core assignment constants
- Increase the constant value to match or exceed actual core count (typically 24-48 for Ascend)
- Verify the constant is used in core assignment arrays like coreNum in UpsampleBicubic2dAAGradCompileInfo
- Profile before/after to confirm reduced memory bandwidth and improved cache hit rates
- Monitor parallelism.block_time_std_us and memory_gm.read_main_memory_kb metrics for improvement

## 检索线索

- MAX_CORE_CONT
- tiling.core.limitation.const
- core assignment
- compile-time constant
- memory bandwidth
- parallelism
- UpsampleBicubic2dAAGradCompileInfo
- coreNum
- pipeline stall
- cache hit rate
