---
type: Optimization
title: "isamax：并行切核优化"
description: "isamax 算子的并行切核优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.implementation.pipeline", "topic.programming_model", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/docs/aclnnIsamax.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_host/isamax.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_host/isamax_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_kernel/isamax.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_kernel/isamax_aiv.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# isamax：并行切核优化

## 来源信息

- experience ID：`7a75a4d3fbff5164`
- 源 schema version：`v2`
- 算子：`isamax`
- source iteration：`round_010_degrade`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`core.parallelization.reduction`
- canonical family：`core.parallelization.reduction`
- 源记录时间：`2026-04-21T05:15:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Parallel reduction scaling for vector operations
~~~~

### Original summary

~~~~text
Applying parallel reduction scaling in the GetNeedVecCoreNum function enables efficient utilization of multiple vector cores for reduction operations. This optimization distributes computation across available cores when tensor length exceeds core capacity, reducing pipeline stalls and improving throughput. Use this pattern for vector reduction operators where tensor size exceeds single-core processing capacity.
~~~~

### Original root mechanism

~~~~text
Distributing reduction operations across multiple vector cores reduces pipeline stalls by enabling concurrent computation of independent work items.
~~~~

### Original causal chain

~~~~text
Parallel reduction scaling applied -> vec_ratio_mean increased from 0.020277 to 0.557483 (27.4x) -> mte2_ratio_mean increased from 0.011225 to 0.765972 (68.2x) -> pipeline stalls reduced (vec_wait_ratio_mean increased from 0.009605 to 0.717566) -> latency improved from 29233.5μs to 1067.08μs (27.4x speedup)
~~~~

## 适用条件

- Operator performs vector reduction operations (like isamax, isamin, sum, max, min)
- Tensor length exceeds single-core processing capacity (tensorLen > MAX_VECCORE_NUM * elements)
- GetNeedVecCoreNum function determines parallelization strategy
- Pipeline stalls (aiv_vec_wait_ratio_mean) indicate underutilized vector cores
- Memory bandwidth utilization (gm_to_ub_kb) shows inefficient data transfer patterns

## 失效边界

- Tensor length is smaller than vector core capacity (tensorLen ≤ elements)
- Memory bandwidth is already saturated (gm_to_ub_bw_usage_rate_pct > 90%)
- Operator has data dependencies requiring sequential execution
- Vector core utilization (vec_ratio_mean) is already high (>0.8)
- Cache hit rates (aiv_read_hit_rate_pct) are already optimal (>70%)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- isamax/op_host/isamax.cpp

### 改动符号

- GetNeedVecCoreNum

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`71c5c633a07bb165416220cecd99377365d3443cf8c326dc6474dfa497c93c43`

````diff
--- Before/op_host/isamax.cpp
+++ After/op_host/isamax.cpp
@@ -29,13 +29,9 @@
 static uint32_t GetNeedVecCoreNum(uint32_t tensorLen, uint32_t elements)
 {
     uint32_t needVecCoreNum = 1;
-    if (tensorLen > MAX_VECCORE_NUM * elements) {
-        needVecCoreNum = MAX_VECCORE_NUM;
-    } else if (tensorLen > elements) {
-        needVecCoreNum = tensorLen / elements;
-    } else {
-        needVecCoreNum = 1;
-    }
+    // Degradation: Force single-core execution regardless of tensor size
+    // This prevents parallel execution across multiple cores
+    needVecCoreNum = 1;
     return needVecCoreNum;
 }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`29233.503906`
- fast duration us：`1067.081299`
- improvement ratio：`0.9634980020721707`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.gm_to_ub_kb: 39322.1 -> 1.57286e+06 (+3899.95%)
- arithmetic.aiv_vec_fops: 2.02508e+07 -> 8.09968e+08 (+3899.69%)
- critical_path.aiv_vector0_time_us_mean: 1062.48 -> 29228.6 (+2650.98%)
- latency.task_duration_us: 1067.08 -> 29233.5 (+2639.58%)
- cache_l2.aiv_read_hit_rate_pct: 8.38 -> 48.96 (+484.25%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  "memory_ub.gm_to_ub_kb",
  "arithmetic.aiv_vec_fops",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "memory_ub.ub_read_bw_scalar_gbps",
  "aiv_utilization.mte3_ratio_mean",
  "pipeline_stalls.aiv_vec_wait_ratio_mean"
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 1.0,
  "overall_deviation": 0.4,
  "metrics_checked": 2,
  "metrics_wrong": 0,
  "performance_metrics_checked": 1,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- 1. Identify reduction operators using GetNeedVecCoreNum function in op_host/isamax.cpp
- 2. Analyze tensor length (tensorLen) and element size (elements) parameters
- 3. When tensorLen > MAX_VECCORE_NUM * elements, set needVecCoreNum = MAX_VECCORE_NUM
- 4. When tensorLen > elements but ≤ MAX_VECCORE_NUM * elements, calculate needVecCoreNum = tensorLen / elements
- 5. Monitor aiv_vec_wait_ratio_mean metric - values <0.1 indicate potential parallelization benefit
- 6. Verify improvement by checking vec_ratio_mean (>0.5) and mte2_ratio_mean (>0.7)
- 7. Validate with gm_to_ub_kb reduction and latency.task_duration_us improvement

## 检索线索

- parallel reduction
- vector core scaling
- GetNeedVecCoreNum
- isamax
- pipeline stalls
- vec_wait_ratio
- tensorLen
- MAX_VECCORE_NUM
- core parallelization
- reduction optimization
