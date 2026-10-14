---
type: Optimization
title: "triu：内存优化"
description: "triu 算子的内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/Triu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/tensorflow_triu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_kernel/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/Triu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/test_triu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# triu：内存优化

## 来源信息

- experience ID：`4f3d3bd09de99958`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_007_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.management.overhead`
- canonical family：`memory.management.overhead`
- 源记录时间：`2026-04-21T00:17:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant tensor allocation in Compute hot path
~~~~

### Original summary

~~~~text
Avoid redundant tensor allocations followed by immediate frees in the Compute function hot path. This optimization eliminates unnecessary memory management overhead in Ascend C kernels, improving vector unit utilization from 30.0% to 44.1% and memory bandwidth usage from 57.5 Gbps to 87.1 Gbps. Apply this pattern when tensor allocations are not actually needed for computation.
~~~~

### Original root mechanism

~~~~text
Unnecessary tensor allocation and immediate free operations in the Compute function's hot path create memory management overhead and pipeline stalls, reducing vector unit utilization and memory bandwidth.
~~~~

### Original causal chain

~~~~text
Removing redundant AllocTensor/FreeTensor calls → reduces scalar overhead → decreases vector wait ratio from 72.5% to 62.0% → improves vector unit utilization from 30.0% to 44.1% → increases memory bandwidth from 57.5 to 87.1 Gbps → reduces latency from 2173μs to 1436μs
~~~~

## 适用条件

- Ascend C kernels with TQue-based tensor allocations in Compute function hot paths
- Kernels showing high aiv_vec_wait_ratio_mean (>60%) and low vector utilization (<40%)
- Operators with memory bandwidth underutilization despite sufficient compute
- When profiling shows high scalar_ratio_mean relative to vec_ratio_mean

## 失效边界

- Tensor allocations are actually needed for intermediate computation results
- When memory allocation patterns are required for data dependencies
- In kernels where AllocTensor/FreeTensor calls are essential for memory management

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- triu/op_kernel/triu.cpp

### 改动符号

- KernelTriu::Compute

### 代码签名

- AllocTensor
- DeQue
- FreeTensor
- TQue
- queue_api

### 原始 diff

- 源 diff SHA-256：`76b8c55b69dd53c1e3bacf5a8fe7974361fd8a1f80136099ec05a05f2f081c7e`

````diff
--- Before/op_kernel/triu.cpp
+++ After/op_kernel/triu.cpp
@@ -120,6 +120,11 @@
     __aicore__ inline void Compute(int32_t cnt, uint32_t initLength, int32_t adjust){
         auto xLocal = inQueueX.DeQue<DTYPE_X>();
         auto yLocal = outQueueY.AllocTensor<DTYPE_X>();
+
+        // Add redundant tensor allocation and immediate free to increase memory management overhead
+        auto redundantLocal = outQueueY.AllocTensor<DTYPE_X>();
+        outQueueY.FreeTensor(redundantLocal);
+
         uint32_t localOffset=0;
         uint32_t currLength=initLength;
         DTYPE_X scalarZero=0;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2173.043457`
- fast duration us：`1435.528687`
- improvement ratio：`0.3393925545410757`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 1434.99 -> 2172.43 (+51.39%)
- latency.task_duration_us: 1435.53 -> 2173.04 (+51.38%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 49.49 -> 32.69 (-33.95%)
- memory_gm.aiv_main_mem_read_bw_gbps: 87.11 -> 57.54 (-33.95%)
- memory_gm.aiv_main_mem_write_bw_gbps: 87.11 -> 57.54 (-33.95%)

### Affected metric groups

- critical_path
- latency
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1434.99,
      "after": 2172.428,
      "delta": 737.4379999999999,
      "rel": 0.5138976578233994
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1435.529,
      "after": 2173.043,
      "delta": 737.5140000000001,
      "rel": 0.5137576461360238
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 49.49,
      "after": 32.69,
      "delta": -16.800000000000004,
      "rel": -0.3394625176803395
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 87.11,
      "after": 57.54,
      "delta": -29.57,
      "rel": -0.33945586040638276
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 87.11,
      "after": 57.54,
      "delta": -29.57,
      "rel": -0.33945586040638276
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 131.33,
      "after": 86.75,
      "delta": -44.58000000000001,
      "rel": -0.33945023985380346
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 131.33,
      "after": 86.75,
      "delta": -44.58000000000001,
      "rel": -0.33945023985380346
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 44.67,
      "after": 29.51,
      "delta": -15.16,
      "rel": -0.3393776583837027
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.021703,
      "after": 0.014795,
      "delta": -0.006907999999999999,
      "rel": -0.318297009630005
    }
  },
  {
    "arithmetic.aiv_vec_fp16_ratio_mean": {
      "before": 0.418831,
      "after": 0.285529,
      "delta": -0.13330200000000003,
      "rel": -0.31827157015598184
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

- Review Compute function in triu.cpp for redundant AllocTensor/FreeTensor pairs
- Remove lines: 'auto redundantLocal = outQueueY.AllocTensor<DTYPE_X>();' and 'outQueueY.FreeTensor(redundantLocal);'
- Verify tensor allocations are only used when actually needed for computation
- Profile after removal to confirm improvements in aiv_utilization.vec_ratio_mean and memory_gm.aiv_main_mem_read_bw_gbps

## 检索线索

- triu
- memory_management_overhead
- tensor_allocation
- scalar_overhead
- AllocTensor
- FreeTensor
- Compute_hot_path
- TQue
- vector_utilization
- memory_bandwidth
