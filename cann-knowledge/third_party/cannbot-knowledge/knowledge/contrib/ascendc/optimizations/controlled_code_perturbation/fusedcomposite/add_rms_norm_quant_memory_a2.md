---
type: Optimization
title: "add_rms_norm_quant：内存与搬运优化"
description: "add_rms_norm_quant 算子的内存与搬运优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/docs/aclnnAddRmsNormQuant.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_host/add_rms_norm_quant_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_host/add_rms_norm_quant_proto.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_host/add_rms_norm_quant_tiling.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_host/add_rms_norm_quant_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant_single_n.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant_split_d.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_kernel/reduce_common.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/op_kernel/rms_norm_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/norm/add_rms_norm_quant/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# add_rms_norm_quant：内存与搬运优化

## 来源信息

- experience ID：`46c1982e6a8f82bf`
- 源 schema version：`v2`
- 算子：`add_rms_norm_quant`
- source iteration：`round_009_degrade`
- 源 category：`norm`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`cache.over.sync`
- canonical family：`cache.over.sync`
- 源记录时间：`2026-04-14T23:14:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Remove unnecessary memory barrier in row processing loop
~~~~

### Original summary

~~~~text
Avoid inserting unnecessary memory barriers (SetFlag/WaitFlag with MTE3_S) between independent row processing iterations in SubProcess loops. This optimization reduces pipeline stalls by allowing memory operations to overlap, improving vector unit utilization from 69.0% to 85.1% and reducing task duration by 18.7%.
~~~~

### Original root mechanism

~~~~text
Removing forced memory synchronization (MTE3_S barrier) between consecutive row processing iterations allows pipeline parallelism and reduces memory operation serialization.
~~~~

### Original causal chain

~~~~text
apply cache.over.sync in op_kernel/add_rms_norm_quant.cpp -> improve cache_l2.aiv_write_hit_rate_pct -> 18.7% latency improvement
~~~~

## 适用条件

- When processing consecutive independent rows in SubProcess loops
- When memory operations between iterations have no data dependencies
- When using MTE3_S barriers between row processing iterations
- When vector unit utilization is below 70% and aiv_vec_wait_ratio_mean is above 25%

## 失效边界

- When there are actual data dependencies between row processing iterations
- When memory operations between iterations require strict ordering
- When using shared buffers that need explicit synchronization
- When working with atomic operations or race conditions

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/add_rms_norm_quant.cpp

### 改动符号

- KernelAddRmsNormQuant::SubProcess

### 代码签名

- SetFlag
- WaitFlag
- pipebarrier
- ub

### 原始 diff

- 源 diff SHA-256：`4222b9a3b762c91943438a1ef86ce111e2469e92256f99aa4a48e84c78a58be2`

````diff
--- Before/op_kernel/add_rms_norm_quant.cpp
+++ After/op_kernel/add_rms_norm_quant.cpp
@@ -471,6 +471,12 @@
             CopyIn(gmBias);
             Compute(iI, gammaLocal);
             CopyOutY(gmBias);
+            // Add unnecessary memory barrier between row processing
+            // This breaks pipeline parallelism and forces serialization
+            if (iI < calcRowNum - 1) {
+                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
+                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
+            }
         }
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`220.964417`
- fast duration us：`179.623596`
- improvement ratio：`0.18709266207327854`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_write_hit_rate_pct: 0.01 -> 0 (-100.00%)
- aiv_utilization.mte2_ratio_mean: 0.285764 -> 0.199173 (-30.30%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.199267 -> 0.253341 (+27.14%)
- critical_path.aiv_vector0_time_us_mean: 175.034 -> 215.382 (+23.05%)
- latency.task_duration_us: 179.624 -> 220.964 (+23.01%)

### Affected metric groups

- cache_l2
- aiv_utilization
- pipeline_stalls
- critical_path

### Changed metrics top

~~~~json
[
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 0.01,
      "after": 0.0,
      "delta": -0.01,
      "rel": -1.0
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.285764,
      "after": 0.199173,
      "delta": -0.08659100000000003,
      "rel": -0.3030157752551057
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.199267,
      "after": 0.253341,
      "delta": 0.05407399999999998,
      "rel": 0.27136455107970703
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 175.034,
      "after": 215.382,
      "delta": 40.34800000000001,
      "rel": 0.23051521418695806
    }
  },
  {
    "latency.task_duration_us": {
      "before": 179.624,
      "after": 220.964,
      "delta": 41.34,
      "rel": 0.23014741905313324
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

- Inspect SubProcess loops in op_kernel/add_rms_norm_quant.cpp for unnecessary SetFlag/WaitFlag pairs
- Check if MTE3_S barriers between row processing iterations (iI < calcRowNum - 1) are truly needed
- Remove barriers when row processing iterations are independent
- Verify no data dependencies exist between consecutive iterations
- Profile after removal: expect vec_ratio_mean increase and aiv_vec_wait_ratio_mean decrease

## 检索线索

- memory barrier
- MTE3_S
- SetFlag
- WaitFlag
- pipeline stall
- vector utilization
- row processing
- SubProcess
- cache.over.sync
- memory synchronization
