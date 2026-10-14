---
type: Optimization
title: "reshape：流水与并行优化"
description: "reshape 算子的流水与并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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
# reshape：流水与并行优化

## 来源信息

- experience ID：`07a2c486f91132b0`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism.reduction`
- canonical family：`pipeline.parallelism.reduction`
- 源记录时间：`2026-04-20T22:30:13`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Optimize TQueBind buffer count for pipelined memory operations
~~~~

### Original summary

~~~~text
For Ascend C operators with pipelined memory operations using TQueBind, reducing the buffer count from 2 to 1 can improve performance by 21.6% when memory bandwidth utilization is below theoretical maximum. This optimization eliminates pipeline stalls caused by synchronization between parallel CopyIn and CopyOut operations, allowing more efficient sequential execution.
~~~~

### Original root mechanism

~~~~text
Using TQueBind buffer count of 1 instead of 2 reduces pipeline stalls by forcing sequential execution of CopyIn and CopyOut operations, eliminating synchronization overhead between parallel memory transfers.
~~~~

### Original causal chain

~~~~text
apply pipeline.parallelism.reduction in reshape/op_kernel/reshape.cpp -> improve aiv_utilization.mte2_ratio_mean -> 21.6% latency improvement
~~~~

## 适用条件

- Operator uses TQueBind with TPosition::VECIN and TPosition::VECOUT for pipelined memory transfers
- Memory bandwidth utilization is below theoretical maximum (gm_to_ub_bw_usage_rate_pct < 40%)
- Pipeline stalls show high aiv_mte2_wait_ratio_mean or aiv_mte3_wait_ratio_mean (>0.99)
- Operator has simple data movement pattern with sequential CopyIn and CopyOut operations
- Memory operations dominate compute time (high mte2/mte3 utilization ratios)

## 失效边界

- Operator requires overlapping memory transfers with computation
- Memory bandwidth utilization is already near maximum (>80%)
- Operator has complex data dependencies requiring parallel memory transfers
- Using TQueBind buffer count of 1 causes pipeline bubbles in compute-bound operations
- Operator uses advanced pipelining techniques with multiple concurrent memory streams

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reshape/op_kernel/reshape.cpp

### 改动符号

- MyReshape::MyReshape
- TQueBind

### 代码签名

- AllocTensor
- InitBuffer
- TBuf
- TQue
- compile_time_specialization
- queue_api
- tbuf
- template
- tile
- tile_shape

### 原始 diff

- 源 diff SHA-256：`ee6dd90945772f89be00f67aaff5304f88f5cc8c93ccad555ffc444c4d050069`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -19,7 +19,7 @@
 template<class DATA, class SHAPE> class MyReshape<DATA, SHAPE, 1> {
     GlobalTensor<DATA> gm_x, gm_y;
     GlobalTensor<SHAPE> gm_shape;
-    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_data;
+    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> q_data;
     ReshapeTilingInfo<1> ti;
     __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.AllocTensor<DATA>();
@@ -36,7 +36,7 @@
         gm_x.SetGlobalBuffer((__gm__ DATA *)x);
         gm_shape.SetGlobalBuffer((__gm__ SHAPE *)shape);
         gm_y.SetGlobalBuffer((__gm__ DATA *)y);
-        p->InitBuffer(q_data, 2, this->ti.tileLength * sizeof(DATA));
+        p->InitBuffer(q_data, 1, this->ti.tileLength * sizeof(DATA));
     }
     __aicore__ inline void exec() {
         for(uint32_t i = 0; i < TI.tileNumber; i++) {
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`3594.671875`
- fast duration us：`2817.056396`
- improvement ratio：`0.21632446744530753`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.mte2_ratio_mean: 0.710371 -> 0.394411 (-44.48%)
- aiv_utilization.mte3_ratio_mean: 0.859547 -> 0.60107 (-30.07%)
- aiv_utilization.scalar_ratio_mean: 0.045995 -> 0.032322 (-29.73%)
- critical_path.aiv_vector0_time_us_mean: 2813.02 -> 3590.94 (+27.65%)
- latency.task_duration_us: 2817.06 -> 3594.67 (+27.60%)

### Affected metric groups

- aiv_utilization
- critical_path
- latency
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.710371,
      "after": 0.394411,
      "delta": -0.31595999999999996,
      "rel": -0.44478167042291983
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.859547,
      "after": 0.60107,
      "delta": -0.25847699999999996,
      "rel": -0.30071305001355364
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.045995,
      "after": 0.032322,
      "delta": -0.013673000000000005,
      "rel": -0.29727144254810317
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2813.018,
      "after": 3590.944,
      "delta": 777.9259999999999,
      "rel": 0.2765449776716679
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2817.056,
      "after": 3594.672,
      "delta": 777.616,
      "rel": 0.2760385310054184
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

- 1. Identify operators using TQueBind<TPosition::VECIN, TPosition::VECOUT, N> in Ascend C code
- 2. Check performance metrics for high pipeline stall ratios (aiv_mte2_wait_ratio_mean > 0.99 or aiv_mte3_wait_ratio_mean > 0.99)
- 3. Verify memory bandwidth is underutilized (gm_to_ub_bw_usage_rate_pct < 40% and ub_to_gm_bw_usage_rate_pct < 40%)
- 4. Reduce TQueBind buffer count from 2 to 1 in both declaration and InitBuffer call
- 5. Update template instantiation: TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> q_data;
- 6. Update buffer initialization: p->InitBuffer(q_data, 1, tileLength * sizeof(DATA));
- 7. Recompile and profile to verify reduced pipeline stalls and improved memory utilization

## 检索线索

- TQueBind
- pipeline
- memory
- bandwidth
- stalls
- mte2
- mte3
- CopyIn
- CopyOut
- buffer
- queue
- reshape
- Ascend C
- TPosition
- VECIN
- VECOUT
- InitBuffer
- AllocTensor
