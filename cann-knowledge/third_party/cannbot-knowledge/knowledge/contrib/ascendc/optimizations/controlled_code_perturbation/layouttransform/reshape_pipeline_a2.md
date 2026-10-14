---
type: Optimization
title: "reshape：流水优化"
description: "reshape 算子的流水优化经验，关注流水线停顿。"
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
# reshape：流水优化

## 来源信息

- experience ID：`fbc4e678f0aa37f6`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_018_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`queue.operation.intra.loop.redundancy`
- canonical family：`queue.operation.intra.loop.redundancy`
- 源记录时间：`2026-04-21T01:57:44`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate intra-loop redundant tensor queue operations
~~~~

### Original summary

~~~~text
Avoid placing tensor queue operations (EnQue/DeQue) inside memory copy loops when performing chunked data transfers. These operations add unnecessary synchronization overhead that serializes pipeline stages and reduces memory bandwidth utilization. Apply this optimization to operators performing chunked memory copies between GM and UB memory, especially in reshape operations.
~~~~

### Original root mechanism

~~~~text
Removing unnecessary EnQue/DeQue tensor queue operations inside chunked copy loops eliminates repeated tensor queue synchronization overhead per memory transfer chunk, reducing pipeline stalls and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Removing redundant EnQue/DeQue calls from CopyIn and CopyOut loops reduces aiv_mte3_wait_ratio_mean from 44.4% to 3.5%, decreases scalar_ratio_mean from 60.2% to 21.1%, and improves memory bandwidth from 2.26 Gbps to 2.77 Gbps, resulting in 18.3% faster execution.
~~~~

## 适用条件

- Operator uses chunked memory copy loops with DataCopy between GM and UB memory
- Tensor queue operations (EnQue/DeQue) appear inside copy loops
- Profile shows high aiv_mte3_wait_ratio_mean (>30%) and high scalar_ratio_mean (>50%)
- Memory bandwidth utilization is below expected levels (<2.8 Gbps for GM memory)

## 失效边界

- Tensor queue operations are required for data dependency between pipeline stages
- Memory copy size is small enough that queue overhead is negligible
- Operator has complex data dependencies requiring intermediate queue synchronization

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- reshape/op_kernel/reshape.cpp

### 改动符号

- MyReshape::CopyIn
- MyReshape::CopyOut

### 代码签名

- DataCopy
- DeQue
- EnQue
- FreeTensor
- datacopy
- queue_api

### 原始 diff

- 源 diff SHA-256：`7d5578799f95132e6589728906da8571273c5bee0b3a9c3a610b97d087428de2`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -32,6 +32,10 @@
             DataCopy(data[current_offset], gm_x[offset + current_offset], copy_size);
             current_offset += copy_size;
             remaining -= copy_size;
+            // Degrade: redundant queue operations inside loop
+            // Force unnecessary tensor queue synchronization per chunk
+            q_data.EnQue<DATA>(data);
+            LocalTensor<DATA> tmp = q_data.DeQue<DATA>();
         }
         q_data.EnQue<DATA>(data);
     }
@@ -46,6 +50,10 @@
             DataCopy(gm_y[offset + current_offset], data[current_offset], copy_size);
             current_offset += copy_size;
             remaining -= copy_size;
+            // Degrade: redundant queue operations inside loop
+            // Force unnecessary tensor queue synchronization per chunk
+            q_data.EnQue<DATA>(data);
+            LocalTensor<DATA> tmp = q_data.DeQue<DATA>();
         }
         q_data.FreeTensor(data);
     }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`390198.46875`
- fast duration us：`318903.53125`
- improvement ratio：`0.18271454967107684`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.034964 -> 0.443837 (+1169.41%)
- aiv_utilization.scalar_ratio_mean: 0.211257 -> 0.601594 (+184.77%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.082228 -> 0.055471 (-32.54%)
- critical_path.aiv_vector0_time_us_mean: 318849 -> 390133 (+22.36%)
- latency.task_duration_us: 318904 -> 390198 (+22.36%)

### Affected metric groups

- pipeline_stalls
- aiv_utilization
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.034964,
      "after": 0.443837,
      "delta": 0.408873,
      "rel": 11.694113945772793
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.211257,
      "after": 0.601594,
      "delta": 0.39033699999999993,
      "rel": 1.8476878872652738
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.082228,
      "after": 0.055471,
      "delta": -0.026756999999999996,
      "rel": -0.3254001070195067
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 318848.548,
      "after": 390132.783,
      "delta": 71284.23499999999,
      "rel": 0.22356769521810707
    }
  },
  {
    "latency.task_duration_us": {
      "before": 318903.531,
      "after": 390198.469,
      "delta": 71294.93799999997,
      "rel": 0.22356271119494114
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

- Inspect CopyIn and CopyOut functions in reshape.cpp for chunked memory copy loops
- Remove EnQue/DeQue calls from inside while loops performing DataCopy operations
- Keep queue operations outside loops: EnQue after all data is copied in CopyIn, FreeTensor after all data is copied in CopyOut
- Verify optimization by checking reduction in aiv_mte3_wait_ratio_mean and scalar_ratio_mean metrics

## 检索线索

- reshape
- queue
- EnQue
- DeQue
- DataCopy
- pipeline_stall
- memory_copy
- chunked_copy
- tensor_queue
- synchronization
