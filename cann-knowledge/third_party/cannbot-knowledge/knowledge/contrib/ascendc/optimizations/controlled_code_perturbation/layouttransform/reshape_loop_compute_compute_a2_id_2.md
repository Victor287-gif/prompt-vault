---
type: Optimization
title: "reshape：循环计算优化"
description: "reshape 算子的循环计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
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

- experience ID：`550eb0ee92bfa359`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_011_refactor`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.inline`
- canonical family：`loop.inline`
- 源记录时间：`2026-04-21T01:00:54`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Replace chunked while-loop copying with single DataCopy operations
~~~~

### Original summary

~~~~text
In Ascend C reshape operators, replacing chunked while-loop memory copies with single DataCopy operations dramatically improves performance by reducing pipeline stalls and increasing memory bandwidth utilization. This optimization is particularly effective for memory-bound operations where DataCopy can leverage hardware acceleration for contiguous memory transfers.
~~~~

### Original root mechanism

~~~~text
Using single DataCopy operations instead of chunked while-loops reduces pipeline stalls by eliminating repeated loop overhead and instruction cache pressure, enabling higher memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Single DataCopy calls replace chunked while-loop copying → reduces scalar instruction ratio from 28.5% to 5.9% → eliminates MTE2/MTE3 wait ratios from ~99% to ~2-4% → enables higher GM memory bandwidth (45.44 Gbps vs 3.06 Gbps) → reduces task duration from 230,773μs to 7,778μs (29.7x speedup).
~~~~

## 适用条件

- Memory copy operations in Ascend C kernels use while-loops with small chunk sizes (e.g., CHUNK_SIZE = 16)
- Copy operations are between GM and UB memory with DataCopy API
- Memory transfers are contiguous and can be handled by single DataCopy calls
- Performance shows high MTE2/MTE3 wait ratios (>95%) indicating memory transfer inefficiency

## 失效边界

- Memory transfers require non-contiguous access patterns
- Data needs to be processed in chunks due to UB size limitations
- Copy operations require data transformation between chunks

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
- DataCopy
- DeQue
- EnQue
- FreeTensor
- datacopy
- queue_api
- ub

### 原始 diff

- 源 diff SHA-256：`93c3b0d2f389228699f02673262bb8e184e4a9de7d6ea0c15ff562ffd9d50d5a`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -23,30 +23,12 @@
     ReshapeTilingInfo<1> ti;
     __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.AllocTensor<DATA>();
-        // Degrade: split single copy into multiple smaller chunks
-        const uint32_t CHUNK_SIZE = 16;  // Small chunk size to increase overhead
-        uint32_t remaining = length;
-        uint32_t current_offset = 0;
-        while (remaining > 0) {
-            uint32_t copy_size = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
-            DataCopy(data[current_offset], gm_x[offset + current_offset], copy_size);
-            current_offset += copy_size;
-            remaining -= copy_size;
-        }
+        DataCopy(data, gm_x[offset], length);
         q_data.EnQue<DATA>(data);
     }
     __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.DeQue<DATA>();
-        // Degrade: split single copy into multiple smaller chunks
-        const uint32_t CHUNK_SIZE = 16;  // Small chunk size to increase overhead
-        uint32_t remaining = length;
-        uint32_t current_offset = 0;
-        while (remaining > 0) {
-            uint32_t copy_size = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
-            DataCopy(gm_y[offset + current_offset], data[current_offset], copy_size);
-            current_offset += copy_size;
-            remaining -= copy_size;
-        }
+        DataCopy(gm_y[offset], data, length);
         q_data.FreeTensor(data);
     }
 public:
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`230772.828125`
- fast duration us：`7778.015625`
- improvement ratio：`0.9662957910244659`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 7767.06 -> 230742 (+2870.77%)
- latency.task_duration_us: 7778.02 -> 230773 (+2866.99%)
- aiv_utilization.scalar_ratio_mean: 0.059335 -> 0.28522 (+380.69%)
- aiv_utilization.vec_ratio_mean: 2e-06 -> 0 (-100.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 2e-06 -> 0 (-100.00%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- arithmetic

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 7767.062,
      "after": 230741.802,
      "delta": 222974.74,
      "rel": 28.70773273085756
    }
  },
  {
    "latency.task_duration_us": {
      "before": 7778.016,
      "after": 230772.828,
      "delta": 222994.812,
      "rel": 28.669883425284805
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.059335,
      "after": 0.28522,
      "delta": 0.22588499999999997,
      "rel": 3.8069436251790676
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 2e-06,
      "after": 0.0,
      "delta": -2e-06,
      "rel": -1.0
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 2e-06,
      "after": 0.0,
      "delta": -2e-06,
      "rel": -1.0
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

- In reshape.cpp CopyIn/CopyOut functions, replace while-loop chunked copying with single DataCopy calls
- Change from: while(remaining > 0) { DataCopy(data[current_offset], gm_x[offset + current_offset], copy_size); ... }
- Change to: DataCopy(data, gm_x[offset], length)
- Ensure LocalTensor allocation and queue operations remain unchanged
- Verify that length parameter fits within UB capacity for single transfer

## 检索线索

- DataCopy
- while-loop
- chunked
- memory_copy
- pipeline_stall
- MTE_wait
- reshape
- Ascend_C
- ub
- gm
- memory_bandwidth
