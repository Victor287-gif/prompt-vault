---
type: Optimization
title: "reshape：数据搬运优化"
description: "reshape 算子的数据搬运优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.layout_transform", "topic.implementation.synchronization", "topic.data_movement.memory_access"]
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
# reshape：数据搬运优化

## 来源信息

- experience ID：`939ec93a341403ea`
- 源 schema version：`v2`
- 算子：`reshape`
- source iteration：`round_004_degrade`
- 源 category：`math`
- 计算分区：`layouttransform`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.transfer.contiguous`
- canonical family：`memory.transfer.fragmentation`
- 源记录时间：`2026-04-20T23:47:41`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use single DataCopy calls for contiguous memory transfers
~~~~

### Original summary

~~~~text
For contiguous memory transfers in Ascend C operators, use single DataCopy calls instead of splitting transfers into multiple small chunks. This optimization reduces function call overhead and allows memory transfer engines (MTE) to operate at peak bandwidth. Apply this technique when copying contiguous data between global memory and unified buffer (UB) in operators like Reshape.
~~~~

### Original root mechanism

~~~~text
Single large DataCopy calls enable hardware memory transfer engines to operate efficiently on contiguous memory regions, while multiple small chunked copies introduce excessive function call overhead and prevent optimal memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Using single DataCopy calls for contiguous transfers -> reduces MTE wait ratios from ~99.7% to ~3.6% -> increases memory bandwidth from 2.27 Gbps to 38.87 Gbps -> reduces latency from 155,273 μs to 4,545 μs (33x speedup).
~~~~

## 适用条件

- Operator performs contiguous memory transfers between global memory and unified buffer
- Using DataCopy API with LocalTensor and global memory pointers
- Transfer size is known at compile time or can be determined at runtime
- Memory regions are contiguous without gaps or striding

## 失效边界

- Memory transfers require non-contiguous access patterns
- Transfer size must be limited due to hardware constraints
- Memory regions have alignment requirements that prevent single large transfers
- Data needs to be processed in chunks during the transfer

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

- AllocTensor
- DataCopy
- DeQue
- EnQue
- FreeTensor
- datacopy
- queue_api
- ub

### 原始 diff

- 源 diff SHA-256：`2d348992517f05acca8e3c609ee8254b7d4fe317efd3bbf8e8a7dd07b8551e67`

````diff
--- Before/op_kernel/reshape.cpp
+++ After/op_kernel/reshape.cpp
@@ -23,12 +23,30 @@
     ReshapeTilingInfo<1> ti;
     __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.AllocTensor<DATA>();
-        DataCopy(data, gm_x[offset], length);
+        // Degrade: split single copy into multiple smaller chunks
+        const uint32_t CHUNK_SIZE = 16;  // Small chunk size to increase overhead
+        uint32_t remaining = length;
+        uint32_t current_offset = 0;
+        while (remaining > 0) {
+            uint32_t copy_size = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+            DataCopy(data[current_offset], gm_x[offset + current_offset], copy_size);
+            current_offset += copy_size;
+            remaining -= copy_size;
+        }
         q_data.EnQue<DATA>(data);
     }
     __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
         LocalTensor<DATA> data = q_data.DeQue<DATA>();
-        DataCopy(gm_y[offset], data, length);
+        // Degrade: split single copy into multiple smaller chunks
+        const uint32_t CHUNK_SIZE = 16;  // Small chunk size to increase overhead
+        uint32_t remaining = length;
+        uint32_t current_offset = 0;
+        while (remaining > 0) {
+            uint32_t copy_size = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+            DataCopy(gm_y[offset + current_offset], data[current_offset], copy_size);
+            current_offset += copy_size;
+            remaining -= copy_size;
+        }
         q_data.FreeTensor(data);
     }
 public:
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`155273.296875`
- fast duration us：`4545.390625`
- improvement ratio：`0.9707265143686671`
- machine confidence：`0.95`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 4539.78 -> 155233 (+3319.38%)
- latency.task_duration_us: 4545.39 -> 155273 (+3316.06%)
- aiv_utilization.scalar_ratio_mean: 0.05087 -> 0.213917 (+320.52%)
- aiv_utilization.vec_ratio_mean: 4e-06 -> 0 (-100.00%)
- arithmetic.aiv_vec_misc_ratio_mean: 4e-06 -> 0 (-100.00%)

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
      "before": 4539.782,
      "after": 155232.533,
      "delta": 150692.751,
      "rel": 33.19382979182701
    }
  },
  {
    "latency.task_duration_us": {
      "before": 4545.391,
      "after": 155273.297,
      "delta": 150727.906,
      "rel": 33.160602905228615
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.05087,
      "after": 0.213917,
      "delta": 0.163047,
      "rel": 3.2051700412816984
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 4e-06,
      "after": 0.0,
      "delta": -4e-06,
      "rel": -1.0
    }
  },
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 4e-06,
      "after": 0.0,
      "delta": -4e-06,
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

- In CopyIn/CopyOut methods of Ascend C operators, use single DataCopy calls for contiguous transfers
- Replace looped chunked copies with direct DataCopy(data, gm_x[offset], length) calls
- Ensure LocalTensor allocation size matches the transfer length
- Verify memory regions are properly aligned for optimal DataCopy performance
- Monitor aiv_mte2_wait_ratio_mean and aiv_mte3_wait_ratio_mean metrics - values near 1.0 indicate excessive memory transfer overhead

## 检索线索

- DataCopy
- memory transfer
- contiguous
- chunked
- MTE wait
- memory bandwidth
- reshape
- Ascend C
- unified buffer
- global memory
