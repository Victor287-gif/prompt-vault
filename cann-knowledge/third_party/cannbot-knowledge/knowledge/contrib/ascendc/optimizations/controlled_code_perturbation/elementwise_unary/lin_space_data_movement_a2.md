---
type: Optimization
title: "lin_space：数据搬运优化"
description: "lin_space 算子的数据搬运优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/docs/aclnnLinspace.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/aclnn_linspace.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/aclnn_linspace.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/fill.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/lin_space.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/lin_space_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/lin_space_l0.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/lin_space_l0.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_host/lin_space_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_base.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_half_and_float.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_need_cast.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_small_shape.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_small_shape_and_cast.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_with_big_shape.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/op_kernel/lin_space_with_big_shape_and_cast.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lin_space/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# lin_space：数据搬运优化

## 来源信息

- experience ID：`6c9da4581139cf10`
- 源 schema version：`v2`
- 算子：`lin_space`
- source iteration：`round_018_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.transfer.inefficient.chunking`
- canonical family：`memory.transfer.inefficient.chunking`
- 源记录时间：`2026-04-20T23:13:12`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use single DataCopy operations instead of chunked copies for memory transfers
~~~~

### Original summary

~~~~text
When copying data from Unified Buffer (UB) to Global Memory (GM) in Ascend C kernels, using a single DataCopy operation for contiguous memory regions provides significantly better performance than breaking the transfer into multiple small chunks. The chunked approach introduces unnecessary synchronization points and reduces effective memory bandwidth utilization, particularly affecting the CopyOut and CopyOutReverse functions in vector operators.
~~~~

### Original root mechanism

~~~~text
Using a single DataCopy operation for contiguous memory transfers reduces synchronization overhead and improves memory bandwidth utilization compared to multiple small chunked copies.
~~~~

### Original causal chain

~~~~text
Replacing chunked DataCopy calls (32-element chunks) with single DataCopy operations -> reduces aiv_vec_wait_ratio_mean from 0.967804 to 0.000179 and aiv_mte3_wait_ratio_mean from 0.993168 to 0.000502 -> improves L2 cache write hit rate from 7.29% to 84.38% -> increases UB-to-GM bandwidth utilization from 4.99% to 23.87% -> reduces task duration from 7499.77us to 1055.16us (7.1x speedup).
~~~~

## 适用条件

- Implementing memory copy operations in Ascend C kernels using DataCopy API
- Transferring contiguous data blocks from Unified Buffer to Global Memory
- Working with vector operators that have CopyOut or CopyOutReverse functions
- When copying data sizes larger than cache line size (typically 128 bytes)

## 失效边界

- When memory transfers need explicit synchronization barriers between chunks
- When dealing with non-contiguous memory access patterns requiring strided copies
- When hardware documentation specifies optimal chunk sizes for specific memory types
- When implementing scatter-gather operations that inherently require multiple transfers

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/lin_space.cpp

### 改动符号

- LinSpaceWithBigShape::CopyOut
- LinSpaceWithBigShape::CopyOutReverse

### 代码签名

- DataCopy
- DeQue
- FreeTensor
- TQue
- datacopy
- queue_api

### 原始 diff

- 源 diff SHA-256：`9de0950197597fbecd5274f5fa8e813570f00d7f02b649d72330468a2ea3044c`

````diff
--- Before/op_kernel/lin_space.cpp
+++ After/op_kernel/lin_space.cpp
@@ -316,7 +316,19 @@
 __aicore__ inline void LinSpaceWithBigShape<T>::CopyOut(const int64_t offset, const int64_t &outLen)
 {
     LocalTensor<T> outLocal = outQueue.DeQue<T>();
-    DataCopy(outputGm[offset], outLocal, outLen);
+    // Degradation: Use inefficient chunking for memory copies
+    // Instead of one large DataCopy, use multiple smaller copies
+    // This increases memory transfer overhead and synchronization points
+    const int64_t CHUNK_SIZE = 32;  // Small chunk size to force many transfers
+    int64_t remaining = outLen;
+    int64_t currentOffset = 0;
+
+    while (remaining > 0) {
+        int64_t copySize = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+        DataCopy(outputGm[offset + currentOffset], outLocal[currentOffset], copySize);
+        currentOffset += copySize;
+        remaining -= copySize;
+    }
     outQueue.FreeTensor(outLocal);
 }

@@ -324,7 +336,19 @@
 __aicore__ inline void LinSpaceWithBigShape<T>::CopyOutReverse(const int64_t offset, const int64_t &outLen)
 {
     LocalTensor<T> outLocal = outQueue.DeQue<T>();
-    DataCopy(outputGm[offset], outLocal[maxOutNum - outLen], outLen);
+    // Degradation: Use inefficient chunking for memory copies
+    // Instead of one large DataCopy, use multiple smaller copies
+    // This increases memory transfer overhead and synchronization points
+    const int64_t CHUNK_SIZE = 32;  // Small chunk size to force many transfers
+    int64_t remaining = outLen;
+    int64_t currentOffset = 0;
+
+    while (remaining > 0) {
+        int64_t copySize = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+        DataCopy(outputGm[offset + currentOffset], outLocal[maxOutNum - outLen + currentOffset], copySize);
+        currentOffset += copySize;
+        remaining -= copySize;
+    }
     outQueue.FreeTensor(outLocal);
 }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`7499.77002`
- fast duration us：`1055.161133`
- improvement ratio：`0.8593075347395787`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_write_hit_rate_pct: 7.29 -> 84.38 (+1057.48%)
- cache_l2.aiv_total_hit_rate_pct: 7.31 -> 84.38 (+1054.31%)
- latency.task_duration_us: 1055.16 -> 7499.77 (+610.77%)
- critical_path.aiv_vector0_time_us_mean: 956.853 -> 6218.71 (+549.91%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0001 -> 0 (-100.00%)

### Affected metric groups

- cache_l2
- latency
- critical_path
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 7.29,
      "after": 84.38,
      "delta": 77.08999999999999,
      "rel": 10.574759945130314
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 7.31,
      "after": 84.38,
      "delta": 77.07,
      "rel": 10.543091655266757
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1055.161,
      "after": 7499.77,
      "delta": 6444.609,
      "rel": 6.107702047365284
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 956.853,
      "after": 6218.708,
      "delta": 5261.855,
      "rel": 5.499125780030997
    }
  },
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0001,
      "after": 0.0,
      "delta": -0.0001,
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

- In Ascend C kernel files like op_kernel/lin_space.cpp, replace chunked DataCopy loops with single DataCopy calls for contiguous memory regions
- For functions like CopyOut and CopyOutReverse in LinSpaceWithBigShape template class, use DataCopy(dst, src, totalLength) instead of while loops with small chunk sizes
- Ensure memory addresses are properly aligned when using single DataCopy operations
- Monitor aiv_mte3_wait_ratio_mean and aiv_vec_wait_ratio_mean metrics - high values indicate potential memory transfer inefficiencies
- Check UB-to-GM bandwidth utilization (memory_ub.ub_to_gm_bw_usage_rate_pct) - values below 20% may indicate suboptimal memory transfer patterns

## 检索线索

- DataCopy
- memory transfer
- chunking
- CopyOut
- CopyOutReverse
- Unified Buffer
- Global Memory
- Ascend C
- vector operator
- memory bandwidth
- synchronization overhead
- L2 cache hit rate
