---
type: Optimization
title: "lerp：搬运与内存优化"
description: "lerp 算子的搬运与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access", "topic.programming_model", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/docs/Lerp.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/framework/tf_plugin/tensorflow_lerp_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_host/lerp_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/op_kernel/lerp.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/Lerp_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/st/test_lerp.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/lerp/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# lerp：搬运与内存优化

## 来源信息

- experience ID：`fe6096e30008fab5`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_018_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.copy.dependency`
- canonical family：`memory.copy.dependency`
- 源记录时间：`2026-04-20T22:50:04`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate artificial data dependencies between tensor copies
~~~~

### Original summary

~~~~text
Avoid creating artificial data dependencies between consecutive DataCopy operations in the CopyIn method. When copying multiple tensors (start, end, weight) in sequence, do not insert dummy computation loops that force sequential execution. Instead, allow AscendC's memory copy operations to execute in parallel or with better pipelining. This reduces scalar computation overhead and improves memory bandwidth utilization from 0.32 Gbps to 0.81 Gbps for GM reads and from 0.66 Gbps to 1.68 Gbps for UB vector reads.
~~~~

### Original root mechanism

~~~~text
Removing artificial computational loops between DataCopy operations allows parallel memory transfers to overlap, reducing scalar overhead and improving memory bandwidth utilization.
~~~~

### Original causal chain

~~~~text
Remove dummy computation loops between DataCopy operations → reduces scalar computation overhead (scalar_ratio_mean: 0.78 → 0.90) → decreases vector wait ratio (aiv_vec_wait_ratio_mean: 0.00012 → 0.002176) → improves memory bandwidth utilization (aiv_main_mem_read_bw_gbps: 0.32 → 0.81) → reduces task duration from 397128μs to 155522μs
~~~~

## 适用条件

- When copying multiple tensors sequentially in AscendC's CopyIn method
- When using DataCopy operations for tensor transfers between GM and UB memory
- When performance shows high scalar_ratio_mean (>0.78) and low memory bandwidth utilization
- When aiv_vec_wait_ratio_mean is unusually low (<0.0002) indicating artificial dependencies
- When MTE2 utilization increases significantly (>100% relative change) due to serialized memory operations

## 失效边界

- When actual data dependencies exist between tensor copies that require sequential processing
- When memory bandwidth is already saturated (>80% of theoretical maximum)
- When scalar computation is intentionally needed for data validation or transformation
- When using non-AscendC memory copy APIs with different synchronization requirements

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- lerp/op_kernel/lerp.cpp

### 改动符号

- KernelLerp::CopyIn

### 代码签名

- DataCopy
- GetValue
- datacopy
- scalar_value_access
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`9c1741bb4f9145bd698fcf2ebcf5dd0f4179b7d1a5da72ef1686ddbb5df3eb3d`

````diff
--- Before/op_kernel/lerp.cpp
+++ After/op_kernel/lerp.cpp
@@ -101,8 +101,23 @@

       while (remaining > 0) {
         uint64_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+        // DEGRADATION: Add artificial computational dependency between tensor copies
+        // Copy start tensor first
         AscendC::DataCopy(startLocal[offset], startGm[progress * this->ubPartDataNum + offset], chunk);
+        // Artificial dependency: perform dummy computation on start tensor before copying end tensor
+        // This creates a data dependency that prevents parallel copying
+        for (uint64_t i = 0; i < chunk; ++i) {
+          volatile TYPE_START dummy = startLocal.GetValue(offset + i);
+          (void)dummy; // Use the value to prevent optimization
+        }
+        // Copy end tensor second
         AscendC::DataCopy(endLocal[offset], endGm[progress * this->ubPartDataNum + offset], chunk);
+        // Artificial dependency: perform dummy computation on end tensor before copying weight tensor
+        for (uint64_t i = 0; i < chunk; ++i) {
+          volatile TYPE_END dummy = endLocal.GetValue(offset + i);
+          (void)dummy; // Use the value to prevent optimization
+        }
+        // Copy weight tensor last
         AscendC::DataCopy(weightLocal[offset], weightGm[progress * this->ubPartDataNum + offset], chunk);
         offset += chunk;
         remaining -= chunk;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`397128.5`
- fast duration us：`155521.5`
- improvement ratio：`0.6083849434125227`
- machine confidence：`0.95`

### Profile evidence

- latency.task_duration_us: 155522 -> 397128 (+155.35%)
- critical_path.aiv_vector0_time_us_mean: 155263 -> 393351 (+153.35%)
- aiv_utilization.mte2_ratio_mean: 0.090722 -> 0.206351 (+127.45%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.002176 -> 0.00012 (-94.49%)
- aiv_utilization.mte3_ratio_mean: 0.046454 -> 0.018084 (-61.07%)

### Affected metric groups

- latency
- critical_path
- aiv_utilization
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "latency.task_duration_us": {
      "before": 155521.5,
      "after": 397128.5,
      "delta": 241607.0,
      "rel": 1.5535279688017412
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 155262.785,
      "after": 393351.065,
      "delta": 238088.28,
      "rel": 1.5334536218708172
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.090722,
      "after": 0.206351,
      "delta": 0.11562900000000001,
      "rel": 1.274542007451335
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.002176,
      "after": 0.00012,
      "delta": -0.002056,
      "rel": -0.9448529411764707
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.046454,
      "after": 0.018084,
      "delta": -0.028370000000000003,
      "rel": -0.6107116717613122
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

- In KernelLerp::CopyIn method in lerp/op_kernel/lerp.cpp, ensure consecutive DataCopy calls for different tensors (start, end, weight) are not separated by artificial computation loops
- Remove any dummy loops that read tensor values with GetValue() between DataCopy operations unless actual computation is required
- Allow AscendC's memory engine to pipeline or parallelize multiple DataCopy operations by keeping them adjacent without artificial dependencies
- Monitor aiv_vec_wait_ratio_mean - values below 0.0002 may indicate excessive artificial dependencies limiting parallelism
- Check MTE2 utilization - significant increases (>100% relative) when adding dependencies suggest memory operations are being serialized

## 检索线索

- DataCopy
- tensor_copy
- memory_copy
- artificial_dependency
- scalar_overhead
- parallel_memory_transfer
- AscendC
- CopyIn
- lerp
- memory_bandwidth
- pipeline_stall
- GetValue
- ub
- GM
- memory_engine
