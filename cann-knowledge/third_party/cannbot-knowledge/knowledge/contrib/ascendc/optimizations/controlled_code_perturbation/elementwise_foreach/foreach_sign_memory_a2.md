---
type: Optimization
title: "foreach_sign：内存优化"
description: "foreach_sign 算子的内存优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/docs/aclnnForeachSign.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/op_host/foreach_sign.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/op_kernel/foreach_sign.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/op_kernel/foreach_sign.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_sign：内存优化

## 来源信息

- experience ID：`0f938ee250f5b82a`
- 源 schema version：`v2`
- 算子：`foreach_sign`
- source iteration：`round_004_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.address.recalculation`
- canonical family：`memory.address.recalculation`
- 源记录时间：`2026-04-19T23:12:10`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Hoist tensor address calculation out of chunk loops
~~~~

### Original summary

~~~~text
In ForeachSignND operators, hoisting tensor address calculations (`SetGlobalBuffer` calls) from per-chunk loops to per-tensor scope eliminates repeated pointer arithmetic for each data chunk. This optimization reduces scalar unit utilization from 99.8% to 13.8% and improves MTE2/MTE3 utilization from 39.6%/35.5% to 99.1%/84.6%. Apply this pattern when tensor processing involves multiple chunks within a single tensor.
~~~~

### Original root mechanism

~~~~text
Moving tensor address calculation from per-chunk to per-tensor scope reduces repeated global memory pointer arithmetic and improves MTE utilization by avoiding redundant address computations.
~~~~

### Original causal chain

~~~~text
Hoisting SetGlobalBuffer calls from chunk loops to tensor scope → reduces scalar unit utilization from 99.8% to 13.8% → decreases MTE2 wait ratio from 99.5% to 0.02% → improves vector unit utilization from 21.3% to 35.7% → reduces task duration from 5571us to 3325us (2.1x speedup).
~~~~

## 适用条件

- Foreach operators processing multiple tensors with chunk-based data movement
- When using SetGlobalBuffer API for tensor addressing in chunk loops
- When scalar unit utilization exceeds 90% due to repeated pointer arithmetic
- When MTE2/MTE3 wait ratios are high (>50%) despite sufficient memory bandwidth

## 失效边界

- Single-chunk tensor processing where address calculation overhead is negligible
- When tensor addresses need dynamic per-chunk adjustment based on data dependencies
- When memory access patterns require per-chunk address recomputation for correctness

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_sign/op_kernel/foreach_sign.cpp

### 改动符号

- ForeachSignND::SingleTensorProcess
- ForeachSignND::CopyIn
- ForeachSignND::ComputeAndCopyOut

### 代码签名

- AllocTensor
- DataCopy
- DataCopyPad
- DeQue
- EnQue
- FreeTensor
- TQue
- bool
- bool_control
- compile_time_specialization
- datacopy
- queue_api
- template

### 原始 diff

- 源 diff SHA-256：`371697d6a651fd51857e119ec79cb4c1a20e4a012d55cd1addec0514f42de047`

````diff
--- Before/op_kernel/foreach_sign.cpp
+++ After/op_kernel/foreach_sign.cpp
@@ -144,10 +144,10 @@
         return (a + b - 1) / b;
     };
     __aicore__ inline void ParseTilingData(const ForeachCommonTilingData *tilingData);
-    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor,
+    __aicore__ inline void SingleTensorProcess(uint16_t tensorIndex, int64_t dataCount, LocalTensor<float> &float32Tensor,
                                                LocalTensor<half> &float16Tensor);
-    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
-    __aicore__ inline void ComputeAndCopyOut(uint16_t index, int64_t dataCount, struct TensorInfo &tensorInfo,
+    __aicore__ inline void CopyIn(uint16_t tensorIndex, uint16_t chunkIndex, int64_t dataCount, bool isRemainder);
+    __aicore__ inline void ComputeAndCopyOut(uint16_t tensorIndex, uint16_t chunkIndex, int64_t dataCount, struct TensorInfo &tensorInfo,
                                              bool isRemainder);
     __aicore__ inline __gm__ T *GetInputTensorAddr(uint16_t index);
     __aicore__ inline __gm__ T *GetOutputTensorAddr(uint16_t index);
@@ -230,9 +230,7 @@
         }

         dataCount = cursorEnd - cursorStart + 1;
-        inTensorGM.SetGlobalBuffer(GetInputTensorAddr(i) + cursorStart);
-        outTensorGM.SetGlobalBuffer(GetOutputTensorAddr(i) + cursorStart);
-        SingleTensorProcess(dataCount, float32Tensor, float16Tensor);
+        SingleTensorProcess(i, dataCount, float32Tensor, float16Tensor);
     }
     if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int64_t>) {
         float32Queue.FreeTensor(float32Tensor);
@@ -242,8 +240,8 @@
 }

 template <typename T>
-__aicore__ inline void ForeachSignND<T>::SingleTensorProcess(int64_t dataCount, LocalTensor<float> &float32Tensor,
-                                                             LocalTensor<half> &float1... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`12799.696289`
- fast duration us：`6653.153319999999`
- improvement ratio：`0.48021006359989266`
- machine confidence：`0.95`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.137901 -> 0.997738 (+623.52%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.995157 -> 0.000185 (-99.98%)
- critical_path.aiv_vector0_time_us_mean: 3310.24 -> 5547.43 (+67.58%)
- latency.task_duration_us: 3325.03 -> 5571.05 (+67.55%)
- cache_l2.aiv_read_hit_rate_pct: 1.67 -> 2.68 (+60.48%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.137901,
      "after": 0.997738,
      "delta": 0.859837,
      "rel": 6.235175959565195
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.995157,
      "after": 0.000185,
      "delta": -0.994972,
      "rel": -0.9998140996847734
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 3310.243,
      "after": 5547.432,
      "delta": 2237.189,
      "rel": 0.675838299484358
    }
  },
  {
    "latency.task_duration_us": {
      "before": 3325.027,
      "after": 5571.052,
      "delta": 2246.0249999999996,
      "rel": 0.6754907554134145
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 1.67,
      "after": 2.68,
      "delta": 1.0100000000000002,
      "rel": 0.6047904191616769
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

- In ForeachSignND::Process function, move SetGlobalBuffer calls outside chunk loops
- Pass tensor index to SingleTensorProcess instead of computing addresses per chunk
- Compute input/output addresses once per tensor using GetInputTensorAddr/GetOutputTensorAddr
- Update CopyIn and ComputeAndCopyOut signatures to accept tensor index parameter
- Ensure address calculations use cursorStart offset only once per tensor

## 检索线索

- foreach_sign
- SetGlobalBuffer
- tensor_address
- pointer_arithmetic
- scalar_utilization
- MTE_wait
- chunk_loop
- address_recalculation
- memory_address
- foreach_operator
