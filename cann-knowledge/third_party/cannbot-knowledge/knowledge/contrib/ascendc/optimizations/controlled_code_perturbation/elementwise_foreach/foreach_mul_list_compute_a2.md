---
type: Optimization
title: "foreach_mul_list：边界计算优化"
description: "foreach_mul_list 算子的边界计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/docs/aclnnForeachMulList.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_host/foreach_mul_list.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_mul_list/op_kernel/foreach_mul_list.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_mul_list：边界计算优化

## 来源信息

- experience ID：`ab36b44ba4ef64a9`
- 源 schema version：`v2`
- 算子：`foreach_mul_list`
- source iteration：`round_003_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tensor.boundary.awareness.break`
- canonical family：`tensor.boundary.awareness.break`
- 源记录时间：`2026-04-14T21:55:00`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Per-chunk tensor buffer setup degradation
~~~~

### Original summary

~~~~text
Moving tensor buffer setup calls (SetGlobalBuffer) from per-tensor to per-chunk loops significantly degrades performance by forcing repeated GetTensorAddr computations. This optimization pattern shows that tensor buffer addresses should be computed once per tensor and reused across chunks rather than recomputed for each chunk. Apply this technique to foreach operators processing tensor lists with chunked data access patterns.
~~~~

### Original root mechanism

~~~~text
Moving SetGlobalBuffer calls from per-tensor loop to per-chunk loop forces expensive GetTensorAddr computation for every data chunk instead of once per tensor.
~~~~

### Original causal chain

~~~~text
Moving SetGlobalBuffer calls to per-chunk loop -> increased GetTensorAddr computations -> higher scalar ratio (0.0788 to 0.9994) and pipeline stalls -> vector wait ratio increased from 0.3078 to 0.9945 -> latency increased from 5260us to 6446us
~~~~

## 适用条件

- Processing foreach operators with tensor lists
- Using chunked data access patterns in foreach kernels
- When SetGlobalBuffer is called inside per-chunk loops
- When GetTensorAddr computations are performed repeatedly
- When scalar utilization exceeds 0.9 and vector wait ratio is high

## 失效边界

- When tensor addresses are constant across chunks
- When buffer setup can be hoisted outside loops
- When scalar utilization should be minimized
- When memory bandwidth is already saturated

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_mul_list/op_kernel/foreach_mul_list.cpp

### 改动符号

- KernelForeachUnary::SingleTensorProcess
- KernelForeachUnary::CopyIn
- KernelForeachUnary::CopyOut
- KernelForeachUnary::CopyInPlus
- ForeachNoScalarBinary::CopyInPlus
- ForeachNoScalarBinary::ProcessPlusInLoop

### 代码签名

- AllocTensor
- DataCopy
- DeQue
- EnQue
- TQue
- bool
- bool_control
- compile_time_specialization
- datacopy
- queue_api
- template

### 原始 diff

- 源 diff SHA-256：`48f61055a322f07bf99c63b3f8d106af77c1f81a827afceb8ca172305be36132`

````diff
--- Before/op_kernel/foreach_mul_list.cpp
+++ After/op_kernel/foreach_mul_list.cpp
@@ -96,13 +96,13 @@
     explicit __aicore__ inline KernelForeachUnary(Predicate& p) : Base(), pred(p) {}
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData);
     __aicore__ inline void Process();
-    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float>& float32Tensor);
+    __aicore__ inline void SingleTensorProcess(uint16_t tensorIdx, int64_t cursorStart, int64_t dataCount, LocalTensor<float>& float32Tensor);

 private:
-    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
+    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder, uint16_t tensorIdx, int64_t cursorStart);
     __aicore__ inline void Compute(uint32_t index, int64_t dataCount, LocalTensor<float>& float32Tensor, bool isRemainder);
-    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder);
-    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder);
+    __aicore__ inline bool CopyOut(uint32_t index, int64_t dataCount, bool isRemainder, uint16_t tensorIdx, int64_t cursorStart);
+    __aicore__ inline void CopyInPlus(uint32_t index, int64_t dataCount, bool isRemainder, uint16_t tensorIdx, int64_t cursorStart);
     __aicore__ inline void BeforeProcess();
     __aicore__ inline void AfterProcess();
     __aicore__ inline void ProcessPlusInLoop(uint32_t index, uint64_t cursorStart);
@@ -179,10 +179,10 @@

         dataCount = cursorEnd - cursorStart + 1;

-        inTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, inTensorsPtr) + cursorStart);
-        outTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, outTensorsPtr) + cursorStart);
-        ProcessPlusInLoop(i, cursorStart);
-        SingleTensorProcess(dataCount, float32Tensor);
+            inTensorsGM.Se... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`14530.310547000001`
- fast duration us：`10516.210449`
- improvement ratio：`0.2762570066906637`
- machine confidence：`0.9`

### Profile evidence

- aiv_utilization.scalar_ratio_mean: 0.078806 -> 0.99944 (+1168.23%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.995561 -> 6.8e-05 (-99.99%)
- pipeline_stalls.aiv_mte3_wait_ratio_mean: 0.995915 -> 0.022798 (-97.71%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.99452 -> 0.307758 (-69.05%)
- cache_l2.aiv_read_hit_rate_pct: 1.23 -> 1.79 (+45.53%)

### Affected metric groups

- aiv_utilization
- pipeline_stalls
- cache_l2
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.078806,
      "after": 0.99944,
      "delta": 0.920634,
      "rel": 11.682283074892775
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.995561,
      "after": 6.8e-05,
      "delta": -0.9954930000000001,
      "rel": -0.9999316968021046
    }
  },
  {
    "pipeline_stalls.aiv_mte3_wait_ratio_mean": {
      "before": 0.995915,
      "after": 0.022798,
      "delta": -0.973117,
      "rel": -0.9771084881741916
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.99452,
      "after": 0.307758,
      "delta": -0.686762,
      "rel": -0.6905461931383985
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 1.23,
      "after": 1.79,
      "delta": 0.56,
      "rel": 0.4552845528455285
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

- In foreach_mul_list.cpp, move SetGlobalBuffer calls from per-chunk loops to per-tensor initialization
- Compute GetTensorAddr once per tensor and store in local variables
- Pass tensor addresses as parameters to chunk processing functions
- In KernelForeachUnary::SingleTensorProcess, accept tensorIdx and cursorStart as parameters
- Update CopyIn, CopyOut, and CopyInPlus methods to accept tensorIdx and cursorStart parameters

## 检索线索

- foreach_mul_list
- SetGlobalBuffer
- GetTensorAddr
- tensor buffer setup
- per-chunk loop
- scalar utilization
- pipeline stalls
- vector wait ratio
- tensor boundary awareness
- chunked data access
