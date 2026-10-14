---
type: Optimization
title: "tril：流水与内存优化"
description: "tril 算子的流水与内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model.execution_unit.scalar"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/docs/Tril.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/aclnn_tril_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/aclnn_tril_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/TrilOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/framework/tf_plugin/tensorflow_tril_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/op_host/tril.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/op_host/tril_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/op_kernel/tril.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/Tril_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/st/test_tril.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/tril/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# tril：流水与内存优化

## 来源信息

- experience ID：`899a0ee7f0d4cfd7`
- 源 schema version：`v2`
- 算子：`tril`
- source iteration：`round_007_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`buffering.reduction`
- canonical family：`buffering.reduction`
- 源记录时间：`2026-04-14T23:05:50`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce double buffering to single buffering
~~~~

### Original summary

~~~~text
Setting BUFFER_NUM=1 instead of BUFFER_NUM=2 in Ascend C tril operator improves performance by 2.08x. This optimization reduces scalar unit overhead from managing multiple queue buffers, allowing more efficient pipeline execution. Apply this when memory copy operations are fully overlapped with compute and scalar utilization is low (scalar_ratio_mean < 0.5).
~~~~

### Original root mechanism

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates pipeline overlap overhead between memory copy and compute operations, improving scalar unit utilization by reducing queue management overhead.
~~~~

### Original causal chain

~~~~text
BUFFER_NUM reduced from 2 to 1 → scalar_ratio_mean increased from 0.3359 to 0.755986 (+125%) → aiv_vector0_time_us_mean reduced from 3861.341 to 2008.306 (-48%) → task_duration_us reduced from 3861.957 to 2008.88 (-48%)
~~~~

## 适用条件

- Operator uses Ascend C pipeline with CopyIn/Compute/CopyOut pattern
- Scalar utilization (scalar_ratio_mean) is below 0.5
- Memory bandwidth utilization (gm_to_ub_bw_usage_rate_pct) is below 50%
- BUFFER_NUM > 1 is configured in op_kernel/*.cpp
- Operator has compute-bound characteristics with low vector unit utilization

## 失效边界

- Memory-bound operators where memory copy latency dominates computation
- Operators with high scalar_ratio_mean (>0.7)
- When BUFFER_NUM=1 already shows high pipeline stalls (aiv_vec_wait_ratio_mean > 0.9)
- Operators using complex data dependencies between tiles

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- tril/op_kernel/tril.cpp

### 改动符号

- BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`4ac49e55045036e6de1962bea3638b8e6caf5e28df43d072cf3e7c562743cc92`

````diff
--- Before/op_kernel/tril.cpp
+++ After/op_kernel/tril.cpp
@@ -3,7 +3,7 @@
  */
 #include "kernel_operator.h"
 using namespace AscendC;
-constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
+constexpr int32_t BUFFER_NUM = 1;                                     // Degradation: reduce from double buffering to single buffering
 constexpr int32_t minNum = 1;

 struct IntegrateParam{
@@ -78,20 +78,20 @@
             if(this->diagVal<=0){
                 init_row = -diagVal;
             }
-            for (int32_t i = 0; i < this->loopCnt-1; i++) {
-                CopyIn(GmOffset,this->fullTileLength);
-                Compute(this->fullCnt, calLength, init_row);
-                CopyOut(GmOffset,this->fullTileLength);
-                if(init_row>0){
-                    init_row-=this->fullRowInc;
-                    if(init_row<0){
-                        calLength-=init_row;
-                    }
-                }else{
-                    calLength+=this->fullRowInc;
+        for (int32_t i = 0; i < this->loopCnt-1; i++) {
+            CopyIn(GmOffset,this->fullTileLength);
+            Compute(this->fullCnt, calLength, init_row);
+            CopyOut(GmOffset,this->fullTileLength);
+            if(init_row>0){
+                init_row-=this->fullRowInc;
+                if(init_row<0){
+                    calLength-=init_row;
                 }
-                GmOffset+=this->fullTileLength;
+            }else{
+                calLength+=this->fullRowInc;
             }
+            GmOffset+=this->fullTileLength;
+        }
             CopyIn(GmOffset,this->lastTileLength);
             Compute(this->lastCnt, calLength, init_row);
             CopyOut(GmOffset,this->lastTileLength);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`3861.957275`
- fast duration us：`2008.880127`
- improvement ratio：`0.4798284952543915`
- machine confidence：`0.9`

### Profile evidence

- critical_path.aiv_vector0_time_us_mean: 2008.31 -> 3861.34 (+92.27%)
- latency.task_duration_us: 2008.88 -> 3861.96 (+92.24%)
- aiv_utilization.scalar_ratio_mean: 0.755986 -> 0.3359 (-55.57%)
- aiv_utilization.mte3_ratio_mean: 0.775568 -> 0.400052 (-48.42%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 31.92 -> 16.6 (-47.99%)

### Affected metric groups

- critical_path
- latency
- aiv_utilization
- memory_ub

### Changed metrics top

~~~~json
[
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2008.306,
      "after": 3861.341,
      "delta": 1853.0349999999999,
      "rel": 0.9226855867581931
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2008.88,
      "after": 3861.957,
      "delta": 1853.0769999999998,
      "rel": 0.9224428537294411
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 0.755986,
      "after": 0.3359,
      "delta": -0.42008600000000007,
      "rel": -0.5556796025323221
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.775568,
      "after": 0.400052,
      "delta": -0.375516,
      "rel": -0.48418191570565056
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 31.92,
      "after": 16.6,
      "delta": -15.32,
      "rel": -0.4799498746867168
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

- Check scalar_ratio_mean in performance profile - if <0.5, consider reducing BUFFER_NUM
- Inspect op_kernel/*.cpp for BUFFER_NUM definition (typically near top of file)
- Change constexpr int32_t BUFFER_NUM from 2 to 1
- Verify pipeline pattern follows CopyIn/Compute/CopyOut sequence
- Profile after change to confirm scalar_ratio_mean improvement and latency reduction

## 检索线索

- BUFFER_NUM
- double buffering
- single buffering
- scalar overhead
- AscendC
- pipeline optimization
- tril
- CopyIn
- Compute
- CopyOut
