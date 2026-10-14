---
type: Optimization
title: "motion_compensation：代码结构优化"
description: "motion_compensation 算子的代码结构优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/docs/MotionCompensation.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/framework/tf_plugin/tensorflow_motion_compensation_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_host/motion_compensation.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_host/motion_compensation_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/op_kernel/motion_compensation.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/image/motion_compensation/opp_kernel_aicpu/.gitkeep
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# motion_compensation：代码结构优化

## 来源信息

- experience ID：`1d84a89ad3f89eba`
- 源 schema version：`v2`
- 算子：`motion_compensation`
- source iteration：`round_000_refactor`
- 源 category：`image`
- 计算分区：`interpolation`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`code.extract.method`
- canonical family：`code.extract.method`
- 源记录时间：`2026-04-14T21:15:40`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Extract duplicate core parameter initialization logic into reusable helper function
~~~~

### Original summary

~~~~text
When initializing multiple sets of core parameters with identical conditional logic patterns, extract the common initialization logic into a reusable helper function. This reduces code duplication, improves instruction cache locality, and can lead to unexpected performance gains by reducing instruction fetch overhead and improving branch prediction accuracy.
~~~~

### Original root mechanism

~~~~text
Extracting duplicated conditional parameter initialization logic into a reusable helper function reduces instruction cache pressure and improves branch prediction by consolidating identical control flow patterns.
~~~~

### Original causal chain

~~~~text
Extracting InitCoreParams helper function → reduced instruction cache pressure → decreased aiv_vector0_time_us_mean from 543.736 to 535.849 (-1.47%) → reduced total task_duration_us from 1166.543 to 1103.922 (-5.37%)
~~~~

## 适用条件

- Multiple sets of core parameters (coreDataNum/tileNum/tailDataNum) require identical conditional initialization logic
- Same if-else pattern repeated with different variable sets in the same kernel
- Core launch parameters follow similar big/small core distribution patterns
- KernelMotionCompensation::Init function contains duplicate conditional logic

## 失效边界

- Parameter initialization logic differs significantly between sets
- Performance-critical hot paths where inlining is beneficial
- Memory constraints prevent additional function call overhead

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- motion_compensation/op_kernel/motion_compensation.cpp

### 改动符号

- KernelMotionCompensation::InitCoreParams
- KernelMotionCompensation::Init

### 代码签名

- coreNum
- core_launch
- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`f60d858d2cb1f0ebb4868c10ec2272caddf5c17da41da07debcb64ba62df1075`

````diff
--- Before/op_kernel/motion_compensation.cpp
+++ After/op_kernel/motion_compensation.cpp
@@ -20,6 +20,27 @@
 {
 public:
     __aicore__ inline KernelMotionCompensation() {}
+
+    __aicore__ inline void InitCoreParams(uint32_t coreNum, int64_t bigCoreDataNum, int64_t smallCoreDataNum,
+                                          int64_t finalBigTileNum, int64_t finalSmallTileNum, int64_t tileDataNum,
+                                          int64_t bigTailDataNum, int64_t smallTailDataNum, int64_t tailBlockNum,
+                                          int64_t &coreDataNum, int64_t &tileNum, int64_t &tailDataNum, int64_t &globalBufferIndex)
+    {
+        if (coreNum < tailBlockNum)
+        {
+            coreDataNum = bigCoreDataNum;
+            tileNum = finalBigTileNum;
+            tailDataNum = bigTailDataNum;
+        }
+        else
+        {
+            coreDataNum = smallCoreDataNum;
+            tileNum = finalSmallTileNum;
+            tailDataNum = smallTailDataNum;
+            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (GetBlockIdx() - tailBlockNum);
+        }
+    }
+
     __aicore__ inline void Init(GM_ADDR points, GM_ADDR timestamp, GM_ADDR out,
                                 TPipe *pipeIn, int64_t N, int64_t ndim, float tMax,
                                 float f, float theta, float r_sin_theta,
@@ -37,32 +58,14 @@
         this->tileDataNum = tileDataNum;
         globalBufferIndex2 = bigCoreDataNum2 * GetBlockIdx();
         this->tileDataNum2 = tileDataNum2;
-        if (coreNum < tailBlockNum)
-        {
-            this->coreDataNum = bigCoreDataNum;
-            this->tileNum = finalBigTileNum;
-            this->tailDataNum = bigTailDataNum;
-        }
-        else
-        {
-            this->coreDataNum = smallCoreDataNum;
-            this->tileNum = finalSmallTileNum;
-            this->tailDataNum = smallTailDataNum;
-            globalB... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1166.543335`
- fast duration us：`1103.922119`
- improvement ratio：`0.05368100277217734`
- machine confidence：`0.85`

### Profile evidence

- latency.task_duration_us: 1103.92 -> 1166.54 (+5.67%)
- memory_ub.ub_read_bw_scalar_gbps: 0.0064 -> 0.0061 (-4.69%)
- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.01876 -> 0.019219 (+2.45%)
- critical_path.aiv_vector0_time_us_mean: 535.849 -> 543.736 (+1.47%)
- aiv_utilization.mte2_ratio_mean: 0.095908 -> 0.096866 (+1.00%)

### Affected metric groups

- latency

### Changed metrics top

~~~~json
[
  {
    "latency.task_duration_us": {
      "before": 1103.922,
      "after": 1166.543,
      "delta": 62.62099999999987,
      "rel": 0.056725928099992454
    }
  },
  {
    "memory_ub.ub_read_bw_scalar_gbps": {
      "before": 0.0064,
      "after": 0.0061,
      "delta": -0.0002999999999999999,
      "rel": -0.046874999999999986
    }
  },
  {
    "pipeline_stalls.aiv_mte2_wait_ratio_mean": {
      "before": 0.01876,
      "after": 0.019219,
      "delta": 0.0004590000000000011,
      "rel": 0.02446695095948833
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 535.849,
      "after": 543.736,
      "delta": 7.886999999999944,
      "rel": 0.014718698737890605
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.095908,
      "after": 0.096866,
      "delta": 0.0009580000000000005,
      "rel": 0.009988739208408063
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

- Identify duplicate conditional parameter initialization patterns in KernelMotionCompensation::Init
- Extract common logic into helper function like InitCoreParams with reference parameters
- Replace duplicate if-else blocks with calls to the helper function
- Test performance impact on aiv_vector0_time_us_mean and task_duration_us metrics

## 检索线索

- core parameter initialization
- duplicate conditional logic
- InitCoreParams
- KernelMotionCompensation
- code extraction
- instruction cache
- branch prediction
- motion_compensation
- bigCoreDataNum
- smallCoreDataNum
- tailBlockNum
