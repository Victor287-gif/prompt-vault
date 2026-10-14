---
type: Optimization
title: "reciprocal：数据搬运优化"
description: "reciprocal 算子的数据搬运优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/docs/Reciprocal.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/framework/tf_plugin/tensorflow_reciprocal_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_host/reciprocal_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/op_kernel/reciprocal.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/Reciprocal_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/st/test_reciprocal.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/reciprocal/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# reciprocal：数据搬运优化

## 来源信息

- experience ID：`57ecd3c94331b026`
- 源 schema version：`v2`
- 算子：`reciprocal`
- source iteration：`round_017_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.dma_batching`
- canonical family：`untitled`
- 源记录时间：`2026-04-21T02:14:55`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Single DMA DataCopy for GM-UB Transfers
~~~~

### Original summary

~~~~text
For Ascend C operators performing GM-to-UB data transfers, using single DataCopy operations with full data size instead of chunked copies significantly improves performance. This optimization reduces DMA overhead, increases L2 cache hit rates from 0.99% to 93.75%, and improves vector unit utilization from 1.83% to 99.19%. Apply this pattern when transferring contiguous memory blocks between GM and UB in vector operators.
~~~~

### Original root mechanism

~~~~text
Using single large DataCopy operations instead of chunked copies reduces DMA overhead and improves L2 cache hit rates by allowing more efficient memory access patterns.
~~~~

### Original causal chain

~~~~text
Replacing chunked DataCopy loops with single DataCopy operations → reduces DMA overhead and improves memory access efficiency → increases L2 cache hit rates from 0.99% to 93.75% → reduces memory bandwidth contention → improves vector unit utilization from 1.83% to 99.19% → reduces task duration from 1347313μs to 24479μs (54x faster).
~~~~

## 适用条件

- Ascend C vector operators using DataCopy for GM-UB transfers
- Transferring contiguous memory blocks between global memory and unified buffer
- When processDataNum represents the full data size to transfer
- When using AscendC::LocalTensor with TQue queue APIs
- When CopyIn/CopyOut functions perform data movement in op_kernel implementations

## 失效边界

- When data must be processed in small chunks due to UB size constraints
- When non-contiguous memory access patterns require chunked transfers
- When using AscendC::GlobalTensor with different memory access patterns
- When memory alignment requirements prevent single large transfers

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/reciprocal.cpp

### 代码签名

- AllocTensor
- DataCopy
- DeQue
- EnQue
- FreeTensor
- TQue
- compile_time_specialization
- constexpr
- datacopy
- queue_api
- ub

### 原始 diff

- 源 diff SHA-256：`e806225fab3a77b4d011830f0056378bb2febf343994502c2064b5bf76d6112a`

````diff
--- Before/op_kernel/reciprocal.cpp
+++ After/op_kernel/reciprocal.cpp
@@ -82,7 +82,16 @@
     __aicore__ inline void CopyIn(int32_t progress)
     {
         AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
-        DataCopy(xLocal, xGm[progress * this->ubPartDataNum], this->processDataNum);
+        // Use chunked copies instead of single DMA to increase overhead
+        constexpr int32_t CHUNK_SIZE = 16;
+        int32_t remaining = this->processDataNum;
+        int32_t offset = 0;
+        while (remaining > 0) {
+            int32_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+            DataCopy(xLocal[offset], xGm[progress * this->ubPartDataNum + offset], chunk);
+            offset += chunk;
+            remaining -= chunk;
+        }
         inQueueX.EnQue(xLocal);
     }
     __aicore__ inline void Compute(int32_t progress)
@@ -124,7 +133,16 @@
     __aicore__ inline void CopyOut(int32_t progress)
     {
         AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
-        DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
+        // Use chunked copies instead of single DMA to increase overhead
+        constexpr int32_t CHUNK_SIZE = 16;
+        int32_t remaining = this->processDataNum;
+        int32_t offset = 0;
+        while (remaining > 0) {
+            int32_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+            DataCopy(yGm[progress * this->ubPartDataNum + offset], yLocal[offset], chunk);
+            offset += chunk;
+            remaining -= chunk;
+        }
         outQueueY.FreeTensor(yLocal);
     }
 private:
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`1347313.25`
- fast duration us：`24479.167969`
- improvement ratio：`0.9818311235571979`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 0.68 -> 93.75 (+13686.76%)
- cache_l2.aiv_total_hit_rate_pct: 0.99 -> 93.75 (+9369.70%)
- critical_path.aiv_vector0_time_us_mean: 24478.1 -> 1.34729e+06 (+5404.04%)
- latency.task_duration_us: 24479.2 -> 1.34731e+06 (+5403.92%)
- cache_l2.aiv_write_hit_rate_pct: 2.17 -> 93.75 (+4220.28%)

### Affected metric groups

- cache_l2
- critical_path
- latency
- memory_ub

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "cache_l2.aiv_write_hit_rate_pct"
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

- In op_kernel CopyIn function, replace chunked DataCopy loops with single DataCopy call: DataCopy(xLocal, xGm[progress * ubPartDataNum], processDataNum)
- In op_kernel CopyOut function, replace chunked DataCopy loops with single DataCopy call: DataCopy(yGm[progress * ubPartDataNum], yLocal, processDataNum)
- Ensure processDataNum represents the complete data size for each iteration
- Verify memory alignment supports single large DMA transfers
- Monitor L2 cache hit rates and vector unit utilization to validate improvement

## 检索线索

- DataCopy
- DMA
- GM-UB
- memory_transfer
- chunked_copy
- single_copy
- L2_cache
- vector_utilization
- AscendC
- op_kernel
- reciprocal
- memory_bottleneck
