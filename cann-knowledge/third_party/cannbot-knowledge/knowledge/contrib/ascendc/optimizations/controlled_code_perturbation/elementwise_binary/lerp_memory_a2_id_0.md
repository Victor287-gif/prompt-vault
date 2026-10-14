---
type: Optimization
title: "lerp：内存优化"
description: "lerp 算子的内存优化经验，关注数据搬运。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
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
# lerp：内存优化

## 来源信息

- experience ID：`3fc503f4064c660d`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_010_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.access.pattern.dma_transfer`
- canonical family：`memory.access.pattern`
- 源记录时间：`2026-04-20T20:33:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use single DMA transfers instead of chunked copies for memory operations
~~~~

### Original summary

~~~~text
Applying single DMA transfers instead of chunked copies in CopyIn and CopyOut methods improves performance by 13.2% for the lerp operator. This optimization reduces memory transfer overhead and improves L2 cache hit rates from 2.8% to 93.75% by enabling more efficient memory access patterns. Use this technique when memory transfers dominate operator execution time.
~~~~

### Original root mechanism

~~~~text
Single DMA transfers enable efficient burst memory access patterns while chunked copies introduce overhead from multiple smaller transfers and reduce L2 cache hit rates.
~~~~

### Original causal chain

~~~~text
apply memory.access.pattern in lerp/op_kernel/lerp.cpp -> improve cache_l2.aiv_read_hit_rate_pct -> 13.2% latency improvement
~~~~

## 适用条件

- Operator uses AscendC DataCopy API for memory transfers
- CopyIn and CopyOut methods transfer data between GM and UB memory
- Memory transfer volume exceeds 32KB (gm_to_ub_kb > 32)
- L2 cache hit rates are below 50%
- aiv_utilization.mte2_ratio_mean or mte3_ratio_mean exceed 0.01

## 失效边界

- Memory transfers are already using single DMA calls
- Data must be processed in chunks due to UB capacity constraints
- Transfer sizes are smaller than 256 elements
- L2 cache hit rates are already above 90%

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- lerp/op_kernel/lerp.cpp

### 改动符号

- KernelLerp::CopyIn
- KernelLerp::CopyOut

### 代码签名

- AllocTensor
- DataCopy
- DeQue
- EnQue
- FreeTensor
- TQue
- datacopy
- queue_api
- ub

### 原始 diff

- 源 diff SHA-256：`30bcc2508a7d77c2522cb9626a8dabb40c2bd9eee73d27e56dfb5130db831872`

````diff
--- Before/op_kernel/lerp.cpp
+++ After/op_kernel/lerp.cpp
@@ -93,9 +93,20 @@
       AscendC::LocalTensor<TYPE_END> endLocal = inQueueEnd.AllocTensor<TYPE_END>();
       AscendC::LocalTensor<TYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<TYPE_WEIGHT>();

-      AscendC::DataCopy(startLocal, startGm[progress * this->ubPartDataNum], this->processDataNum);
-      AscendC::DataCopy(endLocal, endGm[progress * this->ubPartDataNum], this->processDataNum);
-      AscendC::DataCopy(weightLocal, weightGm[progress * this->ubPartDataNum], this->processDataNum);
+      // DEGRADATION: Copy data in small chunks instead of one large DMA transfer
+      // This breaks efficient memory transfers and increases overhead
+      const uint64_t CHUNK_SIZE = 16; // Copy 16 elements at a time
+      uint64_t remaining = this->processDataNum;
+      uint64_t offset = 0;
+
+      while (remaining > 0) {
+        uint64_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+        AscendC::DataCopy(startLocal[offset], startGm[progress * this->ubPartDataNum + offset], chunk);
+        AscendC::DataCopy(endLocal[offset], endGm[progress * this->ubPartDataNum + offset], chunk);
+        AscendC::DataCopy(weightLocal[offset], weightGm[progress * this->ubPartDataNum + offset], chunk);
+        offset += chunk;
+        remaining -= chunk;
+      }

       inQueueStart.EnQue(startLocal);
       inQueueEnd.EnQue(endLocal);
@@ -160,7 +171,20 @@
     __aicore__ inline void CopyOut(int32_t progress)
     {
       AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
-      AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
+
+      // DEGRADATION: Copy data in small chunks instead of one large DMA transfer
+      // This breaks efficient memory transfers and increases overhead
+      const uint64_t CHUNK_SIZE = 16; // Copy 16 elements at a time
+      uint64_t remaining = this->processDataNum;
+      uint64_t offset = 0;
+
+      while (remaining > 0) {
+        uint64_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
+        AscendC::DataCopy(yGm[progress * this->ubPartDataNum + offset], yLocal[offset], chunk);
+        offset += chunk;
+        remaining -= chunk;
+      }
+
       outQueueY.FreeTensor(yLocal);
     }
   }
````

## 观测记录

- bottleneck type：`memory_transfer_bound`
- slow duration us：`155521.5`
- fast duration us：`135004.4375`
- improvement ratio：`0.13192428378069912`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 2.39 -> 93.75 (+3822.59%)
- aiv_utilization.mte3_ratio_mean: 0.001349 -> 0.046454 (+3343.59%)
- cache_l2.aiv_total_hit_rate_pct: 2.8 -> 93.75 (+3248.21%)
- aiv_utilization.mte2_ratio_mean: 0.003219 -> 0.090722 (+2718.33%)
- cache_l2.aiv_write_hit_rate_pct: 7.43 -> 93.75 (+1161.78%)

### Affected metric groups

- cache_l2
- aiv_utilization
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "aiv_utilization.mte3_ratio_mean",
  "cache_l2.aiv_total_hit_rate_pct",
  "aiv_utilization.mte2_ratio_mean",
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

- Check lerp.cpp:93-107 for CopyIn method implementation
- Check lerp.cpp:160-174 for CopyOut method implementation
- Replace chunked DataCopy loops with single DataCopy calls using full processDataNum
- Ensure DataCopy uses contiguous memory regions: DataCopy(dst, src, count)
- Verify gm_to_ub_kb and ub_to_gm_kb metrics decrease after optimization
- Monitor L2 cache hit rates (cache_l2.aiv_total_hit_rate_pct) for improvement

## 检索线索

- DataCopy
- DMA
- memory transfer
- chunked copy
- CopyIn
- CopyOut
- AscendC
- L2 cache
- gm_to_ub
- ub_to_gm
- memory access pattern
