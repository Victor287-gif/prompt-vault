---
type: Optimization
title: "lerp：搬运与内存优化"
description: "lerp 算子的搬运与内存优化经验，关注数据搬运。"
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
# lerp：搬运与内存优化

## 来源信息

- experience ID：`8b2e05e86d04f2fd`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_014_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`memory.copy.optimization`
- canonical family：`memory.copy.optimization`
- 源记录时间：`2026-04-20T21:36:49`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Replace chunked memory copies with single DMA transfers
~~~~

### Original summary

~~~~text
Applying single DataCopy operations instead of loop-based chunked copying reduces memory transfer overhead by 13.2% in the lerp operator. This optimization consolidates multiple small memory transfers into single efficient DMA operations, improving L2 cache hit rates from 2.8% to 93.75% and reducing MTE utilization from 9.1% to 0.3%. Use this pattern when transferring contiguous memory blocks between GM and UB in Ascend C kernels.
~~~~

### Original root mechanism

~~~~text
Single DMA transfers enable hardware prefetching and reduce cache line fragmentation compared to small chunked copies
~~~~

### Original causal chain

~~~~text
Replacing 16-element chunked copies with single DataCopy operations → reduces GM to UB transfer count from 131072KB to 32960KB → decreases L2 cache misses → reduces MTE2/MTE3 utilization from 9.1%/4.6% to 0.3%/0.1% → improves task duration from 155521us to 135004us
~~~~

## 适用条件

- Transferring contiguous memory blocks between GM and UB using AscendC::DataCopy
- Processing large tensors where processDataNum > 16 elements
- Using TQue operations with LocalTensor allocations in CopyIn/CopyOut functions
- Working with vector operators that have memory-bound characteristics

## 失效边界

- Transferring non-contiguous memory regions requiring strided access
- Processing small tensors where chunk size equals total data size
- Memory transfers that require explicit synchronization between chunks
- When hardware DMA engine cannot handle the full transfer size

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

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

- 源 diff SHA-256：`d17f5694334b6991df1a0d9bac5c7c643b42926b9e9de162198c8a0288f2ea73`

````diff
--- Before/op_kernel/lerp.cpp
+++ After/op_kernel/lerp.cpp
@@ -93,20 +93,10 @@
       AscendC::LocalTensor<TYPE_END> endLocal = inQueueEnd.AllocTensor<TYPE_END>();
       AscendC::LocalTensor<TYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<TYPE_WEIGHT>();

-      // DEGRADATION: Copy data in small chunks instead of one large DMA transfer
-      // This breaks efficient memory transfers and increases overhead
-      const uint64_t CHUNK_SIZE = 16; // Copy 16 elements at a time
-      uint64_t remaining = this->processDataNum;
-      uint64_t offset = 0;
-
-      while (remaining > 0) {
-        uint64_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
-        AscendC::DataCopy(startLocal[offset], startGm[progress * this->ubPartDataNum + offset], chunk);
-        AscendC::DataCopy(endLocal[offset], endGm[progress * this->ubPartDataNum + offset], chunk);
-        AscendC::DataCopy(weightLocal[offset], weightGm[progress * this->ubPartDataNum + offset], chunk);
-        offset += chunk;
-        remaining -= chunk;
-      }
+      // Use single efficient DMA transfer instead of chunked copying
+      AscendC::DataCopy(startLocal, startGm[progress * this->ubPartDataNum], this->processDataNum);
+      AscendC::DataCopy(endLocal, endGm[progress * this->ubPartDataNum], this->processDataNum);
+      AscendC::DataCopy(weightLocal, weightGm[progress * this->ubPartDataNum], this->processDataNum);

       inQueueStart.EnQue(startLocal);
       inQueueEnd.EnQue(endLocal);
@@ -172,18 +162,8 @@
     {
       AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();

-      // DEGRADATION: Copy data in small chunks instead of one large DMA transfer
-      // This breaks efficient memory transfers and increases overhead
-      const uint64_t CHUNK_SIZE = 16; // Copy 16 elements at a time
-      uint64_t remaining = this->processDataNum;
-      uint64_t offset = 0;
-
-      wh... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`memory_transfer_bound`
- slow duration us：`155521.5`
- fast duration us：`135004.03125`
- improvement ratio：`0.13192689595972262`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 2.39 -> 93.75 (+3822.59%)
- aiv_utilization.mte3_ratio_mean: 0.001347 -> 0.046454 (+3348.70%)
- cache_l2.aiv_total_hit_rate_pct: 2.8 -> 93.75 (+3248.21%)
- aiv_utilization.mte2_ratio_mean: 0.003222 -> 0.090722 (+2715.70%)
- cache_l2.aiv_write_hit_rate_pct: 7.43 -> 93.75 (+1161.78%)

### Affected metric groups

- cache_l2
- aiv_utilization
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 2.39,
      "after": 93.75,
      "delta": 91.36,
      "rel": 38.22594142259414
    }
  },
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.001347,
      "after": 0.046454,
      "delta": 0.045107,
      "rel": 33.48700816629547
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 2.8,
      "after": 93.75,
      "delta": 90.95,
      "rel": 32.48214285714286
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.003222,
      "after": 0.090722,
      "delta": 0.0875,
      "rel": 27.157045313469894
    }
  },
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 7.43,
      "after": 93.75,
      "delta": 86.32,
      "rel": 11.617765814266487
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

- Replace loop-based chunked copying patterns in lerp.cpp CopyIn/CopyOut functions
- Use single AscendC::DataCopy calls with full processDataNum instead of CHUNK_SIZE loops
- Apply to all AscendC kernels using similar LocalTensor allocations from TQue
- Check for patterns where remaining/offset variables track chunked transfers
- Verify contiguous memory access patterns before applying this optimization

## 检索线索

- DataCopy
- DMA
- memory_transfer
- chunked_copy
- L2_cache
- MTE_utilization
- AscendC
- TQue
- LocalTensor
- CopyIn
- CopyOut
- lerp
- vector_operator
