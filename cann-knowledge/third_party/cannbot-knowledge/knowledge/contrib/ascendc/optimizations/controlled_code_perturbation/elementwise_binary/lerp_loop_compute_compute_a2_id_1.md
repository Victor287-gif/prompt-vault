---
type: Optimization
title: "lerp：循环计算优化"
description: "lerp 算子的循环计算优化经验，关注数据搬运。"
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
# lerp：循环计算优化

## 来源信息

- experience ID：`9f2b6aeda0aea899`
- 源 schema version：`v2`
- 算子：`lerp`
- source iteration：`round_013_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`loop.unroll`
- canonical family：`loop.unroll`
- 源记录时间：`2026-04-20T21:09:12`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use single DataCopy calls instead of chunked copying loops
~~~~

### Original summary

~~~~text
In Ascend C operators, using single DataCopy calls for contiguous memory transfers instead of chunked copying loops improves performance by 8.9%. This optimization reduces loop overhead and enables the hardware to perform more efficient DMA transfers. Apply this pattern when copying contiguous data between global memory and local buffers in CopyIn/CopyOut functions.
~~~~

### Original root mechanism

~~~~text
Replacing while-loop chunked copying with single DataCopy calls eliminates loop overhead and enables efficient DMA transfers by allowing hardware to optimize memory transfers as single operations
~~~~

### Original causal chain

~~~~text
Replacing chunked while-loop copying with single DataCopy calls → reduces loop overhead and enables DMA optimization → improves cache hit rates from 2.4% to 93.75% and reduces memory transfer bandwidth usage → decreases task duration from 155521.5μs to 141668.9μs
~~~~

## 适用条件

- Copying contiguous data between global memory and local buffers in Ascend C operators
- Using DataCopy API with LocalTensor and GlobalTensor pointers
- Transferring data in CopyIn or CopyOut functions of kernel implementations
- When processDataNum represents the total number of elements to transfer

## 失效边界

- When data needs non-contiguous access patterns requiring strided transfers
- When memory regions have alignment constraints requiring chunked transfers
- When hardware documentation explicitly recommends chunked transfers for specific data sizes

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- lerp/op_kernel/lerp.cpp

### 改动符号

- KernelLerp::CopyIn

### 代码签名

- AllocTensor
- DataCopy
- EnQue
- datacopy
- queue_api
- ub

### 原始 diff

- 源 diff SHA-256：`4e76830b4784308bf9ee66317fd4cf525d3bd633ea1c1978a27f7b7fee495ff1`

````diff
--- Before/op_kernel/lerp.cpp
+++ After/op_kernel/lerp.cpp
@@ -93,20 +93,9 @@
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
+      AscendC::DataCopy(startLocal, startGm[progress * this->ubPartDataNum], this->processDataNum);
+      AscendC::DataCopy(endLocal, endGm[progress * this->ubPartDataNum], this->processDataNum);
+      AscendC::DataCopy(weightLocal, weightGm[progress * this->ubPartDataNum], this->processDataNum);

       inQueueStart.EnQue(startLocal);
       inQueueEnd.EnQue(endLocal);
````

## 观测记录

- bottleneck type：`memory_transfer_bound`
- slow duration us：`155521.5`
- fast duration us：`141668.90625`
- improvement ratio：`0.08907188877422093`
- machine confidence：`0.95`

### Profile evidence

- cache_l2.aiv_read_hit_rate_pct: 2.4 -> 93.75 (+3806.25%)
- aiv_utilization.mte2_ratio_mean: 0.003206 -> 0.090722 (+2729.76%)
- memory_ub.gm_to_ub_kb: 32960.8 -> 131072 (+297.66%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 0.11 -> 0.41 (+272.73%)
- memory_gm.aiv_main_mem_read_bw_gbps: 0.22 -> 0.81 (+268.18%)

### Affected metric groups

- cache_l2
- aiv_utilization
- memory_ub
- memory_gm

### Changed metrics top

~~~~json
[
  "cache_l2.aiv_read_hit_rate_pct",
  "aiv_utilization.mte2_ratio_mean",
  "memory_ub.gm_to_ub_kb",
  "memory_ub.gm_to_ub_bw_usage_rate_pct",
  "memory_gm.aiv_main_mem_read_bw_gbps"
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

- Replace while-loop chunked copying patterns with single DataCopy calls in CopyIn/CopyOut functions
- In lerp.cpp KernelLerp::CopyIn function, use AscendC::DataCopy(startLocal, startGm[progress * this->ubPartDataNum], this->processDataNum) instead of chunked loops
- Ensure DataCopy calls transfer the full processDataNum in one operation rather than splitting into CHUNK_SIZE chunks
- Verify that LocalTensor and GlobalTensor pointers are correctly aligned for single large transfers

## 检索线索

- DataCopy
- chunked copying
- while-loop
- memory transfer
- DMA optimization
- CopyIn
- CopyOut
- Ascend C
- lerp
- memory_transfer_bound
