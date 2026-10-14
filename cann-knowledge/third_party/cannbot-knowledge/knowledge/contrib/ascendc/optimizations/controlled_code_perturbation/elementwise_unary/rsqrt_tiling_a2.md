---
type: Optimization
title: "rsqrt：Tiling优化"
description: "rsqrt 算子的Tiling优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/docs/Rsqrt.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/framework/tf_plugin/tensorflow_rsqrt_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_host/rsqrt.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_host/rsqrt_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/op_kernel/rsqrt.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/Rsqrt_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/st/test_rsqrt.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/rsqrt/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# rsqrt：Tiling优化

## 来源信息

- experience ID：`9d2d2f0a419f2cf9`
- 源 schema version：`v2`
- 算子：`rsqrt`
- source iteration：`round_016_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.memory.efficiency.halving`
- canonical family：`tiling.memory.efficiency.halving`
- 源记录时间：`2026-04-21T00:31:43`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Halve tile size by doubling ubDataNumber divisor in tiling calculation
~~~~

### Original summary

~~~~text
For vector operators like Rsqrt, halving tile size by doubling the ubDataNumber divisor from (4,8) to (8,16) reduces tile size by half and doubles loop iterations. This optimization improves performance by 15.7% by reducing memory bandwidth pressure and improving cache hit rates. Apply this technique when profiling shows high memory bandwidth usage and cache misses in L2 cache.
~~~~

### Original root mechanism

~~~~text
Doubling the ubDataNumber divisor in tiling calculation reduces tile size by half, improving memory access efficiency and reducing pipeline stalls
~~~~

### Original causal chain

~~~~text
Increased ubDataNumber divisor from (4,8) to (8,16) → tile size reduced by half → L2 cache hit rates increased (read: 1.6%→3.19%, write: 5.08%→9.67%) → vector wait ratio decreased from 0.453 to 0.429 → task duration reduced from 1729.69μs to 1457.89μs
~~~~

## 适用条件

- Operator uses vector processing with memory-intensive operations
- Profiling shows low L2 cache hit rates (<5%)
- Memory bandwidth usage is high relative to compute
- Tile size calculation uses ubDataNumber divisor pattern similar to rsqrt.cpp:25-27
- Operator has data type dependent tiling (DT_FLOAT vs other types)

## 失效边界

- Operator is compute-bound with high vector utilization (>70%)
- L2 cache hit rates are already high (>20%)
- Memory bandwidth usage is already optimized
- Tile size is already minimal for the data type

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- rsqrt/op_host/rsqrt.cpp

### 改动符号

- TilingFunc

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`b2a0ae5c0b102ea00177dd3f44eff3c6d038bec9be17fccc9e413c52a57d5e93`

````diff
--- Before/op_host/rsqrt.cpp
+++ After/op_host/rsqrt.cpp
@@ -25,7 +25,7 @@
     uint64_t inputLength = inputNum * typeLength;
     uint64_t inputBytes = inputLength / inputNum;

-    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? 4 : 8;
+    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? 8 : 16;
     uint64_t tileBlockNum = (ubSize / BLOCK_SIZE ) / ubDataNumber;
     uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1729.69458`
- fast duration us：`1457.88916`
- improvement ratio：`0.1571407016838777`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.032281 -> 0.069971 (+116.76%)
- cache_l2.aiv_read_hit_rate_pct: 1.6 -> 3.19 (+99.37%)
- cache_l2.aiv_total_hit_rate_pct: 2.32 -> 4.59 (+97.84%)
- cache_l2.aiv_write_hit_rate_pct: 5.08 -> 9.67 (+90.35%)
- aiv_utilization.scalar_ratio_mean: 0.092225 -> 0.138816 (+50.52%)

### Affected metric groups

- pipeline_stalls
- cache_l2
- aiv_utilization
- latency

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean",
  "cache_l2.aiv_read_hit_rate_pct",
  "cache_l2.aiv_total_hit_rate_pct",
  "cache_l2.aiv_write_hit_rate_pct",
  "aiv_utilization.scalar_ratio_mean"
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

- In the tiling function (TilingFunc in rsqrt.cpp), locate the ubDataNumber calculation
- Double the divisor values for both data type branches: from (4,8) to (8,16) or similar multiples
- Ensure tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber correctly computes smaller tiles
- Verify tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes produces appropriate values
- Test with profiling to confirm L2 cache hit rate improvements and reduced task duration

## 检索线索

- tiling
- ubDataNumber
- halving
- cache_hit_rate
- memory_efficiency
- vector_operator
- rsqrt
- pipeline_stalls
- L2_cache
- tile_size_reduction
