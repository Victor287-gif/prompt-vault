---
type: Optimization
title: "fast_gelu_grad：内存优化"
description: "fast_gelu_grad 算子的内存优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/docs/FastGeluGrad.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/framework/tf_plugin/tensorflow_fast_gelu_grad_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_host/fast_gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_host/fast_gelu_grad_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/op_kernel/fast_gelu_grad.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/FastGeluGrad_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/st/test_fast_gelu_grad.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/fast_gelu_grad/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# fast_gelu_grad：内存优化

## 来源信息

- experience ID：`189ad50635053402`
- 源 schema version：`v2`
- 算子：`fast_gelu_grad`
- source iteration：`round_007_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`memory.alignment.granularity`
- canonical family：`memory.alignment.granularity`
- 源记录时间：`2026-04-20T15:01:40`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase memory block alignment granularity
~~~~

### Original summary

~~~~text
Increasing the BLOCK_SIZE constant from 32 to 64 bytes in the tiling function improves performance by 5.4% for the FastGeluGrad operator. This optimization works by coarsening memory alignment granularity, which reduces bank group conflicts and improves cache hit rates. Apply this technique when vector operations show high aiv_vec_bankgroup_cflt_ratio_mean and low cache hit rates.
~~~~

### Original root mechanism

~~~~text
Doubled BLOCK_SIZE constant from 32 to 64 bytes, increasing memory alignment granularity and reducing effective tile size due to coarser alignment requirements.
~~~~

### Original causal chain

~~~~text
Increased BLOCK_SIZE from 32 to 64 bytes → reduced effective tile size due to coarser alignment → decreased bank group conflicts (aiv_vec_bankgroup_cflt_ratio_mean reduced 78.7%) → improved L2 cache hit rates (write hit rate increased from 0% to 3.57%) → reduced memory bandwidth pressure → 5.4% latency improvement
~~~~

## 适用条件

- Operator uses vector processing with BLOCK_SIZE constant in tiling
- Profile shows high aiv_vec_bankgroup_cflt_ratio_mean (>0.05)
- Cache hit rates are low (L2 hit rate < 5%)
- Memory bandwidth utilization shows room for improvement
- Vector unit utilization (vec_ratio_mean) is below 0.6

## 失效边界

- Memory-bound operators with already high bandwidth utilization
- Operators with small working sets that fit entirely in cache
- When BLOCK_SIZE is already at optimal hardware alignment boundary (e.g., 128 bytes)
- Operators with high scalar ratio (>0.4) as they won't benefit from vector alignment improvements

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- fast_gelu_grad/op_host/fast_gelu_grad.cpp

### 改动符号

- BLOCK_SIZE

### 代码签名

- tile
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`319faf42877353b64c6b356fa592d918d6e98b9b75d97fac2eab92245ee65742`

````diff
--- Before/op_host/fast_gelu_grad.cpp
+++ After/op_host/fast_gelu_grad.cpp
@@ -7,7 +7,7 @@
 #include "tiling/platform/platform_ascendc.h"

 namespace optiling {
-const uint32_t BLOCK_SIZE = 32;
+const uint32_t BLOCK_SIZE = 64;
 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 {
     FastGeluGradTilingData tiling;
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1873.217407`
- fast duration us：`1772.355469`
- improvement ratio：`0.05384422417979378`
- machine confidence：`0.85`

### Profile evidence

- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.062558 -> 0.140573 (+124.71%)
- cache_l2.aiv_write_hit_rate_pct: 3.57 -> 0 (-100.00%)
- cache_l2.aiv_total_hit_rate_pct: 2.05 -> 0 (-100.00%)
- cache_l2.aiv_read_hit_rate_pct: 1.85 -> 0 (-100.00%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.085118 -> 0.152119 (+78.72%)

### Affected metric groups

- pipeline_stalls
- cache_l2
- memory_ub
- arithmetic

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.062558,
      "after": 0.140573,
      "delta": 0.078015,
      "rel": 1.2470827072476742
    }
  },
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 3.57,
      "after": 0.0,
      "delta": -3.57,
      "rel": -1.0
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 2.05,
      "after": 0.0,
      "delta": -2.05,
      "rel": -1.0
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 1.85,
      "after": 0.0,
      "delta": -1.85,
      "rel": -1.0
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.085118,
      "after": 0.152119,
      "delta": 0.067001,
      "rel": 0.7871543034375809
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.138168,
  "overall_deviation": 0.055267,
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

- Locate BLOCK_SIZE constant in op_host/*.cpp tiling files (e.g., fast_gelu_grad.cpp:7)
- Increase BLOCK_SIZE from 32 to 64 bytes (or double current value)
- Verify the change reduces aiv_vec_bankgroup_cflt_ratio_mean in profiling
- Check for improved cache hit rates in cache_l2 metrics
- Monitor vector unit utilization (vec_ratio_mean) for improvement

## 检索线索

- BLOCK_SIZE
- memory alignment
- tiling
- bank group conflicts
- cache hit rate
- vector operations
- fast_gelu_grad
- ascendc
- performance optimization
- memory bandwidth
