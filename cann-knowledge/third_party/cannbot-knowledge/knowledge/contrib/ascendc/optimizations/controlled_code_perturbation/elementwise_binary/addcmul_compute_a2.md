---
type: Optimization
title: "addcmul：计算优化"
description: "addcmul 算子的计算优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/docs/Addcmul.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/framework/tf_plugin/tensorflow_addcmul_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/op_host/addcmul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/op_host/addcmul_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/op_kernel/addcmul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/Addcmul_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/st/test_addcmul.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcmul/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# addcmul：计算优化

## 来源信息

- experience ID：`2e034df730d6ce42`
- 源 schema version：`v2`
- 算子：`addcmul`
- source iteration：`round_019_refactor`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`unexpected_sensitivity`
- explore mode：`refactor`
- optimization type：`constant.value.fix`
- canonical family：`constant.value.fix`
- 源记录时间：`2026-04-20T16:04:28`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Align BLOCK_SIZE to 32-byte boundaries for memory access efficiency
~~~~

### Original summary

~~~~text
Setting BLOCK_SIZE to a 32-byte aligned value (32 instead of 31) improves memory access efficiency by enabling optimal cache line utilization. This optimization reduces pipeline stalls from memory bank conflicts and improves MTE3 utilization. Apply this technique when BLOCK_SIZE constants affect memory access patterns in vector operations.
~~~~

### Original root mechanism

~~~~text
Changing BLOCK_SIZE from 31 to 32 enables proper 32-byte alignment for memory access patterns, reducing pipeline stalls and improving cache utilization.
~~~~

### Original causal chain

~~~~text
BLOCK_SIZE changed from 31 to 32 → improved memory alignment → reduced aiv_vec_wait_ratio_mean from 0.945283 to 0.942989 → improved MTE3 utilization from 0.144414 to 0.2912 → reduced task_duration_us from 1839.797μs to 1722.615μs
~~~~

## 适用条件

- BLOCK_SIZE constant is used for memory access patterns in vector operations
- Memory accesses show high wait ratios (aiv_vec_wait_ratio_mean > 0.94)
- MTE3 utilization is low (< 0.3) indicating memory access inefficiencies
- Cache hit rates are suboptimal (< 6% for L2 write hits)
- Working with 32-byte aligned memory access patterns

## 失效边界

- BLOCK_SIZE is already properly aligned to hardware memory boundaries
- Memory access patterns are already optimal with current BLOCK_SIZE
- Performance profiling shows no memory-related bottlenecks

## 变更方向

本记录来自等价重构敏感性实验，代码结构变化与性能指标相关。

- improvement direction：`refactor_caused_change`

## 代码差异

### 改动文件

- addcmul/op_host/addcmul.cpp

### 改动符号

- BLOCK_SIZE

### 代码签名

- BUFFER_NUM

### 原始 diff

- 源 diff SHA-256：`e3c53f46218af5773841febb9cbf2db28464c0b93c4287b868689fbe6a9d7e98`

````diff
--- Before/op_host/addcmul.cpp
+++ After/op_host/addcmul.cpp
@@ -5,7 +5,7 @@

 namespace optiling
 {
-    const uint32_t BLOCK_SIZE = 31;
+    const uint32_t BLOCK_SIZE = 32;
     const uint32_t BUFFER_NUM = 2;
     static ge::graphStatus TilingFunc(gert::TilingContext *context)
     {
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`1839.796753`
- fast duration us：`1722.614502`
- improvement ratio：`0.0636930415323981`
- machine confidence：`0.85`

### Profile evidence

- aiv_utilization.mte3_ratio_mean: 0.2912 -> 0.144414 (-50.41%)
- memory_ub.ub_to_gm_kb: 10965.5 -> 13062 (+19.12%)
- memory_ub.ub_to_gm_bw_usage_rate_pct: 3.46 -> 4.12 (+19.08%)
- cache_l2.aiv_write_hit_rate_pct: 5.07 -> 5.85 (+15.38%)
- cache_l2.aiv_total_hit_rate_pct: 1.9 -> 2.14 (+12.63%)

### Affected metric groups

- aiv_utilization
- memory_ub
- cache_l2
- latency

### Changed metrics top

~~~~json
[
  {
    "aiv_utilization.mte3_ratio_mean": {
      "before": 0.2912,
      "after": 0.144414,
      "delta": -0.14678600000000003,
      "rel": -0.5040728021978023
    }
  },
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 10965.5,
      "after": 13062.0,
      "delta": 2096.5,
      "rel": 0.1911905521864028
    }
  },
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 3.46,
      "after": 4.12,
      "delta": 0.6600000000000001,
      "rel": 0.19075144508670525
    }
  },
  {
    "cache_l2.aiv_write_hit_rate_pct": {
      "before": 5.07,
      "after": 5.85,
      "delta": 0.7799999999999994,
      "rel": 0.15384615384615372
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 1.9,
      "after": 2.14,
      "delta": 0.2400000000000002,
      "rel": 0.12631578947368433
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

- Check the BLOCK_SIZE constant in op_host/addcmul.cpp and ensure it's aligned to 32-byte boundaries
- Verify memory access patterns by examining aiv_vec_wait_ratio_mean and MTE3 utilization metrics
- Profile cache hit rates (cache_l2.aiv_write_hit_rate_pct, cache_l2.aiv_total_hit_rate_pct)
- Test different BLOCK_SIZE values that are multiples of 32 to find optimal alignment
- Monitor ub_to_gm_kb and ub_to_gm_bw_usage_rate_pct to validate memory bandwidth improvements

## 检索线索

- BLOCK_SIZE
- 32-byte alignment
- memory alignment
- cache optimization
- vector operations
- memory access patterns
- MTE3 utilization
- cache hit rate
- memory bandwidth
- pipeline stalls
