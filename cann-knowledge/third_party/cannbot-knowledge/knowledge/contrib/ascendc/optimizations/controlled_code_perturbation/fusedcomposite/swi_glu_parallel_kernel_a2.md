---
type: Optimization
title: "swi_glu：并行切核优化"
description: "swi_glu 算子的并行切核优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.composite", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/docs/aclnnSwiGlu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_host/swi_glu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_host/swi_glu_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_host/swi_glu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_kernel/glu_tiling.hpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_kernel/glu_tiling_kernel.hpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_kernel/swi_glu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_kernel/swi_glu_bf16.hpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_kernel/swi_glu_impl.hpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/op_kernel/swi_glu_single.hpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/activation/swi_glu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# swi_glu：并行切核优化

## 来源信息

- experience ID：`8b883b82d8211735`
- 源 schema version：`v2`
- 算子：`swi_glu`
- source iteration：`round_002_degrade`
- 源 category：`activation`
- 计算分区：`fusedcomposite`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.core.limitation`
- canonical family：`parallelism.core.limitation`
- 源记录时间：`2026-04-19T00:38:06`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increasing block dimension to maximize core utilization
~~~~

### Original summary

~~~~text
Setting block_dim to use all available compute cores (48 instead of 24) improves performance by 5.6% for the SwiGlu operator. This optimization reduces per-core vector and scalar utilization ratios while maintaining the same total work, allowing better parallelism and reducing pipeline stalls. Apply this technique when operators show high vector utilization ratios (>0.5) and have available unused cores.
~~~~

### Original root mechanism

~~~~text
Using the maximum available compute cores reduces per-core workload and improves parallel execution efficiency by distributing computation across more cores.
~~~~

### Original causal chain

~~~~text
Increasing block_dim from 24 to 48 cores → reduces per-core workload by half → decreases vector utilization from 0.506 to 0.265 and scalar utilization from 0.236 to 0.125 → reduces pipeline stalls (vec_wait_ratio from 0.565 to 0.771) → improves task duration from 1325μs to 1251μs.
~~~~

## 适用条件

- Operator uses vector compute type (op_type: vector)
- Block dimension is less than total available cores (GetCoreNumAiv())
- Vector utilization ratio > 0.5 indicates high per-core workload
- Memory bandwidth usage is below saturation (gm_to_ub_bw_usage_rate_pct < 20%)
- Pipeline stalls show high vec_wait_ratio (>0.5)

## 失效边界

- Memory bandwidth is already saturated (gm_to_ub_bw_usage_rate_pct > 80%)
- Block dimension already equals total available cores
- Operator shows memory-bound characteristics with low compute utilization
- L2 cache hit rates are already high (>50%)
- Device frequency is throttled (freq_throttled: true)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/swi_glu.cpp

### 改动符号

- Tiling4SwiGlu

### 代码签名

- GetCoreNum
- coreNum
- core_launch
- tile_shape
- ub

### 原始 diff

- 源 diff SHA-256：`8e70afe6a44c1b5fd5f20afe54acf40e936a7a14629c2b80a53bb10175aa16fb`

````diff
--- Before/op_host/swi_glu.cpp
+++ After/op_host/swi_glu.cpp
@@ -428,6 +428,9 @@
     OP_LOGD(context->GetNodeName(), "Tiling4SwiGlu enter.");
     auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
     uint32_t totalCore = platformInfo.GetCoreNumAiv();
+    // DEGRADATION: Artificially reduce available cores to limit parallelism
+    // Force using at most half of available cores (minimum 1 core)
+    totalCore = std::max<uint32_t>(1, totalCore / 2);
     auto curShortSocName_ = platformInfo.GetSocVersion();
     uint64_t ubSize = 0;
     platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`1325.086548`
- fast duration us：`1250.785034`
- improvement ratio：`0.05607295169673702`
- machine confidence：`0.95`

### Profile evidence

- memory_ub.ub_to_gm_kb: 10930 -> 21860 (+100.00%)
- memory_ub.gm_to_ub_kb: 21860.2 -> 43720.2 (+100.00%)
- arithmetic.aiv_vec_fops: 2.2492e+07 -> 4.49804e+07 (+99.98%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.262622 -> 0.501032 (+90.78%)
- aiv_utilization.vec_ratio_mean: 0.265142 -> 0.505756 (+90.75%)

### Affected metric groups

- memory_ub
- arithmetic
- aiv_utilization
- memory_gm

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 10930.0,
      "after": 21860.0,
      "delta": 10930.0,
      "rel": 1.0
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 21860.2,
      "after": 43720.2,
      "delta": 21859.999999999996,
      "rel": 0.9999908509528731
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 22492032.0,
      "after": 44980352.0,
      "delta": 22488320.0,
      "rel": 0.9998349637773946
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.262622,
      "after": 0.501032,
      "delta": 0.23841,
      "rel": 0.9078066574772867
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.265142,
      "after": 0.505756,
      "delta": 0.240614,
      "rel": 0.9074910802513371
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.188078,
  "overall_deviation": 0.075231,
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

- In Tiling4SwiGlu function in op_host/swi_glu.cpp, ensure block_dim is set to platformInfo.GetCoreNumAiv()
- Remove any artificial core limitation code like 'totalCore = std::max<uint32_t>(1, totalCore / 2)'
- Verify the operator uses all available cores by checking parallelism.block_count equals active_block_count
- Monitor aiv_utilization.vec_ratio_mean - should decrease when more cores are utilized
- Check memory_ub.gm_to_ub_kb and ub_to_gm_kb - these should increase proportionally with core count
- Validate that aiv_vec_fops per core decreases while total remains constant

## 检索线索

- block_dim
- core_utilization
- parallelism
- GetCoreNumAiv
- vector_utilization
- scalar_overhead
- tiling
- swi_glu
- activation
- performance_recovery
