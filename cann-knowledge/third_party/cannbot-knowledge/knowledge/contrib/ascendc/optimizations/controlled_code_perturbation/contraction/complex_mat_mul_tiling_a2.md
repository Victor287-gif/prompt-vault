---
type: Optimization
title: "complex_mat_mul：Tiling优化"
description: "complex_mat_mul 算子的Tiling优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.tiling", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/docs/ComplexMatMul.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/op_host/complex_mat_mul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/op_host/complex_mat_mul_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/op_kernel/complex_mat_mul.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/matmul/complex_mat_mul/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# complex_mat_mul：Tiling优化

## 来源信息

- experience ID：`5f2588f7ca57d8ee`
- 源 schema version：`v2`
- 算子：`complex_mat_mul`
- source iteration：`round_011_degrade`
- 源 category：`matmul`
- 计算分区：`contraction`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tiling.search.initialization`
- canonical family：`tiling.search.suboptimal`
- 源记录时间：`2026-04-21T01:50:57`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Full core count initialization for tiling search
~~~~

### Original summary

~~~~text
When performing tiling search for Ascend C operators, initialize the search with the maximum available core count rather than a reduced subset. This allows the tiling algorithm to explore higher parallelism configurations, reducing scalar overhead and improving overall performance by 19.2% in complex_mat_mul operations.
~~~~

### Original root mechanism

~~~~text
Using full available core count for tiling search initialization enables discovery of optimal parallelism configurations that better utilize hardware resources.
~~~~

### Original causal chain

~~~~text
Using full core count in tiling search -> discovers higher block_dim configuration (12→24) -> reduces scalar instruction ratio (0.975→0.762) -> decreases AIV vector operations (10740928→5423488) -> lowers memory bandwidth usage -> improves latency from 2118μs to 1712μs
~~~~

## 适用条件

- When implementing Ascend C operators with tiling search algorithms
- When using ascendcPlatform.GetCoreNumAiv() or similar core count APIs
- When scalar overhead is identified as the bottleneck (scalar_ratio_mean > 0.9)
- When block_dim can be increased without exceeding hardware limits

## 失效边界

- When memory bandwidth is already saturated (memory_ub.gm_to_ub_bw_usage_rate_pct > 80%)
- When AIC cube utilization is already high (aic_utilization.cube_ratio_mean > 0.8)
- When L2 cache hit rates are low (cache_l2.aic_total_hit_rate_pct < 90%)

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- complex_mat_mul/op_host/complex_mat_mul.cpp

### 改动符号

- TilingFunc

### 代码签名

- GetCoreNum
- TBuf
- coreNum
- core_launch
- tbuf
- ub

### 原始 diff

- 源 diff SHA-256：`94e7b3b65c13546b8fdaf0663c52a7b6ecf34e0d88bfd194d322c465f421cc7a`

````diff
--- Before/op_host/complex_mat_mul.cpp
+++ After/op_host/complex_mat_mul.cpp
@@ -80,6 +80,9 @@

     tiling.SetBufferSpace(-1, -1, -1);
     uint32_t n_core = ascendcPlatform.GetCoreNumAiv();
+    // Start with half the cores to reduce parallelism search space
+    n_core = n_core / 2;
+    if (n_core == 0) n_core = 1;
     tiling.SetDim(n_core);

     while (n_core > 0 && tiling.GetTiling(tiling1.cubeTilingData) == -1)
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2118.622314`
- fast duration us：`1712.734253`
- improvement ratio：`0.19158113190721357`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.gm_to_ub_kb: 5949.1 -> 11830.4 (+98.86%)
- arithmetic.aiv_vec_fops: 5.42349e+06 -> 1.07409e+07 (+98.04%)
- memory_ub.ub_to_gm_kb: 5423.8 -> 10738.8 (+97.99%)
- memory_gm.write_main_memory_kb: 4367.6 -> 7270.1 (+66.46%)
- arithmetic.aic_cube_fp16_ratio: 0.27106 -> 0.445781 (+64.46%)

### Affected metric groups

- memory_ub
- arithmetic
- memory_gm
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 5949.1,
      "after": 11830.4,
      "delta": 5881.299999999999,
      "rel": 0.9886033181489635
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 5423488.0,
      "after": 10740928.0,
      "delta": 5317440.0,
      "rel": 0.980446531826013
    }
  },
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 5423.8,
      "after": 10738.8,
      "delta": 5314.999999999999,
      "rel": 0.9799402632840442
    }
  },
  {
    "memory_gm.write_main_memory_kb": {
      "before": 4367.6,
      "after": 7270.1,
      "delta": 2902.5,
      "rel": 0.6645526147083066
    }
  },
  {
    "arithmetic.aic_cube_fp16_ratio": {
      "before": 0.27106,
      "after": 0.445781,
      "delta": 0.17472099999999996,
      "rel": 0.644584224894857
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

- In the TilingFunc implementation, use the maximum available core count from ascendcPlatform.GetCoreNumAiv() without reduction
- Set tiling.SetDim() with the full core count value before calling GetTiling()
- Verify that the increased block_dim (from 12 to 24 in this case) improves parallelism without causing resource contention
- Monitor scalar_ratio_mean metric - it should decrease with better parallelism (0.975→0.762 in this case)
- Check that aiv_vec_fops decreases with improved tiling (10740928→5423488 in this case)

## 检索线索

- tiling
- core_count
- parallelism
- scalar_overhead
- block_dim
- ascendcPlatform
- GetCoreNumAiv
- SetDim
- GetTiling
- complex_mat_mul
