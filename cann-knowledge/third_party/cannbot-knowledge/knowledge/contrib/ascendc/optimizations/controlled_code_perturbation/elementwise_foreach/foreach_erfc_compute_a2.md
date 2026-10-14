---
type: Optimization
title: "foreach_erfc：边界计算优化"
description: "foreach_erfc 算子的边界计算优化经验，关注缓存冲突与访存模式。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/docs/aclnnForeachErfc.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/op_host/foreach_erfc.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_erfc/op_kernel/foreach_erfc.cpp
    role: implementation
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_erfc：边界计算优化

## 来源信息

- experience ID：`2471b254429335c0`
- 源 schema version：`v2`
- 算子：`foreach_erfc`
- source iteration：`round_003_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`tensor.boundary.ignorance`
- canonical family：`tensor.boundary.ignorance`
- 源记录时间：`2026-04-19T04:00:02`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Enable tensor boundary offset calculations in foreach loops
~~~~

### Original summary

~~~~text
In foreach operators, properly handling tensor boundary offsets by checking `tensorStart` and `tensorEnd` conditions with their corresponding offsets ensures correct memory access patterns. This optimization reduces cache bank conflicts by 23x and vector pipeline stall ratios by 23x, improving L2 cache hit rates from 8.54% to 95.19%. Apply this technique to any foreach operator that processes segmented tensor data with start/end boundaries.
~~~~

### Original root mechanism

~~~~text
Proper tensor boundary offset calculations prevent incorrect memory access patterns that cause cache bank conflicts and excessive pipeline stalls.
~~~~

### Original causal chain

~~~~text
Enable tensor boundary offset checks -> reduces incorrect memory accesses -> lowers cache bank conflict ratios (0.006305 to 0.151898) and bank group conflict ratios (0.004299 to 0.103567) -> decreases vector pipeline stalls -> reduces task duration from 113674.633us to 2379.128us
~~~~

## 适用条件

- Operator uses foreach pattern with segmented tensor processing
- Operator accesses tensor data with start/end boundaries
- Operator shows high aiv_vec_bank_cflt_ratio_mean (>0.04) or aiv_vec_bankgroup_cflt_ratio_mean (>0.10)
- L2 cache hit rates are below 20%
- Vector pipeline stalls (aiv_vec_total_cflt_ratio_mean) exceed 0.10

## 失效边界

- Operator processes contiguous tensor data without segmentation
- Tensor boundaries are already properly handled in the kernel
- Memory access patterns are already optimized for cache locality
- L2 cache hit rates are above 80%

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_erfc/op_kernel/foreach_erfc.cpp

### 改动符号

- KernelForeachUnary::Process

### 原始 diff

- 源 diff SHA-256：`9bb22282a141a47c3df9c6308a314718ba8c108500174787a496ec2663220ed0`

````diff
--- Before/op_kernel/foreach_erfc.cpp
+++ After/op_kernel/foreach_erfc.cpp
@@ -170,12 +170,15 @@
         int64_t cursorStart = 0;
         int64_t cursorEnd = Base::tensorDataCountList[i] - 1;
         int64_t dataCount = 0;
-        if (i == Base::tensorStart) {
-            cursorStart = Base::tensorStartOffset;
-        }
-        if (i == Base::tensorEnd) {
-            cursorEnd = Base::tensorEndOffset;
-        }
+        // Degradation: Remove tensor boundary awareness by ignoring start/end offsets
+        // This treats all tensors as if they start from beginning, causing incorrect
+        // memory access patterns and potential cache inefficiencies
+        // if (i == Base::tensorStart) {
+        //     cursorStart = Base::tensorStartOffset;
+        // }
+        // if (i == Base::tensorEnd) {
+        //     cursorEnd = Base::tensorEndOffset;
+        // }

         dataCount = cursorEnd - cursorStart + 1;
````

## 观测记录

- bottleneck type：`cache_conflict_and_memory_access_pattern`
- slow duration us：`227350.289062`
- fast duration us：`4757.015137`
- improvement ratio：`0.9790762740763319`
- machine confidence：`0.9`

### Profile evidence

- latency.task_duration_us: 2379.13 -> 113675 (+4678.00%)
- critical_path.aiv_vector0_time_us_mean: 2376.54 -> 58023.8 (+2341.53%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.002006 -> 0.048331 (+2309.32%)
- pipeline_stalls.aiv_vec_total_cflt_ratio_mean: 0.006305 -> 0.151898 (+2309.17%)
- pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean: 0.004299 -> 0.103567 (+2309.10%)

### Affected metric groups

- latency
- critical_path
- pipeline_stalls
- arithmetic

### Changed metrics top

~~~~json
[
  {
    "latency.task_duration_us": {
      "before": 2379.128,
      "after": 113674.633,
      "delta": 111295.505,
      "rel": 46.77995677407857
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2376.54,
      "after": 58023.819,
      "delta": 55647.279,
      "rel": 23.41525032189654
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.002006,
      "after": 0.048331,
      "delta": 0.046325,
      "rel": 23.093220338983052
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.006305,
      "after": 0.151898,
      "delta": 0.145593,
      "rel": 23.09167327517843
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.004299,
      "after": 0.103567,
      "delta": 0.09926800000000001,
      "rel": 23.090951384042803
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 121856768.0,
      "after": 2924534656.0,
      "delta": 2802677888.0,
      "rel": 22.999772060260124
    }
  },
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 11037.8,
      "after": 264903.4,
      "delta": 253865.60000000003,
      "rel": 22.999655728496624
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 11039.1,
      "after": 264904.8,
      "delta": 253865.69999999998,
      "rel": 22.99695627360926
    }
  },
  {
    "cache_l2.aiv_read_hit_rate_pct": {
      "before": 6.07,
      "after": 95.08,
      "delta": 89.00999999999999,
      "rel": 14.663920922570014
    }
  },
  {
    "cache_l2.aiv_total_hit_rate_pct": {
      "before": 8.54,
      "after": 95.19,
      "delta": 86.65,
      "rel": 10.146370023419205
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

- In foreach_erfc.cpp KernelForeachUnary::Process function, ensure tensor boundary offset calculations are enabled
- Check for conditions `if (i == Base::tensorStart)` and `if (i == Base::tensorEnd)` to set cursorStart and cursorEnd offsets
- Verify cursorStart is set to Base::tensorStartOffset for the first tensor segment
- Verify cursorEnd is set to Base::tensorEndOffset for the last tensor segment
- Calculate dataCount as cursorEnd - cursorStart + 1 after applying boundary offsets
- Apply this pattern to any foreach operator with similar tensor segmentation logic

## 检索线索

- foreach
- erfc
- tensor
- boundary
- offset
- cache
- conflict
- pipeline
- stall
- memory
- access
- pattern
- segmentation
- cursor
- start
- end
