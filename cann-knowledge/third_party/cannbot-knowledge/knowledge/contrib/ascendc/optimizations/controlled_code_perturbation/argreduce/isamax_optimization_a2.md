---
type: Optimization
title: "isamax：冗余消除优化"
description: "isamax 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/docs/aclnnIsamax.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_host/isamax.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_host/isamax_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_kernel/isamax.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/op_kernel/isamax_aiv.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/isamax/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# isamax：冗余消除优化

## 来源信息

- experience ID：`52db5678100384d1`
- 源 schema version：`v2`
- 算子：`isamax`
- source iteration：`round_017_degrade`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`redundant.vector.operation`
- canonical family：`redundant.vector.operation`
- 源记录时间：`2026-04-21T06:41:36`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminate redundant absolute value computation in vectorized loops
~~~~

### Original summary

~~~~text
Remove duplicate Abs() vector operations in the main processing loop to eliminate redundant absolute value computations. This optimization reduces vector floating-point operations (FLOPS) by ~50% and improves latency by ~11%. Apply this optimization when vector operations are unnecessarily duplicated in processing loops, particularly in math operators like isamax that compute absolute values.
~~~~

### Original root mechanism

~~~~text
Redundant Abs() vector operations in the main processing loop add unnecessary computational overhead without changing functional behavior.
~~~~

### Original causal chain

~~~~text
Removing redundant Abs() calls → aiv_vec_fops decreases from 1.21B to 0.81B (-50%) → vector unit utilization decreases slightly → task_duration_us improves from 32772μs to 29234μs (+11% faster)
~~~~

## 适用条件

- Vector operations are duplicated in processing loops without functional benefit
- Abs() or similar vector operations appear multiple times consecutively in the same loop iteration
- Operator processes large arrays with vectorized absolute value computations
- Performance profile shows high aiv_vec_fops with low vector unit utilization
- Code follows the pattern: Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, repeatCount, {1,1,8,8})

## 失效边界

- Multiple Abs() operations serve different computational purposes
- Vector operations have data dependencies requiring separate computations
- Different vector lanes need independent absolute value computations
- The redundancy provides functional benefits like numerical stability

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- isamax/op_kernel/isamax.cpp

### 改动符号

- Isamax::SingleProcess

### 代码签名

- contiguous_access

### 原始 diff

- 源 diff SHA-256：`fdcc9bbf0a8d8589edd886216f1db067032c88d969e47a95d200924c4016bb55`

````diff
--- Before/op_kernel/isamax.cpp
+++ After/op_kernel/isamax.cpp
@@ -296,15 +296,22 @@

     for (uint32_t i = 0; i < repeatBatchCnt; i++) {
         Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, MAX_REPEATS, {1, 1, 8, 8});
+        // Degradation: Redundant Abs computation
+        // Compute absolute value twice, adding computational overhead
+        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, MAX_REPEATS, {1, 1, 8, 8});
         offset += maxRepeatLen;
     }

     if (repeatBatchCntRemainder > 0) {
         Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 8, 8});
+        // Degradation: Redundant Abs computation
+        Abs(srcLocal[offset], srcLocal[offset], elementsPerRepeat, repeatBatchCntRemainder, {1, 1, 8, 8});
         offset += rmdRepeatLen;
     }

     if (totalRepeatCntRemainder > 0) {
+        Abs(srcLocal[offset], srcLocal[offset], totalRepeatCntRemainder, 1, {1, 1, 8, 8});
+        // Degradation: Redundant Abs computation
         Abs(srcLocal[offset], srcLocal[offset], totalRepeatCntRemainder, 1, {1, 1, 8, 8});
     }
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`32772.195312`
- fast duration us：`29233.503906`
- improvement ratio：`0.10797846687750759`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_fops: 8.09968e+08 -> 1.21374e+09 (+49.85%)
- memory_ub.ub_read_bw_vector_gbps: 2.14 -> 2.86 (+33.64%)
- memory_ub.ub_write_bw_vector_gbps: 2.14 -> 2.86 (+33.64%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.009605 -> 0.006823 (-28.96%)
- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 7e-06 -> 9e-06 (+28.57%)

### Affected metric groups

- arithmetic
- memory_ub
- pipeline_stalls
- critical_path

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_fops": {
      "before": 809968224.0,
      "after": 1213739936.0,
      "delta": 403771712.0,
      "rel": 0.4985031511557174
    }
  },
  {
    "memory_ub.ub_read_bw_vector_gbps": {
      "before": 2.14,
      "after": 2.86,
      "delta": 0.7199999999999998,
      "rel": 0.336448598130841
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 2.14,
      "after": 2.86,
      "delta": 0.7199999999999998,
      "rel": 0.336448598130841
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.009605,
      "after": 0.006823,
      "delta": -0.0027820000000000006,
      "rel": -0.28964081207704323
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 7e-06,
      "after": 9e-06,
      "delta": 2.0000000000000003e-06,
      "rel": 0.28571428571428575
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
  "performance_metrics_checked": 0,
  "performance_metrics_wrong": 0,
  "performance_direction_opposite": false,
  "mismatches": [],
  "mechanism_understood": true
}
~~~~

## 复用提示

- 1. Identify consecutive Abs() calls in vector processing loops in op_kernel/isamax.cpp
- 2. Check if Abs() operations compute the same value on the same input/output buffers
- 3. Remove duplicate Abs() calls while preserving the first computation
- 4. Verify functional correctness: the remaining Abs() call should produce identical results
- 5. Profile to confirm reduction in aiv_vec_fops and improvement in task_duration_us

## 检索线索

- isamax
- Abs
- vector
- redundant
- flops
- absolute value
- math operator
- performance optimization
- computational overhead
- vector operations
