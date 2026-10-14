---
type: Optimization
title: "icamin：流水优化"
description: "icamin 算子的流水优化经验，关注混合或尚未确定的瓶颈。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.compute.reduction", "topic.implementation.synchronization", "topic.implementation.pipeline", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/docs/aclnnIcamin.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_host/icamin.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_host/icamin_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_kernel/icamin.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/op_kernel/icamin_aiv.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/icamin/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# icamin：流水优化

## 来源信息

- experience ID：`25caaf298f2e4b3c`
- 源 schema version：`v2`
- 算子：`icamin`
- source iteration：`round_009_degrade`
- 源 category：`math`
- 计算分区：`argreduce`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.barrier.hotloop`
- canonical family：`pipeline.barrier.hotloop`
- 源记录时间：`2026-04-20T20:27:05`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Eliminating unnecessary pipeline barriers in hot loops
~~~~

### Original summary

~~~~text
Applying AscendC::PipeBarrier<PIPE_V>() after each Abs() call in computation loops introduces unnecessary pipeline synchronization that breaks instruction-level parallelism. This optimization improves vector utilization by 7.79% (vec_ratio_mean from 0.694518 to 0.753198) and reduces task duration by 6.5% (6096.2μs to 5698.3μs). Use this technique when vector operations in tight loops don't have data dependencies requiring explicit synchronization.
~~~~

### Original root mechanism

~~~~text
Removing AscendC::PipeBarrier<PIPE_V>() calls after Abs() operations in vector computation loops eliminates forced pipeline synchronization, allowing better instruction-level parallelism.
~~~~

### Original causal chain

~~~~text
Removing PipeBarrier<PIPE_V>() calls → reduces pipeline wait ratio from 0.057762 to 0.062194 (-7.13%) → improves vector utilization by 7.79% → reduces aiv_vector0_time_us_mean from 6091.221 to 5693.754 (-6.5%) → reduces overall task_duration_us from 6096.202 to 5698.334 (-6.5%).
~~~~

## 适用条件

- Ascend C operators with vector computation loops using Abs() or similar vector operations
- Operators with low vector utilization (vec_ratio_mean < 0.75) despite sufficient computational work
- Code patterns with AscendC::PipeBarrier<PIPE_V>() calls immediately after vector operations in loops
- Operators showing high aiv_vec_wait_ratio_mean (>0.05) indicating pipeline stalls

## 失效边界

- When vector operations have true data dependencies requiring synchronization
- When using different pipeline types (not PIPE_V)
- When working with memory operations that require explicit barrier synchronization
- When debugging race conditions or memory consistency issues

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/icamin.cpp

### 改动符号

- Icamin::SingleProcess

### 代码签名

- PipeBarrier
- contiguous_access
- pipebarrier

### 原始 diff

- 源 diff SHA-256：`c1d65bdac177739aca597f902fc7d9b5f3a092b78cf24e342e9cfe14672001df`

````diff
--- Before/op_kernel/icamin.cpp
+++ After/op_kernel/icamin.cpp
@@ -336,6 +336,7 @@
             Abs(srcLocal[offset + repeatInBatch * elementsPerRepeat],
                 srcLocal[offset + repeatInBatch * elementsPerRepeat],
                 elementsPerRepeat, 1, {1, 1, 8, 8});
+            AscendC::PipeBarrier<PIPE_V>();
         } else if (i < repeatBatchCnt * MAX_REPEATS + repeatBatchCntRemainder) {
             // Remainder batch iterations
             uint32_t remainderIdx = i - repeatBatchCnt * MAX_REPEATS;
@@ -345,10 +346,12 @@
             Abs(srcLocal[offset + remainderIdx * elementsPerRepeat],
                 srcLocal[offset + remainderIdx * elementsPerRepeat],
                 elementsPerRepeat, 1, {1, 1, 8, 8});
+            AscendC::PipeBarrier<PIPE_V>();
         } else {
             // Tail iteration
             offset = repeatBatchCnt * maxRepeatLen + rmdRepeatLen;
             Abs(srcLocal[offset], srcLocal[offset], totalRepeatCntRemainder, 1, {1, 1, 8, 8});
+            AscendC::PipeBarrier<PIPE_V>();
         }
     }
````

## 观测记录

- bottleneck type：`mixed_or_unknown`
- slow duration us：`6096.202148`
- fast duration us：`5698.333984`
- improvement ratio：`0.0652649230358167`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.463984 -> 0.427836 (-7.79%)
- aiv_utilization.vec_ratio_mean: 0.753198 -> 0.694518 (-7.79%)
- arithmetic.aiv_vec_fp32_ratio_mean: 0.289214 -> 0.266682 (-7.79%)
- memory_ub.ub_write_bw_scalar_gbps: 0.0026 -> 0.0024 (-7.69%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.062194 -> 0.057762 (-7.13%)

### Affected metric groups

- arithmetic
- aiv_utilization
- memory_ub
- pipeline_stalls

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.463984,
      "after": 0.427836,
      "delta": -0.03614800000000001,
      "rel": -0.07790785889168594
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.753198,
      "after": 0.694518,
      "delta": -0.058680000000000065,
      "rel": -0.07790780113595637
    }
  },
  {
    "arithmetic.aiv_vec_fp32_ratio_mean": {
      "before": 0.289214,
      "after": 0.266682,
      "delta": -0.022532000000000052,
      "rel": -0.07790770847884283
    }
  },
  {
    "memory_ub.ub_write_bw_scalar_gbps": {
      "before": 0.0026,
      "after": 0.0024,
      "delta": -0.0002000000000000001,
      "rel": -0.07692307692307696
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.062194,
      "after": 0.057762,
      "delta": -0.0044319999999999984,
      "rel": -0.07126089333376208
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.396437,
  "overall_deviation": 0.158575,
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

- 1. Search for AscendC::PipeBarrier<PIPE_V>() calls in op_kernel/*.cpp files, particularly in computation loops
- 2. Examine the SingleProcess() function in Icamin operator (or similar vector operators) for barrier placement
- 3. Remove PipeBarrier<PIPE_V>() calls that follow vector operations like Abs() when no data dependencies exist
- 4. Verify improvement by checking aiv_utilization.vec_ratio_mean and pipeline_stalls.aiv_vec_wait_ratio_mean metrics
- 5. Ensure vector operations remain correct by checking arithmetic.aiv_vec_fp32_ratio_mean and arithmetic.aiv_vec_misc_ratio_mean

## 检索线索

- PipeBarrier
- pipeline
- barrier
- hotloop
- vector
- utilization
- AscendC
- PIPE_V
- Abs
- synchronization
- icamin
- math
- performance
- optimization
