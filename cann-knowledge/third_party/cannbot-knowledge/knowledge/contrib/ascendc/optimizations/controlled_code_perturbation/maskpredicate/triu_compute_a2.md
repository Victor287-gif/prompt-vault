---
type: Optimization
title: "triu：计算优化"
description: "triu 算子的计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/docs/Triu.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/framework/tf_plugin/tensorflow_triu_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_host/triu_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/op_kernel/triu.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/Triu_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/st/test_triu.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/triu/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# triu：计算优化

## 来源信息

- experience ID：`39a1e4db11672b79`
- 源 schema version：`v2`
- 算子：`triu`
- source iteration：`round_019_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`operation.splitting.degradation`
- canonical family：`operation.splitting.degradation`
- 源记录时间：`2026-04-21T02:31:08`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Avoid splitting vectorized operations into small chunks
~~~~

### Original summary

~~~~text
When implementing the triu operator in Ascend C, avoid splitting single vectorized Sub operations into multiple smaller calls (e.g., 32-element chunks). This optimization maintains efficient vectorization by keeping operations contiguous, reducing function call overhead and pipeline stalls. Apply this pattern to vector operations processing contiguous memory regions in triu.cpp and similar matrix operations.
~~~~

### Original root mechanism

~~~~text
Splitting large vector operations into multiple small function calls increases instruction overhead and reduces vectorization efficiency, causing pipeline stalls and higher latency.
~~~~

### Original causal chain

~~~~text
Replacing single vectorized Sub operation with multiple smaller Sub calls → increased instruction count (aiv_vec_fops +228%) → reduced vector unit utilization (vec_ratio_mean +150%) → higher pipeline wait ratios (aiv_vec_wait_ratio_mean +661%) → 7.8x latency increase (18433μs vs 2361μs)
~~~~

## 适用条件

- Implementing vector operations in Ascend C with contiguous memory access patterns
- Processing large contiguous data segments in matrix operations like triu
- Using vector intrinsics (Sub, Adds) with mask operations in Ascend C kernels
- Operations where data is accessed with regular strides in memory

## 失效边界

- When memory access patterns are non-contiguous or require complex indexing
- When data must be processed in small chunks due to memory constraints
- When implementing operations with irregular access patterns that cannot be vectorized efficiently

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- triu/op_kernel/triu.cpp

### 改动符号

- KernelTriu::Compute

### 代码签名

- contiguous_access
- ub

### 原始 diff

- 源 diff SHA-256：`26bdf68d3ec156d6f1b1a5fc022ca86942eeef1327b196969f59356523f849a6`

````diff
--- Before/op_kernel/triu.cpp
+++ After/op_kernel/triu.cpp
@@ -131,7 +131,17 @@
         uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
         Adds(yLocal,xLocal,scalarZero,mask,this->repeatTimes * cnt,{1,1,8,8});
         for(int32_t i=adjust;i<cnt;i++){
-            Sub(yLocal[localOffset],xLocal[localOffset],xLocal[localOffset],currLength);
+            // Degradation: Split single Sub operation into multiple smaller Sub operations
+            // to increase instruction count and function call overhead
+            uint32_t remaining = currLength;
+            uint32_t chunkOffset = 0;
+            while (remaining > 0) {
+                uint32_t chunkSize = (remaining > 32) ? 32 : remaining;
+                Sub(yLocal[localOffset + chunkOffset], xLocal[localOffset + chunkOffset],
+                    xLocal[localOffset + chunkOffset], chunkSize);
+                chunkOffset += chunkSize;
+                remaining -= chunkSize;
+            }
             currLength++;
             localOffset+=this->columnLength;
         }
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`18433.148438`
- fast duration us：`2361.387207`
- improvement ratio：`0.8718945265947085`
- machine confidence：`0.95`

### Profile evidence

- arithmetic.aiv_vec_misc_ratio_mean: 0.017609 -> 0.547936 (+3011.68%)
- critical_path.aiv_vector0_time_us_mean: 2360.81 -> 18432.2 (+680.76%)
- latency.task_duration_us: 2361.39 -> 18433.1 (+680.61%)
- arithmetic.aiv_vec_fops: 1.02752e+08 -> 3.37052e+08 (+228.02%)
- aiv_utilization.vec_ratio_mean: 0.300804 -> 0.752366 (+150.12%)

### Affected metric groups

- arithmetic
- critical_path
- latency
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "arithmetic.aiv_vec_misc_ratio_mean": {
      "before": 0.017609,
      "after": 0.547936,
      "delta": 0.530327,
      "rel": 30.116815264921346
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 2360.809,
      "after": 18432.182,
      "delta": 16071.373,
      "rel": 6.807570201570732
    }
  },
  {
    "latency.task_duration_us": {
      "before": 2361.387,
      "after": 18433.148,
      "delta": 16071.761,
      "rel": 6.8060682133000645
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 102752256.0,
      "after": 337051776.0,
      "delta": 234299520.0,
      "rel": 2.280237233915331
    }
  },
  {
    "aiv_utilization.vec_ratio_mean": {
      "before": 0.300804,
      "after": 0.752366,
      "delta": 0.45156199999999996,
      "rel": 1.5011834949003335
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

- In triu.cpp:317-327, maintain single vectorized Sub calls for entire row segments instead of splitting into 32-element chunks
- Use contiguous vector operations with appropriate mask patterns for matrix triangular operations
- Keep vector operation sizes aligned with hardware vector width (256-bit or 512-bit) for optimal performance
- Minimize function call overhead by processing maximum contiguous segments in single vector operations

## 检索线索

- triu
- vectorization
- operation_splitting
- Ascend C
- Sub
- Adds
- pipeline_stall
- vector_ratio
- aiv_vec_fops
- contiguous_access
