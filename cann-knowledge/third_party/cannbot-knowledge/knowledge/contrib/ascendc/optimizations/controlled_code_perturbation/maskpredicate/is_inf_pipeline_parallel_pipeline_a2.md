---
type: Optimization
title: "is_inf：流水与并行优化"
description: "is_inf 算子的流水与并行优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "topic.implementation.pipeline", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf_def.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_host/is_inf_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_kernel/is_inf.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/op_kernel/is_inf.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/is_inf/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# is_inf：流水与并行优化

## 来源信息

- experience ID：`3b5ad433e6d35003`
- 源 schema version：`v2`
- 算子：`is_inf`
- source iteration：`round_002_degrade`
- 源 category：`math`
- 计算分区：`maskpredicate`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`pipeline.parallelism.reduction`
- canonical family：`pipeline.parallelism.reduction`
- 源记录时间：`2026-04-20T19:54:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Reduce double buffering to single buffering
~~~~

### Original summary

~~~~text
For memory-bound Ascend C operators like is_inf, using single buffering (BUFFER_NUM=1) instead of double buffering can improve performance by 31% by reducing pipeline stalls and bank conflicts. This optimization works when data transfer and computation stages have minimal overlap benefit and the overhead of managing multiple buffers outweighs parallelism gains.
~~~~

### Original root mechanism

~~~~text
Reducing BUFFER_NUM from 2 to 1 eliminates pipeline parallelism overhead, reducing bank conflict stalls and improving overall performance for memory-bound operations
~~~~

### Original causal chain

~~~~text
Changed BUFFER_NUM from 2 to 1 -> Reduced pipeline stalls (aiv_vec_bank_cflt_ratio_mean decreased 85.6%) -> Improved vector unit utilization (vec_ratio_mean increased 39.7%) -> Reduced overall latency by 31%
~~~~

## 适用条件

- Operator has BUFFER_NUM=2 in Ascend C kernel code
- Profile shows high aiv_vec_bank_cflt_ratio_mean (>0.05) indicating bank conflict stalls
- MTE utilization (mte2_ratio_mean, mte3_ratio_mean) is high (>0.9) suggesting memory transfer overhead
- Operator is memory-bound with low arithmetic intensity
- Pipeline parallelism between data transfer and computation shows minimal benefit

## 失效边界

- Operator is compute-bound with high arithmetic utilization
- Data transfer and computation can be effectively overlapped
- Memory bandwidth is the primary bottleneck (not bank conflicts)
- Operator has complex dependency chains requiring double buffering

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_kernel/is_inf.cpp

### 改动符号

- IsInf::BUFFER_NUM

### 代码签名

- BUFFER_NUM
- compile_time_specialization
- constexpr
- ub

### 原始 diff

- 源 diff SHA-256：`55693678576b4780fe406a9a99a91ed106be3b6ec2f2a6d55ac5fe1fe1d8eafe`

````diff
--- Before/op_kernel/is_inf.cpp
+++ After/op_kernel/is_inf.cpp
@@ -12,7 +12,7 @@
         __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const IsInfTilingData* tilingData);
         __aicore__ inline void Process();

-        constexpr static uint8_t BUFFER_NUM = 2;
+        constexpr static uint8_t BUFFER_NUM = 1;
         constexpr static uint8_t DATA_BLOCK = 32;
         constexpr static uint8_t FLOAT_INTERVAL_TYPE = 2;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2636.352783`
- fast duration us：`1818.416382`
- improvement ratio：`0.31025301555781953`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_vec_bank_cflt_ratio_mean: 0.066078 -> 0.009517 (-85.60%)
- pipeline_stalls.aiv_vec_wait_ratio_mean: 0.376079 -> 0.561447 (+49.29%)
- critical_path.aiv_vector0_time_us_mean: 1783.44 -> 2617.54 (+46.77%)
- latency.task_duration_us: 1818.42 -> 2636.35 (+44.98%)
- aiv_utilization.mte2_ratio_mean: 0.921704 -> 0.521699 (-43.40%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- aiv_utilization

### Changed metrics top

~~~~json
[
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.066078,
      "after": 0.009517,
      "delta": -0.056561,
      "rel": -0.855973243742244
    }
  },
  {
    "pipeline_stalls.aiv_vec_wait_ratio_mean": {
      "before": 0.376079,
      "after": 0.561447,
      "delta": 0.18536800000000003,
      "rel": 0.49289643931195315
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1783.438,
      "after": 2617.539,
      "delta": 834.1010000000001,
      "rel": 0.46769273728607336
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1818.416,
      "after": 2636.353,
      "delta": 817.9370000000001,
      "rel": 0.4498074148049732
    }
  },
  {
    "aiv_utilization.mte2_ratio_mean": {
      "before": 0.921704,
      "after": 0.521699,
      "delta": -0.40000499999999994,
      "rel": -0.4339842292102453
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

- Locate BUFFER_NUM constant in Ascend C kernel class definition (e.g., in op_kernel/is_inf.cpp)
- Change BUFFER_NUM from 2 to 1 in the constexpr static declaration
- Verify kernel compilation still works with single buffer configuration
- Profile the operator to confirm reduction in aiv_vec_bank_cflt_ratio_mean
- Check that aiv_utilization.vec_ratio_mean increases and pipeline_stalls.aiv_vec_wait_ratio_mean decreases
- Validate overall latency improvement of 20-40% for similar memory-bound operators

## 检索线索

- BUFFER_NUM
- double_buffering
- single_buffering
- pipeline_stalls
- bank_conflict
- aiv_vec_bank_cflt_ratio_mean
- memory_bound
- Ascend_C
- compile_time_specialization
- constexpr
