---
type: Optimization
title: "addcdiv：并行切核优化"
description: "addcdiv 算子的并行切核优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/docs/Addcdiv.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/aclnn_addcdiv_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/aclnn_addcdiv_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/AddCdivOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/framework/tf_plugin/tensorflow_addcdiv_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/op_host/addcdiv.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/op_host/addcdiv_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/op_kernel/addcdiv.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/Addcdiv_case_alltype.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/st/test_addcdiv.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/addcdiv/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# addcdiv：并行切核优化

## 来源信息

- experience ID：`8bdae0bbc222c349`
- 源 schema version：`v2`
- 算子：`addcdiv`
- source iteration：`round_000_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.core.limitation`
- canonical family：`parallelism.core.limitation`
- 源记录时间：`2026-04-19T08:44:21`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Force single-core execution
~~~~

### Original summary

~~~~text
On `math/addcdiv`, applying `Force single-core execution` yields 88.0% performance improvement (1724.7 vs 14408.4 us, 88.0% difference) by addressing `scalar_overhead_bound`. Artificially limiting parallelism by forcing single-core execution regardless of data size or available cores.
~~~~

### Original root mechanism

~~~~text
Artificially limiting parallelism by forcing single-core execution regardless of data size or available cores.
~~~~

### Original causal chain

~~~~text
apply parallelism.core.limitation in addcdiv/op_host/addcdiv.cpp -> improve memory_ub.ub_to_gm_kb -> 88.0% latency improvement
~~~~

## 适用条件

- The target operator's profiling shows bottleneck in: memory_ub, arithmetic, critical_path
- The kernel code uses related APIs/patterns: coreNum, core_launch
- The operator has a `scalar_overhead_bound` bottleneck profile

## 失效边界

- The operator's bottleneck is unrelated to `scalar_overhead_bound`
- The `parallelism.core.limitation` pattern is not present in the target code
- The same optimization is already applied in the current implementation

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- addcdiv/op_host/addcdiv.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`fe7710ba1089dd7a9f133ac1bbda05a3282ba17485e918766c0c3ddd20a4ab39`

````diff
--- Before/op_host/addcdiv.cpp
+++ After/op_host/addcdiv.cpp
@@ -41,6 +41,9 @@
         // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
         coreNum = (coreNum <  inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
     }
+    // DEGRADATION: Force single-core execution even when parallelization is possible
+    // This reduces parallelism and increases latency by serializing all computation
+    coreNum = 1;
     uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
     uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;


````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`14408.427734`
- fast duration us：`1724.734497`
- improvement ratio：`0.8802968284367286`
- machine confidence：`0.82`

### Profile evidence

- memory_ub.ub_to_gm_kb: 10965.5 -> 526339 (+4699.95%)
- arithmetic.aiv_vec_fops: 2.80571e+07 -> 1.34673e+09 (+4699.95%)
- memory_ub.gm_to_ub_kb: 32896.6 -> 1.57902e+06 (+4699.94%)
- critical_path.aiv_vector0_time_us_mean: 1718.89 -> 14407.5 (+738.19%)
- latency.task_duration_us: 1724.73 -> 14408.4 (+735.40%)

### Affected metric groups

- memory_ub
- arithmetic
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_kb": {
      "before": 10965.5,
      "after": 526338.6,
      "delta": 515373.1,
      "rel": 46.99950754639551
    }
  },
  {
    "arithmetic.aiv_vec_fops": {
      "before": 28057120.0,
      "after": 1346727392.0,
      "delta": 1318670272.0,
      "rel": 46.99948790182314
    }
  },
  {
    "memory_ub.gm_to_ub_kb": {
      "before": 32896.6,
      "after": 1579016.0,
      "delta": 1546119.4,
      "rel": 46.999367715812575
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1718.886,
      "after": 14407.539,
      "delta": 12688.653,
      "rel": 7.381904908178902
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1724.734,
      "after": 14408.428,
      "delta": 12683.694,
      "rel": 7.354000095087127
    }
  },
  {
    "pipeline_stalls.aiv_vec_bank_cflt_ratio_mean": {
      "before": 0.005444,
      "after": 0.031207,
      "delta": 0.025762999999999998,
      "rel": 4.732365907421014
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 6.08,
      "after": 34.84,
      "delta": 28.760000000000005,
      "rel": 4.730263157894738
    }
  },
  {
    "pipeline_stalls.aiv_vec_total_cflt_ratio_mean": {
      "before": 0.019583,
      "after": 0.112179,
      "delta": 0.092596,
      "rel": 4.728386866159424
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 18.25,
      "after": 104.52,
      "delta": 86.27,
      "rel": 4.727123287671232
    }
  },
  {
    "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean": {
      "before": 0.014139,
      "after": 0.080972,
      "delta": 0.066833,
      "rel": 4.726854798783506
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

- Locate the `parallelism.core.limitation` code region in `addcdiv/op_host/addcdiv.cpp` (key symbols: TilingFunc)
- Apply the optimization pattern shown in the code_diff
- After applying, verify that `memory_ub.ub_to_gm_kb` moves in the expected direction

## 检索线索

- After
- BLOCK_SIZE
- Before
- DEGRADATION
- Force
- There
- This
- arithmetic
- coreNum
- core_launch
- critical_path
- everyCoreInputBlockNum
- inputLengthAlgin32
- latency
- memory_ub
- parallelism.core.limitation
- scalar_overhead_bound
- tailBlockNum
