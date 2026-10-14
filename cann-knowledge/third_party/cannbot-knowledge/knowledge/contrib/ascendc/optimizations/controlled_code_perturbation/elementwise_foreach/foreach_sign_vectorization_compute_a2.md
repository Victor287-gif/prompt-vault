---
type: Optimization
title: "foreach_sign：向量化计算优化"
description: "foreach_sign 算子的向量化计算优化经验，关注流水线停顿。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.programming_model.execution_unit.vector"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/docs/aclnnForeachSign.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/op_host/foreach_sign.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/op_kernel/foreach_sign.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/op_kernel/foreach_sign.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/foreach/foreach_sign/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# foreach_sign：向量化计算优化

## 来源信息

- experience ID：`e093cb142bda7c8c`
- 源 schema version：`v2`
- 算子：`foreach_sign`
- source iteration：`round_015_degrade`
- 源 category：`foreach`
- 计算分区：`elementwise_foreach`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`host.vector.initialization.inefficiency`
- canonical family：`host.vector.initialization.inefficiency`
- 源记录时间：`2026-04-20T00:54:42`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Use brace-initialized vector construction instead of loop-based push_back for host-side macro parameter setup
~~~~

### Original summary

~~~~text
In Ascend C host-side macro definitions like FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE, using brace-initialized vector construction instead of loop-based push_back initialization reduces overhead by eliminating repeated function calls. This optimization applies when host-side macros prepare parameter lists for kernel invocation and should be used when the number of elements is known at compile time.
~~~~

### Original root mechanism

~~~~text
Brace-initialized vector construction reduces O(n) overhead by avoiding repeated push_back calls in host-side macro definitions, improving parameter setup efficiency.
~~~~

### Original causal chain

~~~~text
Replacing loop-based push_back with brace-initialized vector construction in FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE macro reduces host-side parameter setup time → decreases aiv_vector0_time_us_mean from 7209.57μs to 5547.43μs → reduces task_duration_us from 7232.47μs to 5571.05μs → improves overall performance by 13.8%.
~~~~

## 适用条件

- Host-side macros prepare parameter lists for Ascend C kernel invocation
- Parameter lists have known size at compile time
- Using std::vector<ge::DataType> or std::vector<ge::Format> for parameter storage
- Macros defined in op_host/*.h files for operator parameter setup

## 失效边界

- Parameter list size is determined at runtime
- Elements need conditional inclusion or filtering
- Using custom allocators or non-standard vector initialization patterns

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- foreach_sign/op_host/foreach_sign_tiling.h

### 改动符号

- FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE
- FOREACH_SCALAR_DTYPE_PREPARE
- FOREACH_SCALAR_TENSOR_DTYPE_PREPARE

### 代码签名

- contiguous_access

### 原始 diff

- 源 diff SHA-256：`06703d39d97b9e7fc7562366709cb9706ee26911f7e1c0f5967ebce271f94180`

````diff
--- Before/op_host/foreach_sign_tiling.h
+++ After/op_host/foreach_sign_tiling.h
@@ -49,18 +49,27 @@
     };

 #define FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(...) \
-    std::vector<ge::DataType> tensor_dtype_list = {__VA_ARGS__}; \
-    std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
+    std::vector<ge::DataType> tensor_dtype_list; \
+    ge::DataType __dtypes[] = {__VA_ARGS__}; \
+    for (size_t __i = 0; __i < sizeof(__dtypes) / sizeof(__dtypes[0]); ++__i) { \
+        tensor_dtype_list.push_back(__dtypes[__i]); \
+    } \
+    std::vector<ge::Format> format_list; \
+    for (size_t __i = 0; __i < tensor_dtype_list.size(); ++__i) { \
+        format_list.push_back(ge::FORMAT_ND); \
+    }

 #define FOREACH_SCALAR_DTYPE_PREPARE \
     std::vector<ge::DataType> scalar_dtype_list; \
-    std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), \
-        [&scalar_dtype_list](ge::DataType dtype) { scalar_dtype_list.push_back(DtypeTensor2Scalar(dtype)); });
+    for (size_t __i = 0; __i < tensor_dtype_list.size(); ++__i) { \
+        scalar_dtype_list.push_back(DtypeTensor2Scalar(tensor_dtype_list[__i])); \
+    }

 #define FOREACH_SCALAR_TENSOR_DTYPE_PREPARE \
     std::vector<ge::DataType> scalar_tensor_dtype_list; \
-    std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), \
-        [&scalar_tensor_dtype_list](ge::DataType dtype) { scalar_tensor_dtype_list.push_back(DtypeScalarToTensor2(dtype)); });
+    for (size_t __i = 0; __i < tensor_dtype_list.size(); ++__i) { \
+        scalar_tensor_dtype_list.push_back(DtypeScalarToTensor2(tensor_dtype_list[__i])); \
+    }

 #define FOREACH_OPDEF_PARAM_TENSOR(PARAM_TYPE, NAME) \
     this->PARAM_TYPE(#NAME) \
````

## 观测记录

- bottleneck type：`pipeline_stall_bound`
- slow duration us：`14851.197266`
- fast duration us：`12799.696289`
- improvement ratio：`0.1381370767794365`
- machine confidence：`0.9`

### Profile evidence

- pipeline_stalls.aiv_mte2_wait_ratio_mean: 0.000185 -> 0.000101 (-45.41%)
- critical_path.aiv_vector0_time_us_mean: 5547.43 -> 7209.57 (+29.96%)
- latency.task_duration_us: 5571.05 -> 7232.47 (+29.82%)
- memory_gm.aiv_main_mem_read_bw_gbps: 7.62 -> 5.86 (-23.10%)
- memory_gm.aiv_main_mem_write_bw_gbps: 7.54 -> 5.8 (-23.08%)

### Affected metric groups

- pipeline_stalls
- critical_path
- latency
- memory_gm

### Changed metrics top

~~~~json
[
  "pipeline_stalls.aiv_mte2_wait_ratio_mean",
  "critical_path.aiv_vector0_time_us_mean",
  "latency.task_duration_us",
  "memory_gm.aiv_main_mem_read_bw_gbps",
  "memory_gm.aiv_main_mem_write_bw_gbps",
  "arithmetic.aiv_vec_misc_ratio_mean",
  "aiv_utilization.vec_ratio_mean",
  "arithmetic.aiv_vec_fp32_ratio_mean",
  "memory_ub.ub_read_bw_vector_gbps",
  "pipeline_stalls.aiv_vec_bankgroup_cflt_ratio_mean"
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

- In FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE macro, replace loop-based push_back with brace initialization: std::vector<ge::DataType> tensor_dtype_list = {__VA_ARGS__};
- For format lists with default values, use constructor with size and value: std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
- In FOREACH_SCALAR_DTYPE_PREPARE and FOREACH_SCALAR_TENSOR_DTYPE_PREPARE macros, use std::for_each with lambda instead of explicit loops for clarity
- Apply to all host-side parameter preparation macros in op_host/*.h files

## 检索线索

- vector initialization
- brace initialization
- push_back
- host-side macro
- parameter setup
- FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE
- FOREACH_SCALAR_DTYPE_PREPARE
- FOREACH_SCALAR_TENSOR_DTYPE_PREPARE
- op_host
- ge::DataType
- ge::Format
