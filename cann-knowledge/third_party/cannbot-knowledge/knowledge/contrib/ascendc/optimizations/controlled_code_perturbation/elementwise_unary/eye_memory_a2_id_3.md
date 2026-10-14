---
type: Optimization
title: "eye：内存优化"
description: "eye 算子的内存优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.data_movement.memory_access"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/docs/Eye.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/aclnn_eye_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/aclnn_eye_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/EyeOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_host/eye.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_host/eye_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/op_kernel/eye.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/Eye_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/st/test_eye.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/eye/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# eye：内存优化

## 来源信息

- experience ID：`901539d28e2b79b3`
- 源 schema version：`v2`
- 算子：`eye`
- source iteration：`round_005_degrade`
- 源 category：`math`
- 计算分区：`elementwise_unary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`kernel.memory.access.pattern`
- canonical family：`kernel.memory.access.pattern`
- 源记录时间：`2026-04-20T14:40:24`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Row-major memory access for eye matrix initialization
~~~~

### Original summary

~~~~text
For eye matrix initialization operations in Ascend C, using row-major memory access patterns instead of column-major access significantly improves performance. This optimization reduces memory bandwidth from 4.53 GB/s to 0.5 GB/s and improves L2 cache hit rates from 87.5% to 99.22% by accessing memory contiguously. Apply this pattern when initializing matrices with structured patterns like identity matrices.
~~~~

### Original root mechanism

~~~~text
Applying row-major memory access pattern for matrix initialization improves cache locality by accessing contiguous memory addresses, reducing cache misses and memory bandwidth consumption.
~~~~

### Original causal chain

~~~~text
Row-major memory access -> Improved cache locality (L2 read hit rate: 87.5% -> 99.22%) -> Reduced memory bandwidth (4.53 GB/s -> 0.5 GB/s) -> Lower scalar overhead (scalar ratio: 1.0 -> 0.124802) -> 43.2% latency improvement (2.76ms -> 1.57ms)
~~~~

## 适用条件

- Initializing matrices with structured patterns like identity matrices
- Working with row-major memory layouts in Ascend C kernels
- Operations involve writing to large 2D arrays where memory access pattern affects cache performance
- Using GlobalTensor::SetValue() for scattered writes to global memory

## 失效边界

- Memory access pattern is already optimal for the data layout
- Working with column-major data layouts where column-major access would be optimal
- Small matrices where cache effects are negligible
- Operations already vectorized with contiguous memory access

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- eye/op_kernel/eye.cpp

### 改动符号

- KernelEye::Process

### 代码签名

- SetValue
- contiguous_access
- scalar_value_access

### 原始 diff

- 源 diff SHA-256：`541da3d2607bf238a38900a05be6592291d60add604609aa66d6cd1d2a279416`

````diff
--- Before/op_kernel/eye.cpp
+++ After/op_kernel/eye.cpp
@@ -26,9 +26,10 @@

         yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y_ref + this->blockLength * GetBlockIdx(), this->blockLength);
     }
-    __aicore__ inline void Process() {
+        __aicore__ inline void Process() {
         int32_t index, t;
         if(mark == 0){
+            // Write 1s on diagonal first
             for(int32_t i = 0; i < num_rows; i++){
                 index = i * num_columns + i;
                 if(i < num_columns){
@@ -40,9 +41,11 @@
                     (void)unused;
                     yGm.SetValue(index, 1);
                 }
-                // Degradation: Write 0 to non-diagonal elements in a scattered pattern
-                // This creates inefficient memory access by writing to every element
-                for(int32_t k = 0; k < num_columns; k++) {
+            }
+            // Degradation: Write 0 to non-diagonal elements in COLUMN-MAJOR order
+            // This creates terrible cache locality for row-major memory layout
+            for(int32_t k = 0; k < num_columns; k++) {
+                for(int32_t i = 0; i < num_rows; i++) {
                     if(k != i) {
                         // Degradation: Add unnecessary global memory read before write
                         uint32_t write_idx = i * num_columns + k;
@@ -55,6 +58,7 @@
             }
         }else{
             for(int32_t i = 0; i < batchNum; i++){
+                // Write 1s on diagonal first for this batch
                 for(int32_t j = 0; j < num_rows; j++){
                     if(j < num_columns){
                         t = j * num_columns + j;
@@ -65,9 +69,11 @@
                         (void)unused;
                         yGm.SetValue(index, 1);
                     }
-                    // Degradation: Write 0 to non-diagonal elements in a scattered pattern
-                    // This creates inefficient memory access ... (line truncated to 2000 chars)
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2761094.5`
- fast duration us：`1568295.25`
- improvement ratio：`0.43200232733794514`
- machine confidence：`0.9`

### Profile evidence

- memory_gm.aiv_main_mem_read_bw_gbps: 0.5 -> 4.53 (+806.00%)
- memory_gm.aiv_main_mem_write_bw_gbps: 0.5 -> 4.53 (+806.00%)
- aiv_utilization.scalar_ratio_mean: 1 -> 0.124802 (-87.52%)
- critical_path.aiv_vector0_time_us_mean: 1.56826e+06 -> 2.76104e+06 (+76.06%)
- latency.task_duration_us: 1.5683e+06 -> 2.76109e+06 (+76.06%)

### Affected metric groups

- memory_gm
- aiv_utilization
- critical_path
- latency

### Changed metrics top

~~~~json
[
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 0.5,
      "after": 4.53,
      "delta": 4.03,
      "rel": 8.06
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 0.5,
      "after": 4.53,
      "delta": 4.03,
      "rel": 8.06
    }
  },
  {
    "aiv_utilization.scalar_ratio_mean": {
      "before": 1.0,
      "after": 0.124802,
      "delta": -0.875198,
      "rel": -0.875198
    }
  },
  {
    "critical_path.aiv_vector0_time_us_mean": {
      "before": 1568263.375,
      "after": 2761038.75,
      "delta": 1192775.375,
      "rel": 0.7605708288634873
    }
  },
  {
    "latency.task_duration_us": {
      "before": 1568295.25,
      "after": 2761094.5,
      "delta": 1192799.25,
      "rel": 0.7605705940893464
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

- In KernelEye::Process() function in eye/op_kernel/eye.cpp, ensure nested loops follow row-major order: outer loop over rows, inner loop over columns
- When initializing identity matrices, write diagonal elements first using contiguous memory access pattern
- Avoid unnecessary global memory reads before writes using yGm.GetValue() when not needed
- For batch operations, maintain consistent row-major access patterns across batches

## 检索线索

- memory_access_pattern
- row_major
- cache_locality
- eye_matrix
- identity_matrix
- ascend_c
- global_memory
- SetValue
- memory_bandwidth
- scalar_overhead
