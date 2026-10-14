---
type: Optimization
title: "add_custom：并行切核优化"
description: "add_custom 算子的并行切核优化经验，关注标量开销。"
tags: ["task.optimization", "topic.performance.optimization.optimization_overview", "paradigm.elementwise", "topic.tiling", "topic.data_movement.memory_access", "topic.programming_model"]
status: stable
sources:
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/docs/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/docs/AddCustom.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/.gitkeep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationATBPlugin/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationATBPlugin/aclnn_add_operation.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationATBPlugin/aclnn_add_operation.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationATBPlugin/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationTest/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationTest/script/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationTest/script/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationTest/script/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/AddOperationTest/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/ATBInvocation/readme.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclNNInvocationNaive/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclNNInvocationNaive/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclNNInvocationNaive/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclNNInvocationNaive/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclNNInvocationNaive/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclNNInvocationNaive/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/inc/common.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/inc/op_runner.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/inc/operator_desc.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/input/.keep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/output/.keep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/scripts/acl.json
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/scripts/add_custom_dynamic_shape.json
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/scripts/add_custom_static_shape.json
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/scripts/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/scripts/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/src/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/src/common.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/src/op_runner.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOfflineModel/src/operator_desc.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/inc/common.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/inc/op_runner.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/inc/operator_desc.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/input/.keep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/output/.keep
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/scripts/acl.json
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/scripts/gen_data.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/scripts/verify_result.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/src/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/src/common.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/src/main.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/src/op_runner.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/AclOnlineModel/src/operator_desc.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/jit/extension_add.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/jit/pytorch_npu_helper.hpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/jit/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/jit/test/test_add_custom.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/load_library/CMakeLists.txt
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/load_library/extension_add.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/load_library/pytorch_npu_helper.hpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/load_library/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/load_library/test/test_add_custom.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/csrc/extension_add.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/csrc/extension_add1.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/csrc/function.h
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/csrc/pytorch_npu_helper.hpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/csrc/register.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/custom_ops/__init__.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/custom_ops/add_custom.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/run_graph.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/setup.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/test/test_add_custom.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/CppExtensions/setup/test/test_add_custom_graph.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/PytorchInvocation/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/PytorchInvocation/op_plugin_patch/AddCustomKernelNpuApi.cpp
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/PytorchInvocation/op_plugin_patch/op_plugin_functions.yaml
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/PytorchInvocation/run_op_plugin.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/PytorchInvocation/test_ops_custom.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/PytorchInvocation/test_ops_custom_register_in_graph.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowBuildIn/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowBuildIn/run.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowBuildIn/run_add_custom.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowBuildIn/run_add_custom_tf2.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowCustom/README.md
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowCustom/RegisterTensorFlowCustomOp/custom_assign_add_custom.cc
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowCustom/build.sh
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowCustom/run_add_custom_tf_1_15.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/examples/TensorflowInvocation/AscendCustomToTensorFlowCustom/run_add_custom_tf_2_6_5.py
    role: example
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/framework/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/framework/tf_plugin/CMakeLists.txt
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/framework/tf_plugin/tensorflow_add_custom_plugin.cc
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/op_host/add_custom.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/op_host/add_custom_tiling.h
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/op_kernel/add_custom.cpp
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/opp_kernel_aicpu/.gitkeep
    role: implementation
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/tests/st/.gitkeep
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/tests/st/AddCustom_case_all_type.json
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/tests/st/README.md
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/tests/st/msopst.ini
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/tests/st/test_add_custom.py
    role: supporting
  - resource: https://gitee.com/ascend/cann-ops/blob/35fcd12e27bc446aef29bdf801d2869c15685f84/src/math/add_custom/tests/ut/.gitkeep
    role: supporting
created_at: "2026-08-31T07:22:50Z"
updated_at: "2026-09-21T10:19:04Z"
platforms: ["a2"]
---
# add_custom：并行切核优化

## 来源信息

- experience ID：`2406a1f3d12bbc76`
- 源 schema version：`v2`
- 算子：`add_custom`
- source iteration：`round_001_degrade`
- 源 category：`math`
- 计算分区：`elementwise_binary`
- experience type：`performance_recovery`
- explore mode：`degrade`
- optimization type：`parallelism.core.scaling`
- canonical family：`parallelism.core.limitation`
- 源记录时间：`2026-04-19T08:38:59`
- mechanism understood：`True`

## 优化主张

### Original optimization title

~~~~text
Increase core parallelism in tiling for memory-bound operations
~~~~

### Original summary

~~~~text
For memory-bound operations like vector addition, increasing core parallelism from 24 to 48 cores reduces memory bandwidth usage per core and improves overall performance. This optimization works by doubling the number of active cores in the tiling calculation, which distributes memory operations more evenly across available hardware resources. Apply this technique when memory bandwidth usage rates (gm_to_ub_bw_usage_rate_pct, ub_to_gm_bw_usage_rate_pct) are high and vector utilization (vec_ratio_mean) is low.
~~~~

### Original root mechanism

~~~~text
Increasing core parallelism reduces per-core memory bandwidth pressure and improves overall memory utilization by distributing memory operations across more cores.
~~~~

### Original causal chain

~~~~text
Increase block_dim from 24 to 48 cores → memory bandwidth usage per core decreases (gm_to_ub_bw_usage_rate_pct: 29.57% → 8.31%, ub_to_gm_bw_usage_rate_pct: 16.38% → 4.6%) → vector utilization improves (vec_ratio_mean: 0.087027 → 0.042848) → task duration reduces from 2738.755μs to 2585.612μs.
~~~~

## 适用条件

- Memory bandwidth usage rates (gm_to_ub_bw_usage_rate_pct, ub_to_gm_bw_usage_rate_pct) exceed 15%
- Vector utilization (vec_ratio_mean) is below 0.1
- Operation is memory-bound with high memory bandwidth consumption
- Available cores are underutilized (block_count < device maximum)
- Operation has regular data access patterns suitable for parallel distribution

## 失效边界

- Operation is compute-bound with high arithmetic intensity
- Memory bandwidth is already saturated at device level
- Data dependencies prevent effective parallelization
- Core-to-core communication overhead would dominate
- Operation size is too small to benefit from additional parallelism

## 变更方向

本记录来自受控劣化实验。下方 diff 从正常实现变为人为劣化实现，优化方向与 diff 相反。

- improvement direction：`degraded_is_slow`

## 代码差异

### 改动文件

- op_host/add_custom.cpp

### 改动符号

- TilingFunc

### 代码签名

- coreNum
- core_launch

### 原始 diff

- 源 diff SHA-256：`86a34ad13a69b83d2730e94431021b08a0366168a26a0d1cce0b863a355e08d6`

````diff
--- Before/op_host/add_custom.cpp
+++ After/op_host/add_custom.cpp
@@ -49,6 +49,11 @@
     {
         // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
         coreNum = (coreNum <  inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
+        // DEGRADATION: Artificially limit core usage to at most half of available cores
+        // This reduces parallelism and increases per-core workload
+        if (coreNum > 1) {
+            coreNum = (coreNum + 1) / 2; // ceil(coreNum/2)
+        }
     }

     uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
````

## 观测记录

- bottleneck type：`scalar_overhead_bound`
- slow duration us：`2738.754883`
- fast duration us：`2585.611816`
- improvement ratio：`0.05591704023992423`
- machine confidence：`0.9`

### Profile evidence

- memory_ub.ub_to_gm_bw_usage_rate_pct: 4.6 -> 16.38 (+256.09%)
- memory_ub.gm_to_ub_bw_usage_rate_pct: 8.31 -> 29.57 (+255.84%)
- memory_gm.aiv_main_mem_write_bw_gbps: 8.09 -> 16.48 (+103.71%)
- memory_gm.aiv_main_mem_read_bw_gbps: 16.19 -> 32.97 (+103.64%)
- memory_ub.ub_write_bw_vector_gbps: 8.09 -> 16.47 (+103.58%)

### Affected metric groups

- memory_ub
- memory_gm
- pipeline_stalls
- arithmetic

### Changed metrics top

~~~~json
[
  {
    "memory_ub.ub_to_gm_bw_usage_rate_pct": {
      "before": 4.6,
      "after": 16.38,
      "delta": 11.78,
      "rel": 2.5608695652173914
    }
  },
  {
    "memory_ub.gm_to_ub_bw_usage_rate_pct": {
      "before": 8.31,
      "after": 29.57,
      "delta": 21.259999999999998,
      "rel": 2.5583634175691934
    }
  },
  {
    "memory_gm.aiv_main_mem_write_bw_gbps": {
      "before": 8.09,
      "after": 16.48,
      "delta": 8.39,
      "rel": 1.0370828182941905
    }
  },
  {
    "memory_gm.aiv_main_mem_read_bw_gbps": {
      "before": 16.19,
      "after": 32.97,
      "delta": 16.779999999999998,
      "rel": 1.0364422483014204
    }
  },
  {
    "memory_ub.ub_write_bw_vector_gbps": {
      "before": 8.09,
      "after": 16.47,
      "delta": 8.379999999999999,
      "rel": 1.0358467243510505
    }
  }
]
~~~~

### Prediction deviation

~~~~json
{
  "direction_deviation": 0.0,
  "magnitude_deviation": 0.184579,
  "overall_deviation": 0.073832,
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

- 1. Examine the tiling function in op_host/add_custom.cpp to identify coreNum calculation
- 2. Increase core parallelism by adjusting the coreNum calculation to use more available cores
- 3. Ensure each core has sufficient work (at least 32B per core as per BLOCK_SIZE constraint)
- 4. Monitor memory bandwidth metrics (gm_to_ub_bw_usage_rate_pct, ub_to_gm_bw_usage_rate_pct) to verify reduced per-core pressure
- 5. Check vector utilization (vec_ratio_mean) improves with increased parallelism

## 检索线索

- core_parallelism
- memory_bandwidth
- tiling
- coreNum
- vector_utilization
- memory_bound
- bandwidth_pressure
- add_custom
- math_operator
