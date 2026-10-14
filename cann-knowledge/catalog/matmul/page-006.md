# matmul: 351–420 / 567

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **weight_quant_matmul_kernel** · 611 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/kernel/weight_quant_matmul_kernel.h`
- **blaze_group_matmul_pertoken_quant_selector** · 119 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_group_matmul_pertoken_quant_selector.h`
- **blaze_group_matmul_tiling_data** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_group_matmul_tiling_data.h`
- **blaze_matmul_tiling** · 457 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_matmul_tiling.h`
- **blaze_matmul_tiling_data** · 50 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_matmul_tiling_data.h`
- **quant_matmul_mx_tiling** · 549 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/mx/quant_matmul_mx_tiling.h`
- **quant_matmul_mx_tiling_data** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/mx/quant_matmul_mx_tiling_data.h`
- **Matmul Layout 与数据格式开发指南** · 322 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-matmul-layout.md`
- **Grouped MatMul 项目侧 Host/Tiling/ABI 与执行差异** · 299 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/group-matmul-delta.md`
- **Cube MatMul -> Vector Elementwise -> Vector Per-token Quant** · 196 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-design.md`
- **Cube MatMul -> Vector Elementwise -> Vector Per-token Quant 开发指导** · 104 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-development.md`
- **Cube MatMul -> Vector GLU -> Vector Per-token Quant** · 189 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-design.md`
- **Cube MatMul -> Vector GLU -> Vector Per-token Quant 开发指导** · 52 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-development.md`
- **Cube MatMul -> Vector GLU** · 150 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-design.md`
- **Cube MatMul -> Vector GLU 开发指导** · 54 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-development.md`
- **Cube MatMul -> Vector Per-token Quant** · 138 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-design.md`
- **Cube MatMul -> Vector Per-token Quant 开发指导** · 45 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-development.md`
- **Cube MatMul → Vector Softmax/Reduce 场景设计指导** · 205 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-design.md`
- **Cube MatMul → Vector Softmax/Reduce 开发指导** · 99 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-development.md`
- **CV Sync 与两阶段 Kernel 编排专题** · 156 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv-sync-and-two-phase-entry.md`
- **从 group_matmul_kernel_cv1_v2.h 适配为 Softmax/Reduce 两阶段 Kernel** · 331 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv1-v2-adaptation-for-softmax-reduce.md`
- **Online Softmax/Reduce Epilogue 设计专题** · 174 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/online-softmax-reduce-epilogue-design.md`
- **Pure Reduce 适配指导** · 204 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/reduce-adaptation-guide.md`
- **Blaze Matmul 工程模板指南** · 348 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_guide.md`
- **----------------------------------------------------------------------------------------------------------** · 157 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/CMakeLists.txt`
- **acl_utils** · 70 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/acl_utils.h`
- **common_utils** · 84 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/common_utils.h`
- **io_utils** · 163 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/common/io_utils.h`
- **matmul_blaze_torch** · 55 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_extension/matmul_blaze_torch.cpp`
- **ops** · 22 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_extension/ops.h`
- **register** · 22 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_extension/register.cpp`
- **matmul_blaze_kernel** · 33 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_kernel/matmul_blaze_kernel.cpp`
- **matmul_blaze_kernel** · 99 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_kernel/matmul_blaze_kernel.h`
- **matmul_constant** · 31 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_constant.h`
- **matmul_tiling_data** · 45 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_tiling_data.h`
- **matmul_tiling_stub** · 77 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/op_tiling/matmul_tiling_stub.h`
- **----------------------------------------------------------------------------------------------------------** · 94 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/run.sh`
- **coding=utf-8** · 152 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/scripts/gen_data_mxfp4.py`
- **coding=utf-8** · 114 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/scripts/gen_data_mxfp8.py`
- **----------------------------------------------------------------------------------------------------------** · 132 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/scripts/verify_result.py`
- **test_matmul_blaze** · 145 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze.cpp`
- **coding=utf-8** · 141 行 · upstream
  `cannbot-skills/ops/ascendc-direct-invoke-template/references/matmul_blaze_template/test_matmul_blaze_torch.py`
- **get-all-gather-quant-matmul 场景设计** · 93 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/get-all-gather-quant-matmul/design.md`
- **get-all-gather-quant-matmul 开发指导** · 70 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/get-all-gather-quant-matmul/development.md`
- **场景：put-all-gather-quant-matmul（AllGather + QuantMatmul 变体扩展）** · 83 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/design.md`
- **场景开发指导：put-all-gather-quant-matmul** · 62 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/development.md`
- **put-all-to-all-quant-matmul 场景设计指导** · 106 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/design.md`
- **put-all-to-all-quant-matmul 开发指导** · 74 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/development.md`
- **============================================================================** · 95 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/CMakeLists.txt`
- **find ascend toolkit** · 59 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/cmake/ascend.cmake`
- **shmem 来源：gitcode.com/cann/shmem v1.5.0，由 cmake 首次配置时自动 clone。** · 75 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/cmake/shmem.cmake`
- **tensor_api 来源：gitcode.com/cann/asc-devkit（feature/tensor_api_from_9.0.0 分支，commit ad3d3bf），** · 48 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/cmake/tensor_api.cmake`
- **acl_utils** · 73 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/acl_utils.h`
- **common_utils** · 181 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/common_utils.h`
- **io_utils** · 175 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/io_utils.h`
- **common_utils** · 99 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/common_utils.h`
- **integral_constant** · 36 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/integral_constant.h`
- **tuple_utils** · 34 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/kernel_utils/tuple_utils.h`
- **block_mmad** · 27 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_mmad.h`
- **block_scheduler_policy** · 35 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_scheduler_policy.h`
- **block_scheduler_utils** · 27 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/block_scheduler_utils.h`
- **quant_matmul_mx_block_mmad_swat** · 392 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_mmad_swat.h`
- **quant_matmul_mx_block_scheduler_swat** · 290 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/block/quant_matmul_mx_block_scheduler_swat.h`
- **all_to_all_comm_udma** · 255 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_comm_udma.h`
- **all_to_all_matmul_impl** · 370 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h`
- **heavy_kernels** · 128 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/heavy_kernels.h`
- **qbmm_mx_kernel** · 395 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/qbmm_mx_kernel.h`
- **dispatch_policy** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/policy/dispatch_policy.h`
- **copy_scale_l1_to_l0a** · 61 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/copy_scale_l1_to_l0a.h`
- **copy_scale_l1_to_l0b** · 54 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tile/copy_scale_l1_to_l0b.h`

下一页：page-007.md
