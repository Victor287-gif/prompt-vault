# matmul: 491–560 / 567

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **x_all_gather_matmul_tiling** · 45 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_tiling.h`
- **x_all_gather_matmul_tiling_key** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_tiling_key.h`
- **参考工程（all_gather_matmul/）改造食谱** · 281 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/codebase_map.md`
- **通信层：HCCL in-kernel 高阶 API（A2/910B）** · 201 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/comm_hccl.md`
- **计算层：AscendC::Matmul 与 HCCL 流水耦合（910B / A2）** · 50 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/matmul_fusion.md`
- **910B（A2）MC2 架构心智模型** · 142 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/mc2_architecture.md`
- **算子架构与可复用框架（910B/A2 / aclnn 单算子 API）** · 167 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/op_architecture.md`
- **hccl-matmul 通算并行调优（tileCnt，910B / A2）** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/pipeline_tuning.md`
- **hccl-matmul 路线代码审查验收条件** · 39 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/review-checklist.md`
- **CANNBot 工作流在 910B（A2）MC2 场景的具体化** · 152 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/workflow_integration.md`
- **Local Matmul 策略** · 77 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/comm-compute/local_matmul.md`
- **MatMul 族 — 兜底算法（官方参考）** · 45 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/fallback-algorithm.md`
- **matmul 族 TilingData 字段语义对照表（跨变体）** · 200 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-fields.md`
- **MatMul 族 — 通用 Tiling 流程** · 376 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-flow.md`
- **MatMul 族 — 变体差异说明** · 145 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-variants.md`
- **MatMul API 常量折叠优化设计** · 466 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/constant_folding_design.md`
- **matmul 减少重复载入（A-Full-Load / B-Full-Load）优化设计** · 344 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/fullload_design.md`
- **matmul MTE2 预加载（MTE2 Preload）优化设计** · 320 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/mte2_preload_design.md`
- **matmul Pingpong 双缓冲优化设计** · 254 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/pingpong_design.md`
- **matmul 小数据块合并载入（Scale / 小矩阵 Access Coalescing）优化设计** · 282 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/scale_coalescing_design.md`
- **小 M、宽 N Matmul：从载入量出发选择分块与流水** · 137 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/small_m_wide_n_pipeline_design.md`
- **matmul StreamK 调度模式技术文档** · 374 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/streamk_design.md`
- **SWAT 调度模式技术文档** · 423 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/swat_design.md`
- **MC² Local Matmul 优化设计** · 130 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/mc2/local_matmul_design.md`
- **----------------------------------------------------------------------------------------------------------** · 40 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/CMakeLists.txt`
- **MatmulBlazeExample** · 90 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/README.md`
- **Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 28 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/build.sh`
- **examples** · 13 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/examples/README.md`
- **op_api** · 13 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/README.md`
- **aclnn_matmul_blaze_example** · 49 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/aclnn_matmul_blaze_example.cpp`
- **aclnn_matmul_blaze_example** · 29 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/aclnn_matmul_blaze_example.h`
- **matmul_blaze_example** · 44 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/matmul_blaze_example.cpp`
- **matmul_blaze_example** · 17 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/matmul_blaze_example.h`
- **----------------------------------------------------------------------------------------------------------** · 33 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_graph/CMakeLists.txt`
- **matmul_blaze_example_proto** · 23 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_graph/matmul_blaze_example_proto.h`
- **----------------------------------------------------------------------------------------------------------** · 95 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/CMakeLists.txt`
- **matmul_blaze_example_tiling** · 94 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/arch35/matmul_blaze_example_tiling.cpp`
- **matmul_blaze_example_tiling_data** · 42 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/arch35/matmul_blaze_example_tiling_data.h`
- **matmul_blaze_example_tiling_key** · 14 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/arch35/matmul_blaze_example_tiling_key.h`
- **matmul_blaze_example_def** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/matmul_blaze_example_def.cpp`
- **matmul_blaze_example_infershape** · 30 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/matmul_blaze_example_infershape.cpp`
- **----------------------------------------------------------------------------------------------------------** · 35 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/CMakeLists.txt`
- **custom_compile_options** · 1 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/custom_compile_options.ini`
- **blaze/** · 9 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/include/blaze/README.md`
- **tensor_api/** · 9 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/include/tensor_api/README.md`
- **matmul_blaze_example_arch35** · 112 行 · upstream
  `cannbot-skills/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/matmul_blaze_example_arch35.cpp`
- **bm_good_b_double_buffer_pingpong** · 49 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_b_double_buffer_pingpong.asc`
- **bm_good_sync01_loop_pingpong** · 52 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_sync01_loop_pingpong.asc`
- **Ascend C GMM 高阶 API Tiling 策略** · 313 行 · upstream
  `cannbot-skills/ops/ascendc-tiling-design/references/matmul/ascendc-api-gmm-tiling.md`
- **Ascend C Matmul 高阶 API Tiling 策略** · 408 行 · upstream
  `cannbot-skills/ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md`
- **MatMul 类算子场景路由** · 32 行 · upstream
  `cannbot-skills/ops/ascendc-tiling-design/references/matmul/patterns.md`
- **Reduce 类算子 - AR 全载分支** · 151 行 · upstream
  `cannbot-skills/ops/ascendc-tiling-design/references/reduction/ar-fullload.md`
- **Reduce 类算子 - ARA 全载分支** · 228 行 · upstream
  `cannbot-skills/ops/ascendc-tiling-design/references/reduction/ara-fullload.md`
- **Kernel 路由：Grouped Matmul 类算子** · 220 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/grouped-matmul.md`
- **Kernel 路由：Matmul 类算子** · 179 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/matmul.md`
- **Kernel 路由：Quant Matmul 类算子** · 40 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/quant-matmul.md`
- **Catlass 矩阵乘法模板清单** · 83 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/matmul-templates.md`
- **选到最优的 BlockMmad 与 BlockEpilogue（影响精度 + 性能）** · 124 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/mmad-epilogue-selection.md`
- **纯 Matmul — 完整代码骨架** · 76 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/basic-matmul.md`
- **Grouped Matmul — 分组矩阵乘（含 MoE 融合）实现注意事项** · 188 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/grouped-matmul.md`
- **Quant Matmul — 量化矩阵乘** · 91 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/quant-matmul.md`
- **Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 331 行 · upstream
  `cannbot-skills/ops/ops-spec-gen/examples/fused_quant_matmul/spec.yaml`
- **Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 201 行 · upstream
  `cannbot-skills/ops/ops-spec-gen/examples/matmul/spec.yaml`
- **matmul API and Tile Shapes (DEBUG_GUIDEBOOK.md §9.19)** · 352 行 · upstream
  `cannbot-skills/ops/pypto-general-debug/references/matmul.md`
- **MATMUL 组件经验** · 105 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_classified/matmul.md`
- **案例：多 Matmul 独立 TileShape 优化** · 77 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/cases/per-matmul-tile-shapes.md`
- **matmul-l2-layout** · 53 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/matmul-l2-layout.md`
- **GEMM 与 CV 融合编码规范** · 43 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/gemm-cv-fusion.md`
- **GEMM 算子性能优化最佳实践** · 513 行 · upstream
  `cannbot-skills/ops/tilelang-perf-optimization/Ascend910/references/best-practices/gemm_intrinsic_optimize.md`
- **GEMM 一侧驻留** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/fullload_design.md`

下一页：page-009.md
