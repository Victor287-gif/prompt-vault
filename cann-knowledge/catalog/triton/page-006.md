# triton: 351–420 / 465

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Element-wise 算子优化** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_elementwise.md`
- **Triton-Ascend Interpolate 算子开发参考文档** · 235 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_interpolate.md`
- **MatMul 算子优化** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_matmul.md`
- **入参静态化 优化模式** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/constexpr_parameters.md`
- **连续拷贝聚合优化** · 220 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/continuous_copy_aggregation.md`
- **Cube/MTE3 分阶段批量解耦优化** · 463 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/cube_mte3_decoupling.md`
- **Ascend CV 融合算子生成方法论 — CV 融合篇** · 349 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/cv_fusion.md`
- **Ascend CV 融合算子生成方法论 — Batch 流水线篇** · 458 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/cv_fusion_pingpong.md`
- **Ascend CV 融合算子生成方法论 — Tiling 篇** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/cv_fusion_tiling.md`
- **调试检查清单** · 93 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/debug_checklist.md`
- **Device-side Gather 连续化** · 336 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/device_side_gather.md`
- **维度合并优化** · 296 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/dimension_merge.md`
- **维度合并与大 BLOCK 累加（归一化算子专用）** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/dimension_merge_large_block.md`
- **离散访存优化模式** · 270 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/discrete_memory_access.md`
- **FA 形态 Phase 4 优化点清单** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/fa_phase4_optimization_checklist.md`
- **Attention / FlashAttention 类算子优化点库（Triton-Ascend）** · 700 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/flash_attention_optimization.md`
- **Triton-Ascend 通用优化洞察** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/general_insights.md`
- **Grid 形状与多路径特化** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/grid_dispatch_specialization.md`
- **grid 超 65535 的两种分派改写** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/grid_overflow_dispatch.md`
- **Histogram-like / Small-Output-Table Reduction 优化** · 145 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/histogram_like_table_reduction.md`
- **Host 侧张量维度拼接优化（Host-Side Tensor Concat）** · 320 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/host_tensor_concat.md`
- **Kernel 分裂优化** · 779 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/kernel_splitting.md`
- **Latency-Bound 循环维度 Tile 合并优化** · 314 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/latency_bound_tile_merge.md`
- **Libdevice 函数使用** · 298 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/libdevice_usage.md`
- **Load 指令重排序** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/load_order.md`
- **循环不变量外提优化** · 233 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/loop_invariant_hoisting.md`
- **Ascend Matmul Transpose 优化** · 340 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/matmul_transpose.md`
- **内存访问优化实践** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/memory_access_practice.md`
- **混合策略自动选择** · 233 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/mixed_strategy.md`
- **Multibuffer / 双缓冲优化（通信-计算重叠）** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/multibuffer_and_double_buffering.md`
- **减少遍历优化** · 567 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/pass_merge.md`
- **性能优化实践** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/perf_practice.md`
- **Permute / Layout-transform 算子优化** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/permute_layout_transform.md`
- **输出预初始化优化（Output Preinitialization）** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/preinitialized_output_optimization.md`
- **消除冗余的边界运算** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/redundant_boundary_operation.md`
- **Scalar 转 Vector 优化模式** · 479 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/scalar_to_vector.md`
- **激活函数 + 量化融合（Activation + Quantization Fusion）优化经验** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/swiglu_quant.md`
- **Tiling 优化** · 192 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/tiling_optimization.md`
- **分核优化** · 692 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/vector_core_partition.md`
- **Triton-Ascend 验证与调试工作流** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/workflow_and_debugging.md`
- **Workspace 物化解耦优化** · 408 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/workspace_decoupling.md`
- **高级优化技巧合集** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/advanced_optimization.md`
- **NPU Autotune 配置与 AutoTilingTuner** · 174 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/autotune_on_npu.md`
- **Bank Conflict 消除** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/bank_conflict.md`
- **care_padding 优化** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/care_padding.md`
- **compile_hint 与 Ascend 扩展 API 速查** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/compile_hint_extension.md`
- **NPU 编译参数速查（含 910_95 差异）** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/compile_params.md`
- **计算与分支优化** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/compute_and_branch.md`
- **Core Mapping 策略** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/core_mapping.md`
- **CV 流水线优化（Cube-Vector 融合）** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/cv_pipeline_optimization.md`
- **数据搬运与内存访问模式优化** · 197 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/data_movement_optimization.md`
- **数据类型与精度保护（910_95 平台）** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/data_type_precision.md`
- **数据类型优化：避免 Scalar Fallback** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/data_types.md`
- **NPU 数据类型支持矩阵与约束** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/data_types_ascend.md`
- **FixPipe 随路操作与 Bias 融合优化** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/fixpipe_bias_fusion.md`
- **Grid/Program ID 与 AI Core 对应关系** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/grid_and_program_id.md`
- **指令级并行 (ILP) 优化** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/ilp.md`
- **内存访问优化** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/memory_access_optimization.md`
- **在线算法** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/online_algorithms.md`
- **算子融合策略** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/operator_fusion.md`
- **并行化与循环优化** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/parallelism_and_loop.md`
- **NPU 性能优化策略总览** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/perf_optimization_overview.md`
- **性能调优方法论** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/performance_tuning.md`
- **Persistent Kernel 模式** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/persistent_kernels.md`
- **标量降级规避（910_95 / 910B）** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/scalar_degradation_avoidance.md`
- **SPMD 模型在 NPU 上的映射** · 124 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/spmd_on_npu.md`
- **Store 合并优化** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/store_merge.md`
- **同步与 Barrier（NPU 三层同步机制）** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/sync_and_barrier.md`
- **Tiling 策略** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/tiling.md`
- **Tiling 与 Grid 分核策略（910_95 平台）** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/tiling_and_grid_910_95.md`

下一页：page-007.md
