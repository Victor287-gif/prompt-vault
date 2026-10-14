# matmul: 141–210 / 567

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **HF32** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/apis/simd_api/tensor_api/matrix_compute/mmad_compute_key_features/hf32.md`
- **mmad计算关键特性说明<a name="ZH-CN_TOPIC_00000025121725631"></a>** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/apis/simd_api/tensor_api/matrix_compute/mmad_compute_key_features/mmad_compute_key_features.md`
- **unit_flag** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/apis/simd_api/tensor_api/matrix_compute/mmad_compute_key_features/unit_flag.md`
- **a3 MIX many-small-matmul cube primitives — P-P119 (hand-rolled cube vs IterateAll) + P-P120 (Neumann (I+P) L0C-fold)** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/examples/a3_mix_small_matmul_cube.md`
- **共享 KV head 的 Decode GEMV 合并** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/attention/grouped_decode_gemv_to_gemm.md`
- **KFC Launch ABI 与 Matmul 异步范围** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/attention/kfc_launch_abi_and_matmul_async_scope.md`
- **Blaze Matmul 逻辑形状与物理存储的一致性** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/blaze/matmul_logical_shape_and_physical_storage.md`
- **Blaze 注册 Matmul 的 mock Tiling 与属性传递** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/blaze/registry_matmul_mock_tiling_and_attribute_propagation.md`
- **CATLASS Matmul 模板索引与实例化合同** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/matmul_template_catalog_and_instantiation_contracts.md`
- **APACE AllToAll Matmul 的全局与局部坐标** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/communications/apace_alltoall_matmul_global_local_coordinates.md`
- **APACE localMatmul 累加模式与 L0C 容量** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/communications/apace_local_matmul_accumulation_modes_and_l0c.md`
- **MC2 本地 Matmul 与通信依赖的重叠** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/communications/local_matmul_and_communication_dependencies.md`
- **异步 Matmul 的结果空间与完成等待** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/async_result_buffers_and_completion.md`
- **Matmul 基本块流量模型与输出对齐** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/base_block_traffic_model_and_alignment.md`
- **Batch/Group Matmul 分区身份与分支顺序** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/batch_group_matmul_partition_identity_and_branch_order.md`
- **Count 分组 Matmul 的参考语义与 API 模式边界** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/count_group_reference_and_api_mode_boundaries.md`
- **Matmul 高阶 API 的执行模式与 Host 契约** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/execution_mode_and_host_contracts.md`
- **Fixpipe 直达 UB 的镜像布局与复用槽合同** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/fixpipe_ub_mirror_and_slot_reuse_contract.md`
- **GroupedMatmul 的组内尺寸、地址与反量化坐标** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/group_shapes_and_dequant_coordinates.md`
- **Matmul 高阶替换与常驻回调坐标** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/high_level_replacement_and_resident_callbacks.md`
- **IBShare 共享范围与 Split K 初始化** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/ibshare_scope_and_split_k_initialization.md`
- **Matmul 的 K 分段与部分和归约** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/k_partition_and_partial_sum_reduction.md`
- **手工 Dot Product 与 Cube 示例的适用条件** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/manual_dot_product_and_cube_example_contracts.md`
- **Matmul fallback 资源等式与评分单位** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/matmul_fallback_resource_equations_and_score_units.md`
- **Matmul 全载和 StreamK 能力条件** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/matmul_full_load_and_streamk_capability_conditions.md`
- **MIX 重写的工作映射与资源归属** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/mix_rewrite_work_mapping_and_resource_ownership.md`
- **多阶段 Cube–Vector 流水的任务与槽位生命周期** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/multistage_cv_pipeline_lifetimes.md`
- **MX Matmul scale 驻留与预取权衡** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/mx_matmul_scale_residency_and_preload_tradeoffs.md`
- **幂零矩阵的有限 Neumann 三角逆** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/nilpotent_neumann_triangular_inverse.md`
- **Matmul 搬运模板与 K 重排合同** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/transport_templates_and_k_reorder_contracts.md`
- **TSCM 预加载与 L1 所有权** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/tscm_preload_is_l1_ownership.md`
- **双 Matmul 共享与卷积两级缓存模式** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/two_matmul_share_and_convolution_buffer_modes.md`
- **GroupedMatmul算子性能调优案例<a name="ZH-CN_TOPIC_0000002107767061"></a>** · 255 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/grouped_matmul_tuning.md`
- **AIV核上的ND2NZ格式转换<a name="ZH-CN_TOPIC_0000002340907814"></a>** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/aiv_nd2nz_conversion.md`
- **Matmul高阶API开启IBShare模板共享A和B矩阵数据<a name="ZH-CN_TOPIC_0000002088668238"></a>** · 149 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_ibshare_ab.md`
- **Matmul高阶API开启IBShare模板共享B矩阵数据<a name="ZH-CN_TOPIC_0000002340907802"></a>** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_ibshare_b.md`
- **Matmul高阶API开启L2 Cache切分<a name="ZH-CN_TOPIC_0000002341067578"></a>** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_l2_cache_split.md`
- **Matmul高阶API开启MDL模板<a name="ZH-CN_TOPIC_0000002374905781"></a>** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_mdl_template.md`
- **Matmul高阶API开启MTE2 Preload<a name="ZH-CN_TOPIC_0000002360914053"></a>** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_mte2_preload.md`
- **Matmul高阶API开启多核K轴错峰访问内存<a name="ZH-CN_TOPIC_0000002374905793"></a>** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_multi_core_k_stagger.md`
- **Matmul高阶API开启多核切K<a name="ZH-CN_TOPIC_0000002353756705"></a>** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_multi_core_split_k.md`
- **Matmul高阶API开启NBuffer33模板<a name="ZH-CN_TOPIC_0000002327075450"></a>** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_nbuffer33.md`
- **Matmul性能优化策略总览<a name="ZH-CN_TOPIC_0000002360993885"></a>** · 172 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_optimization_overview.md`
- **Matmul高阶API开启纯Cube模式<a name="ZH-CN_TOPIC_0000002326915590"></a>** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_pure_cube.md`
- **Matmul高阶API开启Tiling全量常量化<a name="ZH-CN_TOPIC_0000002374865629"></a>** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_tiling_constant.md`
- **Matmul算子优化Tiling策略<a name="ZH-CN_TOPIC_0000001893038953"></a>** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_tiling_strategy.md`
- **Matmul高阶API开启UnitFlag<a name="ZH-CN_TOPIC_0000002353876489"></a>** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_unit_flag.md`
- **4:2稀疏矩阵乘<a name="ZH-CN_TOPIC_0000002532388135"></a>** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/4_2_sparse_matmul.md`
- **Batch Matmul基础功能<a name="ZH-CN_TOPIC_0000002532228171"></a>** · 122 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/batch_matmul_basics.md`
- **Batch Matmul复用Bias矩阵<a name="ZH-CN_TOPIC_0000002500548102"></a>** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/batch_matmul_reuse_bias.md`
- **Matmul特性介绍<a name="ZH-CN_TOPIC_0000002500468238"></a>** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/matmul_features.md`
- **矩阵乘输出的Channel拆分<a name="ZH-CN_TOPIC_0000002532228167"></a>** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/matmul_output_channel_split.md`
- **矩阵乘输出的N方向对齐<a name="ZH-CN_TOPIC_0000002532228169"></a>** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/matmul_output_n_alignment.md`
- **矩阵乘输出的量化/反量化** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/matmul_output_quant_dequant.md`
- **MxMatmul场景** · 269 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/mx_matmul_scenario.md`
- **单次矩阵乘局部输出<a name="ZH-CN_TOPIC_0000002500548100"></a>** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/single_matmul_local_output.md`
- **TSCM输入的矩阵乘<a name="ZH-CN_TOPIC_0000002500468244"></a>** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/tscm_input_matmul.md`
- **通过L0C Buffer数据暂存实现高效的矩阵乘结果累加<a name="ZH-CN_TOPIC_0000001918025254"></a>** · 355 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/l0c_buffer_matmul_accumulate.md`
- **Matmul开启AtomicAdd选项<a name="ZH-CN_TOPIC_0000001925216954"></a>** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/matmul_atomic_add.md`
- **高阶极限打磨<a name="ZH-CN_TOPIC_000000_MATMUL_LEAKYRELU_ADV"></a>** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/typical_operator_cases/simd_operator_impl/matmul_leakyrelu_fusion_operator/extreme_optimization.md`
- **入门功能落地<a name="ZH-CN_TOPIC_000000_MATMUL_LEAKYRELU_ENTRY"></a>** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/typical_operator_cases/simd_operator_impl/matmul_leakyrelu_fusion_operator/getting_started.md`
- **中级性能优化<a name="ZH-CN_TOPIC_000000_MATMUL_LEAKYRELU_MID"></a>** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/typical_operator_cases/simd_operator_impl/matmul_leakyrelu_fusion_operator/intermediate_optimization.md`
- **高级极限打磨** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/typical_operator_cases/simd_operator_impl/matmul_matrix_operator/advanced_extreme_optimization.md`
- **入门功能落地<a name="ZH-CN_TOPIC_000000_ADD_ENTRY"></a>** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/typical_operator_cases/simd_operator_impl/matmul_matrix_operator/getting_started.md`
- **中级性能优化** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/typical_operator_cases/simd_operator_impl/matmul_matrix_operator/intermediate_optimization.md`
- **含有Matmul高阶API的算子精度问题<a name="ZH-CN_TOPIC_0000002070127457"></a>** · 602 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/programming_guide/appendix/faq/matmul_advanced_api_precision.md`
- **Blaze 本地 SWAT 调度坐标与尾块覆盖** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_swat_scheduler_coordinates_and_tail_coverage.md`
- **Blaze SWAT Tiling 搜索与资源边界** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_swat_tiling_search_and_resource_bounds.md`
- **CoC Matmul Tile、累加器与 Workspace 交接** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/coc_matmul_tiles_accumulators_and_workspace_handoffs.md`
- **手动 Mmad 累加器与 UnitFlag 闭环** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/manual_mmad_accumulators_and_unit_flags.md`

下一页：page-004.md
