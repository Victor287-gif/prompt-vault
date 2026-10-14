# pipeline-memory: 351–420 / 731

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **异步 Matmul 的结果空间与完成等待** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/async_result_buffers_and_completion.md`
- **Matmul 高阶替换与常驻回调坐标** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/high_level_replacement_and_resident_callbacks.md`
- **多阶段 Cube–Vector 流水的任务与槽位生命周期** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/multistage_cv_pipeline_lifetimes.md`
- **MX Matmul scale 驻留与预取权衡** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/mx_matmul_scale_residency_and_preload_tradeoffs.md`
- **TSCM 预加载与 L1 所有权** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/tscm_preload_is_l1_ownership.md`
- **双 Matmul 共享与卷积两级缓存模式** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/matmul/two_matmul_share_and_convolution_buffer_modes.md`
- **对齐单位、布局行距与填充值** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/alignment_units_layout_stride_and_padding_values.md`
- **BF16 别名缓冲与平台采样路径** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/bf16_alias_buffers_and_platform_sampling_paths.md`
- **Ascend C 缓冲所有权、流水交接与槽位布局** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/buffer_ownership_and_slot_layout.md`
- **DMA 原子写回的状态范围与输出初始化** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/dma_atomic_write_scope.md`
- **双缓冲物理槽与常驻参数预算** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/double_buffer_slot_counts_and_resident_parameters.md`
- **DSL 转换的 Cast Staging 与 DMA 所有权** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/dsl_cast_staging_and_dma_ownership.md`
- **高阶 Scratch 复用与 TPipe Reset 边界** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/high_level_scratch_reuse_and_tpipe_reset.md`
- **L1 常驻分区的字节布局与复用坐标** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/l1_resident_partition_coordinates.md`
- **手动双缓冲、三缓冲与二维槽的游标合同** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/manual_slot_cursors_and_event_ownership.md`
- **优化组合的资源与依赖合同** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/optimization_composition_resource_contracts.md`
- **分阶段别名与转置状态常驻** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/phase_aliasing_and_transposed_state_residency.md`
- **串行计算阶段的资源重建与预计算结果交接** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/phase_boundaries_and_resource_reinitialization.md`
- **常驻参数的扩宽视图与复用范围** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/promoted_constant_views_and_reuse_domains.md`
- **量化 Scatter 的复制归属与 Stream 顺序** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/quantized_scatter_copy_ownership_and_stream_order.md`
- **递推状态版本与梯度 lane 累计** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/recurrent_state_versions_and_gradient_lanes.md`
- **常驻参数与跨迭代状态的资源预算** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/resident_data_and_state_budget.md`
- **TBuf 常驻仍占 UB 且需要真实数据依赖** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/resident_tbuf_is_ub_and_requires_data_dependencies.md`
- **回退搬运的别名、原子重叠与短长度** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/rollback_copy_aliases_atomic_overlap_and_small_counts.md`
- **尾部 DMA、原子补零与压紧** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/tail_dma_atomic_zero_padding_and_compaction.md`
- **TQueBind 复用中的内存收益与中间计算** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/tquebind_reuse_and_intermediate_compute.md`
- **Cube/Vector 缓冲生命周期与输出行距** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/translated_cube_vector_buffers_and_output_stride.md`
- **UB Bank 映射与带 Padding 的物理步长** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/ub_bank_mapping_and_padded_pitches.md`
- **UB 分区字节核算与复制别名** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/ub_partition_byte_accounting_and_copy_aliases.md`
- **UB 驻留生命周期与架构相关bank端口** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/ub_resident_lifetimes_and_architecture_specific_bank_ports.md`
- **权重反量化的 UB、L1 与 Bias 联合预算** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/memory/weight_dequant_ub_l1_and_bias_budget.md`
- **Matmul高阶API开启L2 Cache切分<a name="ZH-CN_TOPIC_0000002341067578"></a>** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_l2_cache_split.md`
- **Matmul高阶API开启MTE2 Preload<a name="ZH-CN_TOPIC_0000002360914053"></a>** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_mte2_preload.md`
- **Matmul高阶API开启NBuffer33模板<a name="ZH-CN_TOPIC_0000002327075450"></a>** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/best_practices/matmul_tuning_cases/matmul_nbuffer33.md`
- **DoubleBuffer场景<a name="ZH-CN_TOPIC_0000002532228161"></a>** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_impl/vector_programming/double_buffer_scenario.md`
- **通过BT Buffer实现高效的bias计算<a name="ZH-CN_TOPIC_0000001918185182"></a>** · 336 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/bt_buffer_bias_compute.md`
- **通过FP Buffer存放量化参数实现高效随路量化<a name="ZH-CN_TOPIC_0000001947984233"></a>** · 342 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/fp_buffer_quantization.md`
- **通过L0C Buffer数据暂存实现高效的矩阵乘结果累加<a name="ZH-CN_TOPIC_0000001918025254"></a>** · 355 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/l0c_buffer_matmul_accumulate.md`
- **较小矩阵长驻L1 Buffer，仅分次搬运较大矩阵<a name="ZH-CN_TOPIC_0000001918185186"></a>** · 292 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/matrix_compute/small_matrix_l1_buffer.md`
- **避免同地址访问<a name="ZH-CN_TOPIC_0000002338550312"></a>** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_same_address_access.md`
- **避免bank冲突（NPU架构版本2201）<a name="ZH-CN_TOPIC_0000002499562140"></a>** · 378 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_2201.md`
- **避免bank冲突（NPU架构版本3510）<a name="ZH-CN_TOPIC_0000002531242067"></a>** · 305 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/avoid_bank_conflict_npu_arch_3510.md`
- **概述<a name="ZH-CN_TOPIC_0000002534006769"></a>** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/overview.md`
- **高效的使用搬运API<a name="ZH-CN_TOPIC_0000001846719836"></a>** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/efficient_transfer_api.md`
- **GM地址尽量512B对齐<a name="ZH-CN_TOPIC_0000001892919369"></a>** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md`
- **设置合理的L2 CacheMode<a name="ZH-CN_TOPIC_0000002356424662"></a>** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md`
- **L2 Cache切分<a name="ZH-CN_TOPIC_0000001893038949"></a>** · 221 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/l2_cache_split.md`
- **非对齐场景减少无效数据的搬运<a name="ZH-CN_TOPIC_0000002497818522"></a>** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/reduce_invalid_data_transfer.md`
- **非连续搬运场景减少搬运次数<a name="ZH-CN_TOPIC_0000002529778475"></a>** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/reduce_non_contiguous_transfer.md`
- **通过缩减Tensor ShapeInfo维度，优化栈空间<a name="ZH-CN_TOPIC_0000001918624089"></a>** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/reduce_shapeinfo_optimize_stack.md`
- **算子与高阶API共享临时Buffer<a name="ZH-CN_TOPIC_0000001893038929"></a>** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/shared_temp_buffer.md`
- **尽量一次搬运较大的数据块<a name="ZH-CN_TOPIC_0000001893771813"></a>** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/transfer_larger_data_blocks.md`
- **纯搬运类算子VECIN和VECOUT建议复用<a name="ZH-CN_TOPIC_0000001893038937"></a>** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/memory_access/vecin_vecout_reuse.md`
- **开启DoubleBuffer<a name="ZH-CN_TOPIC_0000001893038945"></a>** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md`
- **启用Iterate或IterateAll异步接口避免AIC/AIV同步依赖<a name="ZH-CN_TOPIC_0000001893695885"></a>** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_iterate_avoid_sync.md`
- **使用Unified Buffer提升内存访问效率<a name="ZH-CN_TOPIC_0000002511242528"></a>** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simd_simt_hybrid_optimization/memory_access/ub_memory_access.md`
- **避免UB的Bank冲突** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simt_operator_optimization/memory_access/avoid_ub_bank_conflict_simt.md`
- **DCache访问优化** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simt_operator_optimization/memory_access/dcache_access.md`
- **访存合并** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simt_operator_optimization/memory_access/memory_access_merge.md`
- **使用短向量类型提升效率** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simt_operator_optimization/memory_access/short_vector_efficiency.md`
- **通过类型对齐提升搬运效率** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/operator_practice/simt_operator_optimization/memory_access/type_alignment_transfer.md`
- **优化经验中的硬件模型与缓冲份数** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/performance/evolution_hardware_models_and_buffer_counts.md`
- **KV 工作集估计与 L2 策略候选** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/performance/kv_cache_footprint_for_l2_candidates.md`
- **MC2 缓存状态控制与预热的区别** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/performance/mc2_cache_conditioning_is_distinct_from_warmup.md`
- **MC2 流水递推与错误配平剪枝** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/performance/mc2_pipeline_recurrence_and_invalid_balance_pruning.md`
- **流水瓶颈证据与向量指令说明边界** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/performance/pipeline_bound_evidence_and_vector_instruction_claims.md`
- **Scalar 头部时间与真实访存指令** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/performance/scalar_prologue_and_actual_memory_operations.md`
- **深度池化 Direct Buffer 路径与同步修订** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/pooling/direct_buffer_depth_path_and_sync_revisions.md`
- **升精度计算中的状态、舍入边界与 UB 成本** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/precision/promoted_compute_state_and_memory_cost.md`
- **内存层级** · 117 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/memory_hierarchy.md`

下一页：page-007.md
