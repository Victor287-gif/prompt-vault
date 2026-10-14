# reduction-broadcast: 561–630 / 1263

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **归约算法选择与数值边界** · 22 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/algorithm_routing_and_accuracy_limits.md`
- **归约轴映射与全载、分载设计** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/axis_mapping_and_load_partition.md`
- **Blaze Pure Reduce 的 partial 状态与模板缺口** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/blaze_pure_reduce_partial_state.md`
- **Block 归约压缩的语义与尾部条件** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/block_compaction_requires_matching_semantics.md`
- **自定义归约树工作量与 Welford 计数** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/custom_reduction_tree_work_and_welford_counts.md`
- **确定性梯度与 Workspace 发布** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/deterministic_gradients_and_workspace_publication.md`
- **私有 Workspace 与固定顺序归并** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/deterministic_workspace_merge.md`
- **折叠归约的进度与余项保留** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/folding_progress_and_remainder_preservation.md`
- **分片极值归约的索引与并列值语义** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/index_tracking_and_tie_semantics.md`
- **归约扫描中的目标 logit 保存** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/inline_target_logit_capture.md`
- **Reduce 的有效范围、存储布局与临时空间契约** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/interface_and_storage_contracts.md`
- **多输出归约的存活缓冲预算** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/live_buffer_budget.md`
- **MSE 的 Lane 累加与跨核归约** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/mse_lane_and_cross_core_reductions.md`
- **ORI 归约路由、填充值与方差遍历** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/ori_routes_padding_and_variance_passes.md`
- **归约 fallback 轴路由与序列化字段遗漏** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/reduction_fallback_axis_routes_and_abi_omissions.md`
- **归约 fallback 容量单位与分组 workspace** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/reduction_fallback_capacity_units_and_group_workspace.md`
- **标量归约替换与原输入保留** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/scalar_loop_replacement_and_input_preservation.md`
- **AR FullLoad归约临时区随批量行数增长** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_ar_full_load_binary_scratch_scales_with_rows.md`
- **ARA FullLoad的零填充行与四路折叠** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_ara_full_load_zero_row_and_four_way_fold.md`
- **Softmax 公共Cast视图与DMA步长单位** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_base_cast_views_and_dma_stride_units.md`
- **Softmax 公共归约树的有限适用范围** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_common_tree_helpers_have_finite_reduction_domains.md`
- **Softmax 有限域稳定性与 NaN/Inf 处理** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_finite_domain_and_special_values.md`
- **ARA Online的chunk状态与全掩码初始化** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_online_chunk_state_and_all_masked_bootstrap.md`
- **Softmax 重算、折叠与统计量归并** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_recompute_fold_and_statistic_merge.md`
- **Softmax Recompute配对、缓存根与缓冲角色** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_recompute_pairing_cache_root_and_buffer_roles.md`
- **整行 Softmax 的容量与非有限值策略** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_row_capacity_and_nonfinite_policy.md`
- **AR SmallR转置预取与padding生命周期** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_small_r_transpose_preload_and_padding_lifetimes.md`
- **独立 Softmax 的驻留、重读与在线统计** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_storage_and_input_passes.md`
- **Softmax 模板的轴映射与Norm扫描次数** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/softmax_template_axis_mapping_and_norm_pass_counts.md`
- **特化归约 Mask、Lane 宽度与采样点组** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/specialized_reduction_masks_lane_widths_and_point_groups.md`
- **统计遍历、融合与真实搬运量** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/statistics_passes_fusion_and_traffic_accounting.md`
- **分块统计归约中的 Welford 状态与跨组合并** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/streaming_statistics_and_group_merge.md`
- **Welford 方差修正与 Running 统计量的语义** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/variance_correction_and_running_statistics.md`
- **向量折叠归约与非二次幂长度** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/vector_folding_and_non_power_of_two_lengths.md`
- **AscReduceMax** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/regbase/asc_reduce_max_reference_contract.md`
- **AscReduceMin** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/regbase/asc_reduce_min_reference_contract.md`
- **AscReduceSum** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/regbase/asc_reduce_sum_reference_contract.md`
- **归一化类型矩阵与分发边界** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/registration/normalization_type_matrix_and_dispatch_boundaries.md`
- **SHMEM 布局单位、尾块地址与归约复位** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/layout_units_tail_partition_and_reduction_reset.md`
- **SHMEM 归约中的贡献计数与目标写权** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/reduction_contribution_ownership.md`
- **APACE ReduceScatter 时序与通知计数** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/apace_reduce_scatter_schedule_and_credit_audit.md`
- **广播紧凑源同步与复用槽身份** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/broadcast_compact_source_sync_and_slot_identity.md`
- **常驻Softmax状态的递推槽与复用事件** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md`
- **CSV 用例转换中的形状与预期值保真** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/testing/preserving_case_semantics_during_csv_conversion.md`
- **Broadcast 合轴、路由与字段一致性** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/broadcast_collapse_routing_and_field_consistency.md`
- **多维广播的坐标映射与展开阶段** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/broadcast_coordinates_and_expansion_stage.md`
- **Broadcast fallback 尾块公式与缩块状态** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/broadcast_fallback_tail_equations_and_shrink_state.md`
- **广播模板容量、分核与DMA有效长度** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/broadcast_template_capacity_partition_and_dma_limits.md`
- **广播模板选路与每tile有效shape** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/broadcast_template_routes_and_per_tile_shapes.md`
- **卷积的空间分核与分层资源预算** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/convolution_spatial_and_reduction_partition.md`
- **动态 UB Broadcast 的重载与存储约束** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/dynamic_broadcast_overloads_and_storage_contracts.md`
- **OneDim 广播tile数与标量驻留** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/onedim_broadcast_tile_counts_and_scalar_residency.md`
- **Softmax Host容量公式与空维失效条件** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/softmax_host_capacity_formulas_and_empty_shape_failures.md`
- **Ascend C 向量广播的地址语义与标量替换边界** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/broadcast_addressing_and_scalar_substitution.md`
- **Norm 融合的 Affine 重排与流水边界** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/norm_fusion_affine_rounding_and_pipeline_limits.md`
- **逐行广播的成本与读后写依赖** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/row_broadcast_cost_and_read_after_write.md`
- **转置、寄存器循环与 Scatter 累加** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/transpose_register_loops_and_scatter_accumulation.md`
- **Vector 效率的mask生命周期与归约接口形态** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/vector_efficiency_mask_lifecycle_and_reduction_api_shapes.md`
- **Cat 950 ACLNN 算子知识** · 1053 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/cat_950.md`
- **ChunkCat A2/A3 ACLNN 算子知识** · 444 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/chunk_cat.md`
- **ChunkCat 950 ACLNN 算子知识** · 589 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/chunk_cat_950.md`
- **CircularPad2d A2/A3 ACLNN 算子知识** · 565 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/circular_pad_2d.md`
- **CircularPad2dBackward A2/A3 ACLNN 算子知识** · 544 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/circular_pad_2d_backward.md`
- **CircularPad2dBackward 950 ACLNN 算子知识** · 654 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/circular_pad_2d_backward_950.md`
- **CircularPad3d A2/A3 ACLNN 算子知识** · 1109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/circular_pad_3d.md`
- **CircularPad3dBackward A2/A3 ACLNN 算子知识** · 996 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/circular_pad_3d_backward.md`
- **ClampMin 950 ACLNN 算子知识** · 628 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/clamp_min_950.md`
- **ConfusionTranspose 950 ACLNN 算子知识** · 793 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/confusion_transpose_950.md`
- **DiagFlat A2/A3 ACLNN 算子知识** · 547 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/diag_flat.md`
- **Hardtanh 950 ACLNN 算子知识** · 655 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/conversion/hardtanh_950.md`

下一页：page-010.md
