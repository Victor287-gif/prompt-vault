# sync-scalar: 351–420 / 674

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Barrier 顺序与公式等价性** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/precision/barrier_ordering_and_formula_equivalence.md`
- **异步执行** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/programming_guide/compilation_and_execution/async_execution.md`
- **同步机制<a name="ZH-CN_TOPIC_0000002554691683"></a>** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/programming_guide/programming_model/ai_core_simt_programming/synchronization.md`
- **标量归约替换与原输入保留** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/reductions/scalar_loop_replacement_and_input_preservation.md`
- **Flash VF 的掩码、Cast 分布与标量槽** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/regbase/flash_vf_masks_casts_and_scalar_slots.md`
- **RegBase 同步的作用域与 UB 生产消费依赖** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/regbase/synchronization_layers.md`
- **Blaze 本地 MX 模板的 L1 尺度窗口与事件** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_local_l1_scale_windows_and_events.md`
- **Blaze-SHMEM 环形槽、释放信用与 Flag 预算** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_sample_ring_credit_and_flag_budget.md`
- **SIMT 共享存储与同步作用域** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/simt/shared_memory_and_sync_scope.md`
- **Radix mask精度、DMA交接与barrier协议** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/sorting/radix_mask_precision_dma_handoff_and_barrier_protocol.md`
- **向量递推中的真实依赖与别名冲突** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/alias_hazards_and_vector_recurrence.md`
- **APACE ReduceScatter 时序与通知计数** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/apace_reduce_scatter_schedule_and_credit_audit.md`
- **原子模式切换与首次覆盖顺序** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/atomic_mode_and_initial_write_order.md`
- **Attention 流水 slot 与跨核交接的平台约束** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/attention_schedule_slots_and_platform_handoffs.md`
- **Blaze 局部槽位与跨核通道的分别闭合** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/blaze_local_slots_and_cross_core_channels.md`
- **Blaze 两阶段初始化与空闲路径** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/blaze_two_phase_initialization_and_idle_paths.md`
- **广播紧凑源同步与复用槽身份** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/broadcast_compact_source_sync_and_slot_identity.md`
- **条件消费者与空任务发布** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/conditional_consumers_and_empty_publication.md`
- **Ascend C 跨核通知通道与缓冲复用** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/cross_core_channels_and_reuse.md`
- **双缓冲模板队列占用与事件闭合** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/double_buffer_template_queue_occupancy_and_event_closure.md`
- **事件完成语义与实际复用者** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/event_completion_and_actual_reuser.md`
- **事件所有权、重叠执行与原子完成** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/event_ownership_overlap_and_atomic_completion.md`
- **手动 Mmad 累加器与 UnitFlag 闭环** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/manual_mmad_accumulators_and_unit_flags.md`
- **Ping-Pong 初始许可与延迟 Atomic 刷新** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/ping_pong_event_credits_and_delayed_atomic_flush.md`
- **精度转换链中的同步作用域** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/precision_chain_dependency_scope.md`
- **队列身份与跨核 Pipe 合同** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/queue_identity_and_cross_core_pipe_contracts.md`
- **常驻Softmax状态的递推槽与复用事件** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/resident_softmax_state_slot_recurrence_and_reuse_events.md`
- **审查规则中的同步必须回到真实依赖** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/review_rules_require_actual_dependency_proofs.md`
- **环形缓冲的地址值与索引重绑定** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/ring_region_value_and_rebinding.md`
- **分块流水的启动、事件复用与排空** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/tiled_pipeline_event_lifecycle.md`
- **Vector 流水事件方向与读取位置** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/vector_pipeline_event_direction_and_read_placement.md`
- **Workspace Ring 的 Credit 与整行统计发布** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/workspace_ring_credits_and_bulk_statistics.md`
- **Scalar读写数据<a name="ZH-CN_TOPIC_0000002375401226"></a>** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/technical_appendix/concepts_and_terms/memory_access/scalar_read_write.md`
- **OneDim 广播tile数与标量驻留** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/onedim_broadcast_tile_counts_and_scalar_residency.md`
- **注册 PyTorch 适配器的 Tensor 身份与异步所有权** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/registry_adapter_tensor_identity_and_async_ownership.md`
- **Ascend C 向量广播的地址语义与标量替换边界** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/broadcast_addressing_and_scalar_substitution.md`
- **标量参数向量链与小工作量分发** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/scalar_tensor_chains_and_small_work_dispatch.md`
- **向量谓词、标量数学与 Axpy 修改语义** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/vector/vector_predicates_scalar_math_and_axpy_mutation.md`
- **BitwiseXorScalar 950 ACLNN 算子知识** · 917 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/math/bitwise_xor_scalar_950.md`
- **GeScalar 950 ACLNN 算子知识** · 928 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/math/ge_scalar_950.md`
- **InplaceBitwiseOrScalar 950 ACLNN 算子知识** · 596 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/math/inplace_bitwise_or_scalar_950.md`
- **InplaceGeScalar 950 ACLNN 算子知识** · 653 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/math/inplace_ge_scalar_950.md`
- **InplaceLtScalar 950 ACLNN 算子知识** · 940 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/math/inplace_lt_scalar_950.md`
- **InplacePowTensorScalar 950 ACLNN 算子知识** · 1071 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/math/math/inplace_pow_tensor_scalar_950.md`
- **ForeachAddScalar A2/A3 ACLNN 算子知识** · 686 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_add_scalar.md`
- **ForeachAddScalar 950 ACLNN 算子知识** · 680 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_add_scalar_950.md`
- **ForeachAddScalarList A2/A3 ACLNN 算子知识** · 633 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_add_scalar_list.md`
- **ForeachAddScalarList 950 ACLNN 算子知识** · 651 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_add_scalar_list_950.md`
- **ForeachAddcdivScalar A2/A3 ACLNN 算子知识** · 601 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcdiv_scalar.md`
- **ForeachAddcdivScalar 950 ACLNN 算子知识** · 540 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcdiv_scalar_950.md`
- **ForeachAddcdivScalarList A2/A3 ACLNN 算子知识** · 625 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcdiv_scalar_list.md`
- **ForeachAddcdivScalarList 950 ACLNN 算子知识** · 576 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcdiv_scalar_list_950.md`
- **ForeachAddcmulScalar A2/A3 ACLNN 算子知识** · 523 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcmul_scalar.md`
- **ForeachAddcmulScalar 950 ACLNN 算子知识** · 474 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcmul_scalar_950.md`
- **ForeachAddcmulScalarList A2/A3 ACLNN 算子知识** · 597 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcmul_scalar_list.md`
- **ForeachAddcmulScalarList 950 ACLNN 算子知识** · 511 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_addcmul_scalar_list_950.md`
- **ForeachDivScalar A2/A3 ACLNN 算子知识** · 634 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_div_scalar.md`
- **ForeachDivScalar 950 ACLNN 算子知识** · 618 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_div_scalar_950.md`
- **ForeachDivScalarList A2/A3 ACLNN 算子知识** · 540 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_div_scalar_list.md`
- **ForeachDivScalarList 950 ACLNN 算子知识** · 535 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_div_scalar_list_950.md`
- **ForeachLerpScalar A2/A3 ACLNN 算子知识** · 505 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_lerp_scalar.md`
- **ForeachLerpScalar 950 ACLNN 算子知识** · 513 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_lerp_scalar_950.md`
- **ForeachMaximumScalar A2/A3 ACLNN 算子知识** · 436 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_maximum_scalar.md`
- **ForeachMaximumScalar 950 ACLNN 算子知识** · 492 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_maximum_scalar_950.md`
- **ForeachMaximumScalarList A2/A3 ACLNN 算子知识** · 661 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_maximum_scalar_list.md`
- **ForeachMaximumScalarList 950 ACLNN 算子知识** · 636 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_maximum_scalar_list_950.md`
- **ForeachMinimumScalar A2/A3 ACLNN 算子知识** · 524 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_minimum_scalar.md`
- **ForeachMinimumScalar 950 ACLNN 算子知识** · 559 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_minimum_scalar_950.md`
- **ForeachMinimumScalarList A2/A3 ACLNN 算子知识** · 687 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_minimum_scalar_list.md`
- **ForeachMinimumScalarList 950 ACLNN 算子知识** · 609 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_minimum_scalar_list_950.md`

下一页：page-007.md
