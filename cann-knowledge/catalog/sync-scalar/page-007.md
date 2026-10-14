# sync-scalar: 421–490 / 674

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **ForeachMulScalar A2/A3 ACLNN 算子知识** · 734 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_mul_scalar.md`
- **ForeachMulScalar 950 ACLNN 算子知识** · 717 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_mul_scalar_950.md`
- **ForeachMulScalarList A2/A3 ACLNN 算子知识** · 596 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_mul_scalar_list.md`
- **ForeachMulScalarList 950 ACLNN 算子知识** · 590 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_mul_scalar_list_950.md`
- **ForeachPowScalar A2/A3 ACLNN 算子知识** · 603 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_pow_scalar.md`
- **ForeachPowScalar 950 ACLNN 算子知识** · 557 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_pow_scalar_950.md`
- **ForeachPowScalarAndTensor A2/A3 ACLNN 算子知识** · 589 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_pow_scalar_and_tensor.md`
- **ForeachPowScalarList A2/A3 ACLNN 算子知识** · 691 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_pow_scalar_list.md`
- **ForeachSubScalar A2/A3 ACLNN 算子知识** · 627 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_sub_scalar.md`
- **ForeachSubScalar 950 ACLNN 算子知识** · 572 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_sub_scalar_950.md`
- **ForeachSubScalarList A2/A3 ACLNN 算子知识** · 668 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_sub_scalar_list.md`
- **ForeachSubScalarList 950 ACLNN 算子知识** · 614 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/foreach/foreach_sub_scalar_list_950.md`
- **SyncBatchNormGatherStats 950 ACLNN 算子知识** · 844 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/nn/norm/sync_batch_norm_gather_stats_950.md`
- **DistributeBarrier A3 ACLNN 算子知识** · 700 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/distribute_barrier.md`
- **DistributeBarrier 950 ACLNN 算子知识** · 680 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/transformer/mc2/distribute_barrier_950.md`
- **ForeachAddcdivScalar 950 Direct Invoke 算子知识** · 459 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/operators/direct/nn/foreach/foreach_addcdiv_scalar_950.md`
- **SIMD PipeBarrier and DataCopy alignment** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/f_p4_simd_pipebarrier_and_datacopy_alignment.md`
- **Cross Core Sync 与 WorkspaceQueue 详细参考** · 362 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/fa_cross_core_sync_workspacequeue.md`
- **Lift loop-invariant scalar instructions out of per-tile loops** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_116_lift_loop_invariant_scalar_instructions.md`
- **AiCore-only pipelines must use PipeBarrier<PIPE_ALL>, not SyncAll — cross-core barriers cause multi-core hangs** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_213_aicore_only_pipebarrier_not_syncall.md`
- **In a Reset-bearing multi-stage MIX cube kernel, drive cube-internal L0 pipe sync with Reset-safe dynamic back-to-back FetchEventID fences — NOT persistent Buffer<> credits, NOT hardcoded event ids** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_223_reset_safe_fetcheventid_l0_fences.md`
- **Whole-grid SyncAll requires all blocks co-resident — cap blockDim at min(work, cores) and tile extra work into a per-block loop with no-op-guarded slots so every block makes the SAME SyncAll count** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_226_whole_grid_syncall_blockdim_cap_equal_trip.md`
- **TQue bug RESOLVED in CANN 9.0.0 — prefer TQue over PipeBarrier** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_4_tque_resolved_prefer_over_pipebarrier.md`
- **VEC pipeline sync choice — TQue auto-rotation vs manual TBuf event sync** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_94_vec_pipeline_sync_tque_vs_tbuf.md`
- **De-scalarize flash online-softmax — hand-rolled mem-based VEC online-softmax + precision-safety triad (replaces `SoftmaxFlashV2` scalar pole)** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p101_de_scalarize_flash_online_softmax_hand_rolled_me.md`
- **SetFlag/WaitFlag fine-grained event sync (replacing PipeBarrier<PIPE_ALL>)** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p26_setflag_waitflag_fine_grained_event_sync_replaci.md`
- **bf16 scalar conversion — Cast(bf16→float) + GetValue** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p27_bf16_scalar_conversion_cast_bf16_float_getvalue.md`
- **Batch Preload Cache — eliminating scalar GM read bottleneck** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p29_batch_preload_cache_eliminating_scalar_gm_read_b.md`
- **WarpReduceAddSync + warp-lane-0 atomic** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p2_warpreduceaddsync_warp_lane_0_atomic.md`
- **fp16/bf16 Scalar Kernel Argument Passing** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p30_fp16_bf16_scalar_kernel_argument_passing.md`
- **Per-uid 8-aligned GM scalar slot (multi-core scalar write)** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p49_per_uid_8_aligned_gm_scalar_slot_multi_core_scal.md`
- **Reciprocal via tensor Div(1, x), not scalar `1/x`** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p54_reciprocal_via_tensor_div_1_x_not_scalar_1_x.md`
- **All "scalar arithmetic" in bf16/fp16 must go through the VEC path** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p56_all_scalar_arithmetic_in_bf16_fp16_must_go_throu.md`
- **Row-Scalar VEC Multiply — avoiding per-row scalar multiply on the Scalar pipe** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p62_row_scalar_vec_multiply_avoiding_per_row_scalar.md`
- **VEC↔Scalar interleaved compute with paired V_S / S_V flags** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p64_vec_scalar_interleaved_compute_with_paired_v_s_s.md`
- **Manual TBuf pipeline with explicit `SetFlag/WaitFlag<MTE2_V>` event sync (V220-confirmed; A5 likely-applicable)** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p75_manual_tbuf_pipeline_with_explicit_setflag_waitf.md`
- **Aligned-base scratch via `Duplicate(0) + scalar SetValue` for inline VEC reductions with unaligned index offsets** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p76_aligned_base_scratch_via_duplicate_0_scalar_setv.md`
- **V220 PipeBarrier<PIPE_V> coalescing — remove redundant intra-pipe barriers** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p92_v220_pipebarrier_pipe_v_coalescing_remove_redund.md`
- **`LocalMemBar<MemType::UB>` replaces `SetFlag<MTE2_V>`+`WaitFlag` in A5 L2 MicroAPI [V351, microapi-sync]** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p95_localmembar_memtype_ub_replaces_setflag_mte2_v_w.md`
- **autogen 安装头标量属性按值声明与库 aclScalar* ABI 不符** · 250 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/161001_autogen_scalar_aclscalar_abi_mismatch.md`
- **安装头 scalar 类型与源码不符致堆破坏 161002** · 204 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/161002_autogen_header_scalar_type_mismatch.md`
- **ACLNN L2 缓存键漏 scalar 致批跑误命中** · 172 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_l2_cache_key_missing_scalar_partial_mismatch.md`
- **ACL 事件计时 API 是 aclrtEventElapsedTime** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclrt_event_elapsed_time_no_duration_api.md`
- **aclrtMemcpyAsync 局部变量 use-after-free** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclrtmemcpyasync_use_after_free.md`
- **aclrtMemsetD32Async 在 CANN 9.x 被移除** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclrtmemsetd32async_cann_9_x.md`
- **aclScalar 无 Get()，读标量用 ToInt64()** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclscalar_no_get_use_toint64.md`
- **AcquireChannel 无 UB_MEM 链路致 syncBuf 失败** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/acquirechannel_ub_mem_syncbuf.md`
- **双AIV CrossCoreFlag 的 subBlock 契约缺陷** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aiv_crosscoreflag_subblock_507015.md`
- **arch22 禁 half 标量运算，NaN 检测须位级** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch22_half_scalar_op_banned.md`
- **arch35 HardEvent 无 FIX_V，显式同步编译失败** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_hard_event_no_fix_v.md`
- **arch35 SyncAll 无 HardEvent 模板形式** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_syncall_isaivonly_signature.md`
- **arch35 无 Xor/And/ShiftRight 标量重载，骨架改标量** · 275 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_vector_bitop_no_scalar_overload.md`
- **BF16 标量 cast 后端失败—改用位序键** · 295 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bf16_scalar_compare_by_raw_bits.md`
- **bisheng 不支持 bf16 标量强转时用位布局精确提升到 fp32** · 823 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bf16_scalar_to_fp32_bit_extension.md`
- **CrossCore 反压 flag 方向错致通算流水死锁** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/crosscore_flag.md`
- **DynamicCompileStaticFlag(false) 致 E40021** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/dynamic_compile_static_flag_false_e40021.md`
- **`AscendC::SyncFunc<>` does not exist** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_13_ascendc_syncfunc_does_not_exist.md`
- **`PipeBarrier<PIPE_S>` not valid on Ascend950PR** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_15_pipebarrier_pipe_s_not_valid_on_ascend950pr.md`
- **dav-m200 (Ascend310P3, arch 200x): the scalar-subtraction VEC primitive Subs is absent (error: no template nam** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_194_dav_m200_ascend310p3_arch_200x_the_scalar_subtra.md`
- **unknown type name '<Op>TilingData' flagged on an upstream op_kernel/.h line means the tiling struct is defined** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_200_unknown_type_name_op_tilingdata_flagged_on_an_up.md`
- **dav-m200 arch impl has NO Divs (tensor÷scalar)** · 139 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_208_dav_m200_arch_impl_has_no_divs_tensor_scalar.md`
- **dav-m200: `SetFlag`/`WaitFlag<HardEvent::MTE2_S>` DEADLOCKS for scalar consumers of a TQue-dequeued VECIN buffer** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_294_dav_m200_setflag_waitflag_hardevent_mte2_s_deadl.md`
- **dav-m200: a pure 2-byte GM scalar store (`GlobalTensor<T>::SetValue`, T∈{half,int16}) sharing a 64B line with MTE3-DMA-written data zero-clobbers the PRECEDING 32B block, in either issue order** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_296_dav_m200_a_pure_2_byte_gm_scalar_store_globalten.md`
- **`deploy_to_a5.sh` syncs `current_task/` not `workspace/<op>/kernel/`** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_30_deploy_to_a5_sh_syncs_current_task_not_workspace_o.md`
- **`REGIST_MATMUL_OBJ` lands NO-SYNC branch with single `-DASCENDC_MATMUL_AICORE` — runtime MPU 507015 once cube uses the matmul object [V220+V351, ALL_MODES, build-config + KFC-sync-activation]** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_57_regist_matmul_obj_lands_no_sync_branch_with_single.md`
- **Scalar-pipe accumulator array (float[]) ~20× slower than VEC-pipe accumulator (TBuf<VECCALC>) for per-element reduction ops on V220** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_61_scalar_pipe_accumulator_array_float_20_slower_than.md`
- **`EVENT_ID4..7` / `PIPE_FIX` undefined identifier in host_bisheng_obj preview compile — wrap kernel-side includes with `#if defined(__CCE_AICORE__)` guard** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_64_event_id4_7_pipe_fix_undefined_identifier_in_host.md`
- **A scalar read (`GetValue` / raw `__gm__` deref) of data just `DataCopyPad`'d into UB needs `HardEvent::MTE2_S` — `MTE2_V` alone leaves the scalar pipe racing the DMA → garbage indices → OOB GM access ** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md`
- **`SToMTE3Sync()` after V-pipe Cast/Muls/Adds → deterministic garbage on V351 — must use `VToMTE3Sync()`** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_81_stomte3sync_after_v_pipe_cast_muls_adds_determinis.md`
- **cannbot `grade_batch` returns numpy scalar types + full per-case arrays → `TypeError: not JSON serializable` and a 100s-of-MB verdict — coerce recursively to native + trim to a scalar summary before p** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_85_cannbot_grade_batch_returns_numpy_scalar_types_ful.md`

下一页：page-008.md
