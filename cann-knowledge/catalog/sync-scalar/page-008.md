# sync-scalar: 491–560 / 674

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **errcode 264 workspace 标量 GM 访问崩溃** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/errcode_264_workspace_scalar_gm_access.md`
- **StorageShape scalar GetShapeSize 返回 1** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gert_storage_shape_scalar_getshapesize_returns_one.md`
- **GM 标量写须 SetValue(index, value) 两参** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/globaltensor_scalar_write_setvalue_two_args.md`
- **HardEvent V_V 编译报错 range[4,6]** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/hard_event_v_v_range_compile_error.md`
- **kernel 无 libc 数学函数需自实现** · 217 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_no_libc_math_scalar_self_implement.md`
- **LocalTensor operator()(offset) 标量元素读写** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/localtensor_operator_call_scalar_rw.md`
- **MIX 循环内 SyncAll 挂死，去 Cube 化绕开** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mix_syncall_deadlock_pure_aiv_escape.md`
- **混合精度 scales 用 TilingData 标志双绑定** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mixed_dtype_scales_tilingdata_flag_dual_tensor.md`
- **探针末尾 MTE2_S/MTE3_S 同步不可由 stream sync 替代** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mte2_s_mte3_s_stream_sync.md`
- **AscendC VEC API scalar-suffix convention (`Adds` vs `Add`, `Muls` vs `Mul`, ...)** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_169_ascendc_vec_api_scalar_suffix_convention_adds_vs_a.md`
- **Cross-core group barrier — every paired sub-block must participate symmetrically, and each Set must follow that sub-block's own real GM write** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_190_cross_core_group_barrier_every_paired_sub_block_mu.md`
- **V220→V351 vec compute-chain port — `PipeBarrier<PIPE_V>` + V→S sync MUST surround every reduce/elementwise/scalar-GetValue tuple; `WholeReduceMax` default `ORDER_VALUE_INDEX` packs index bits into the** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_195_v220_v351_vec_compute_chain_port_pipebarrier_pipe.md`
- **For a MIX cross-core-sync op, host/launch is the cascade differentiator — vendor FA = single-op-aclnn + crossCoreSync:1 + global flags + no-cascade, so isolation is necessary-not-sufficient** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_210_for_a_mix_cross_core_sync_op_host_launch_is_the_ca.md`
- **On V351 (A5), the MIX_AIC_1_2 AIV→AIC cross-core flag AND-reduces across BOTH AIV sub-blocks — an idle sub-block that never SetFlags hangs the AIC forever** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_212_on_v351_a5_the_mix_aic_1_2_aiv_aic_cross_core_flag.md`
- **bf16 scalar cast unsupported in bisheng — use SIMD Cast + GetValue** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_21_bf16_scalar_cast_unsupported_in_bisheng_use_simd_c.md`
- **A scalar-pipe write to GM is not coherent for a subsequent cube/MTE2 read — stage in UB then DataCopy (MTE3) to flush** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_225_a_scalar_pipe_write_to_gm_is_not_coherent_for_a_su.md`
- **A SIMT-scalar transcendental (exp/log/log1p inside a `__simt_callee__` / VF context) has a DIRECT scalar intrinsic shipped in the CANN binary — call it, do NOT hand-fit a range-reduce + Horner polynom** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_242_a_simt_scalar_transcendental_exp_log_log1p_inside.md`
- **TBuf has NO automatic synchronization — must use TQue or explicit sync** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_25_tbuf_has_no_automatic_synchronization_must_use_tqu.md`
- **PipeBarrier only supports MTE2/V/MTE3 — not PIPE_S** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_35_pipebarrier_only_supports_mte2_v_mte3_not_pipe_s.md`
- **A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the FULL NZ-padded extent (roun** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_545_a_scalar_vec_op_muls_adds_on_an_nz_format_buffer.md`
- **SIMD multi-core GM scalar write has 8-element granularity contention** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_70_simd_multi_core_gm_scalar_write_has_8_element_gran.md`
- **flag 改了不生效：opp/vendors 有 stale 同名算子安装** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opp_vendors_stale_mixed_flag_readings.md`
- **on dav-m200, scalar GM stores sharing a 64B GM sector with an in-flight MTE3 DataCopy (UB→GM) silently DROP th** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_123_on_dav_m200_scalar_gm_stores_sharing_a_64b_gm_se.md`
- **on dav-m200, multi-core scalar GM stores are silently DROPPED under concurrency, even with per-core exclusive** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_124_on_dav_m200_multi_core_scalar_gm_stores_are_sile.md`
- **msprof_perf_summary.py `_jsonl_scalar_value` crashes on attr inputs declared `dtype:"int"` whose value is a LIST — canonical quick run dies at the first such case [shared perf-infra, input-space gap]** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_179_msprof_perf_summary_py_jsonl_scalar_value_crashe.md`
- **`PipeBarrier<PIPE_ALL>()` between TBuf-resident MTE2 load and V compute → silent crash 507015 (V220-confirmed)** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_21_pipebarrier_pipe_all_between_tbuf_resident_mte2_lo.md`
- **SIMD binary-scalar VEC ops (Divs/Muls/Adds/Subs/Sub/Add/Mul/Div) reject int32, int16, int8, bf16 — supported dtype list verified** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md`
- **`MatmulImpl<>` with manual `CrossCoreSetFlag`/`WaitFlag` + `MIX_AIC_1_2` deadlock on V220 [V220, mixed-mode-sync]** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md`
- **`event_t(0)` for cube-internal pipe sync (`MTE1_M` / `M_FIX` / `MTE2_MTE1`) collides with AIC↔AIV CrossCoreSetFlag `FLAG_CANON_DONE` chain in `MIX_AIC_1_2` mode → silent hang [V220 + V351, mixed-mode-** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_35_event_t_0_for_cube_internal_pipe_sync_mte1_m_m_fix.md`
- **`TPipe::Reset()` frees the GLOBAL `g_tpipeImpl` event pool + buffer cursor on arch35 — a multi-stage MIX kernel that calls `pipe_->Reset()` between stages CANNOT carry persistent cross-call sync state** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_45_tpipe_reset_frees_the_global_g_tpipeimpl_event_poo.md`
- **bf16 Scalar Cast Failure** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_4_bf16_scalar_cast_failure.md`
- **a3/arch22 whole-device `SyncAll<false>()` MIX (cube+vector) requires `KERNEL_TYPE_MIX_AIC_1_1` (1:1) — the no-macro 1:2 default makes idle AIV cores skip the barrier loop → 507014 deadlock (narrows PB** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_57_a3_arch22_whole_device_syncall_false_mix_cube_vect.md`
- **PipeBarrier<PIPE_S> 编译失败 — 只保留目标架构支持的管线** · 240 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pipebarrier_pipe_s_is_not_supported.md`
- **PipeBarrier 的 PIPE_V 是全局枚举** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pipebarrier_pipe_v_uses_global_enum.md`
- **移植tiling误拒rank=0标量输入** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ported_tiling_rank0_scalar_rejected.md`
- **纯搬运算子 TBuf+PipeBarrier 逐块搬运骨架** · 379 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pure_copy_tbuf_pipebarrier_skeleton.md`
- **单 rank mock HCCL 致 AIC+AIV SyncAll 死锁** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rank_mock_hccl_aic_aiv_syncall.md`
- **RunOpApi 后缺 synchronize 致 .to 越序错** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/runopapi_requires_synchronize_before_tensor_ops.md`
- **RunOpApiV2 后再同步同 stream 段错误** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/runopapiv2_external_stream_sync_segfault.md`
- **标量NaN自比较后端Cannot select seto** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/scalar_nan_self_compare_seto_bitwise.md`
- **SetFlag/WaitFlag 是自由函数非 TPipe 成员** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/setflag_waitflag_free_functions.md`
- **ASCENDC_TPL_UINT_SEL漏UI_LIST标志致TBE报错** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tpl_uint_sel_requires_ui_list_flag.md`
- **vector API 整型缺位走标量 S 管线** · 522 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vector_api_integer_dtype_gaps_scalar_route.md`
- **VF 体内不支持标量指令——外提标量计算** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vf_body_scalar_instruction_backend_fatal.md`
- **aclnn 带宽计时用事件法剔除 host 开销** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/aclnn_timing_aclrtevent_kernel_only.md`
- **flat 多核切分 elementwise scalar 主导诊断** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/flat_elementwise_scalar.md`
- **msprof vec_ratio ≠ atomicAdd share (SIMT scalar GM reads also go through the VEC pipe)** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_20_msprof_vec_ratio_atomicadd_share_simt_scalar_gm_re.md`
- **Use aclrtEvent not chrono for NPU timing; memset outside event** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_5_use_aclrtevent_not_chrono_for_npu_timing_memset_ou.md`
- **msprof scalar_ratio > 0.9 = scalar pipe saturated → replace with VEC instructions** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_82_msprof_scalar_ratio_0_9_scalar_pipe_saturated_repl.md`
- **逐 rank 通信路径的 MTE2 主导诊断** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/rank_flag_mte2.md`
- **scatter 逐元素 RMW 实测偏离 MTE2 目标 4–5 量级** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/scatter_scalar_rmw_per_element_latency.md`
- **理论MTE3 bound落空：半精度链Scalar pipe被低估** · 121 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/theory_bound_missing_scalar_pipe_candidate.md`
- **950 mix(1,2) mode=2 漏 AIV1 flag 致全零** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/950_mix_1_2_mode_2_aiv1_flag.md`
- **arch35 bf16 标量 GM 读写丢 nan 位模式** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_bf16_gm_scalar_nan_lost.md`
- **Max/Min 逐元素非标量广播 须 Duplicate 至同长 buffer** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/count_based_max_min_is_element_wise_not_scalar_broadcast.md`
- **MTE2跨流水未建依赖导致UB读到上次调用残值** · 282 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/direct_mte_slot_reuse_needs_bidirectional_hard_events.md`
- **Reg 固定地址跨块累加需逐 VF PipeBarrier** · 269 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fixed_address_vf_accumulator_needs_per_call_barrier.md`
- **LocalTensor 混用 Vector/Scalar 写致输出全错** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/localtensor_vector_scalar_mixed_write.md`
- **DataCopyPad 后未先同步 Cast 导致低精度 NaN** · 364 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mte2_barrier_must_precede_vector_cast.md`
- **多核 S 管线标量写跨核不可见** · 290 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multicore_scalar_gm_write_cross_core_invisible.md`
- **Lane host dir is NOT auto-synced from project workspace — `deploy_to_npu_lane.sh` reads stale source unless caller manually copies first** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_113_lane_host_dir_is_not_auto_synced_from_project_work.md`
- **V220→V351 port discipline — explicit `V→MTE3` sync required between Cast/V-compute and DataCopy store; `PipeBarrier<PIPE_V>` is NOT sufficient** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_193_v220_v351_port_discipline_explicit_v_mte3_sync_req.md`
- **`pipe_barrier(PIPE_ALL)` is REQUIRED between VEC compute (Add/Muls) and MTE2 DataCopy when both touch the same UB buffer — silent precision corruption without it** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_247_pipe_barrier_pipe_all_is_required_between_vec_comp.md`
- **ACLRT_LAUNCH_KERNEL in pybind11 is ASYNC — aclrtSynchronizeStream(stream) before any local at::Tensor goes out** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_546_aclrt_launch_kernel_in_pybind11_is_async_aclrtsy.md`
- **host-derived scalar coefficients: compute in DOUBLE, cast ONCE to the tensor dtype** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_550_host_derived_scalar_coefficients_compute_in_doub.md`
- **bf16 Adds with a zero scalar may hit PB-4** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_74_bf16_adds_with_a_zero_scalar_may_hit_pb_4.md`
- **OpDef 属性截断 double β 致 NaN 失配** · 233 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/opdef_scalar_attr_double_truncation_nan.md`
- **单标量输出向量链丢值：回读标量侧再写** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scalar_output_v_mte3_loss_readback.md`
- **标量 S pipe 误用 TQue 导致输出全常数** · 308 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scalar_s_pipe_requires_explicit_mte_dependencies.md`
- **Select张量-标量模式索引不更新，须TT模式** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/select_tensor_scalar_mode_tensor_operand_no_update.md`

下一页：page-009.md
