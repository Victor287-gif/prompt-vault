# pipeline-memory: 491–560 / 731

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **DataCopyPad 第4参类名 CTAD 失配** · 260 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_4th_param_ctad_bare_braces.md`
- **DataCopyPad CopyIn 小输入 32B DMA 越界** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_copyin_32b_dma.md`
- **多块 DataCopyPad dst 基址非 32B 对齐崩溃** · 181 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_dst_32b.md`
- **DataCopyPad 加载缺少 Pad 参数导致编译失败** · 220 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_gm_to_ub_requires_pad_params.md`
- **DataCopyPad 无匹配重载 LocalTensor 实参不一致** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_localtensor.md`
- **DataCopyPad 五参编译失败—LocalTensor 子视图替代偏移** · 171 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_offset_via_tensor_subview.md`
- **507035 C 超 per_buf_elems 致 Phase1 UB 越界** · 214 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_phase1_chunk_loop_per_buf_elems_507035.md`
- **DataCopyPad rightPad 字段为 uint8_t** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_rightpad_uint8_t.md`
- **DataCopyPad UB→GM 四参调用导致编译失败** · 281 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_ub_to_gm_has_no_pad_params.md`
- **DataCopyPad UB→GM 不收 padParams，3 参重载** · 230 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_ub_to_gm_no_pad_params_3arg.md`
- **DataCopyPad 无 UB→UB 重载，改 GM 跨步直取** · 294 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopypad_ub_to_ub_no_overload.md`
- **DataCopyPad blockLen uint16_t 窄化截断风险** · 225 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/datacopyparams_blocklen_uint16_silent_narrowing.md`
- **设备侧 N*E 缓冲上限须按 UB 推导扩容** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/device_buffer_cap_ub_derivation_resize.md`
- **DataCopy alignment overwrite in strided/chunked copies** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_16_datacopy_alignment_overwrite_in_strided_chunked_co.md`
- **VECIN-only pipeline cannot do GM→UB→GM pass-through** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_21_vecin_only_pipeline_cannot_do_gm_ub_gm_pass_throug.md`
- **Multi-block aligned DataCopy overwrite race** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_22_multi_block_aligned_datacopy_overwrite_race.md`
- **DataCopyPad UB→GM crashes on Ascend950PR (507035)** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_23_datacopypad_ub_gm_crashes_on_ascend950pr_507035.md`
- **dav-m200: `SetFlag`/`WaitFlag<HardEvent::MTE2_S>` DEADLOCKS for scalar consumers of a TQue-dequeued VECIN buffer** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_294_dav_m200_setflag_waitflag_hardevent_mte2_s_deadl.md`
- **Top-K buffer `effective_kept` vs `buffer_len` — always compute effective_kept separately** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_32_top_k_buffer_effective_kept_vs_buffer_len_always_c.md`
- **32B-aligned `DataCopy(GM, ub, count)` to under-allocated `torch::empty({C}, ...)` overflows adjacent torch tensors** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_41_32b_aligned_datacopy_gm_ub_count_to_under_allocate.md`
- **A5 V351 runtime error 507035 + EZ9999 `errcode:(95) MTE instruction DDR address out of range` — kernel GM extent overshoot (fault tree) [V351, port_a3_to_a5]** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_53_a5_v351_runtime_error_507035_ez9999_errcode_95_mte.md`
- **TBuf workspace buffer used without `pipe_.InitBuffer()` — unallocated UB access → 507035 vector core exception on V220** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_62_tbuf_workspace_buffer_used_without_pipe_initbuffer.md`
- **`TPipe::InitBuffer(TQue<T>&)` requires 3 arguments — different from TBuf InitBuffer which takes 2** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_70_tpipe_initbuffer_tque_t_requires_3_arguments_diffe.md`
- **DataCopy with raw `__gm__` pointer silently returns zeros on Ascend950PR pure-AIV kernels** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_79_datacopy_with_raw_gm_pointer_silently_returns_zero.md`
- **`SToMTE3Sync()` after V-pipe Cast/Muls/Adds → deterministic garbage on V351 — must use `VToMTE3Sync()`** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_81_stomte3sync_after_v_pipe_cast_muls_adds_determinis.md`
- **V220/CANN 8.5.1 `TPipe::InitBuffer(TQue&, depth)` 2-arg form does not compile — use the 3-arg `(TQue&, depth, buf_size_bytes)` form** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_82_v220_cann_8_5_1_tpipe_initbuffer_tque_depth_2_arg.md`
- **HardSwish 逐元素 kernel 的 TQue 双缓冲实现配方** · 352 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/elementwise_tque_double_buffer_inplace_recipe.md`
- **FP16 double-buffer UB 预算漏乘 2 致 VEC_ERROR** · 214 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/fp16_double_buffer_ub_budget_queue_vs_tbuf_multiplier.md`
- **TilingKey 缓存 correction 致 tilingdata 复用** · 153 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_tilingkey_cache_reuse_per_call_attr_pollution.md`
- **GEIR 算子缓存不感知 .h 修改，掩盖 kernel 修复** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_op_cache_ignores_header_edits.md`
- **kernel 源码改动未生效因 binary 缓存** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_binary_cache_printf_hash_toggle.md`
- **aclnn 手动 Run 须 AbandonCache 防堆损坏** · 174 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/manual_run_abandon_cache_heap_corruption.md`
- **memory-driven stitch 与显式 max_num 冲突致挂起** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/memory_driven_stitch_max_num.md`
- **arch35 MIX：AIC 跑 DataCopyPad 挂起加守卫** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mix_aic_mte_datacopypad_hang_guard.md`
- **探针末尾 MTE2_S/MTE3_S 同步不可由 stream sync 替代** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mte2_s_mte3_s_stream_sync.md`
- **NDDMA非对齐尾轴需NdDmaConfig右填充与DataCopyPad回写** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/nddma_nddmaconfig_datacopypad.md`
- **A3 Memory-based ↔ A5 Register-based API mapping (L2 substitution table) [V351, microapi-l2]** · 161 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_152_a3_memory_based_a5_register_based_api_mapping_l2_s.md`
- **DataCopy `count` form silently truncates non-32B-aligned transfers — never paper over with host-side pad+narrow in pybind [V351+V220, ALL_MODES, data-movement-correctness + anti-cheat]** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_167_datacopy_count_form_silently_truncates_non_32b_ali.md`
- **UB buffer layout must be validated before first compile — one-line offset bug cascades to all downstream buffers** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_215_ub_buffer_layout_offset_chain_validation.md`
- **pybind path CAN use multiple MTE2 TQue — agent must not blame pybind structure** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_266_pybind_path_can_use_multiple_mte2_tque_agent_must.md`
- **PipeBarrier only supports MTE2/V/MTE3 — not PIPE_S** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_35_pipebarrier_only_supports_mte2_v_mte3_not_pipe_s.md`
- **DataCopy count is an ELEMENT count on every arch; byte-count expressions carried from byte-typed upstream sour** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_407_datacopy_count_is_an_element_count_on_every_arch.md`
- **DataCopyParams stride slots are direction-asymmetric and in 32-byte datablocks on dav-m200 (never byte units** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_415_datacopyparams_stride_slots_are_direction_asymme.md`
- **Avoid UB bank conflicts — consecutive VEC instructions should hit different banks** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_50_avoid_ub_bank_conflicts_consecutive_vec_instructio.md`
- **A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the FULL NZ-padded extent (roun** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_545_a_scalar_vec_op_muls_adds_on_an_nz_format_buffer.md`
- **The per-SoC DataCopy impl header IS the alignment/count contract — UB-src has a 32B assert, GM-dst has NONE, and unaligned counts silently round DOWN; never rely on the rounding, quantize counts yours** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_561_the_per_soc_datacopy_impl_header_is_the_alignmen.md`
- **dav-m200: vector-Cast output MUST feed DataCopy through a VECOUT TQue** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_674_dav_m200_vector_cast_output_must_feed_datacopy_t.md`
- **torch copy_-written GM is NOT immediately MTE2-consumable by AI kernels on this stack** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_676_torch_copy_written_gm_is_not_immediately_mte2_co.md`
- **DataCopyPad / DataCopyExtParams are NOT SUPPORTED on dav-m200 (Ascend310P3), BOTH directions** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md`
- **DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_11_datacopy_to_tbuf_veccalc_silent_corruption_on_mult.md`
- **on dav-m200, an MTE2 `DataCopy` (GM→UB) whose BYTE count is not a 32B multiple leaves a stale tail — bytes past the down-rounded quantum are never written; bulk-copy the aligned prefix and scalar-wide** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_127_on_dav_m200_an_mte2_datacopy_gm_ub_whose_byte_co.md`
- **`Adds<int32_t>` buffer-to-buffer (dst ≠ src) silently corrupts output at N=1088** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_13_adds_int32_t_buffer_to_buffer_dst_src_silently_cor.md`
- **`DataCopy(UB, TBuf<TPosition::A1>)` in pure-AIV kernel — silent miscompile → runtime illegal instruction** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_16_datacopy_ub_tbuf_tposition_a1_in_pure_aiv_kernel_s.md`
- **UB aliasing cross-row V→MTE2 hazard — silent data corruption** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_17_ub_aliasing_cross_row_v_mte2_hazard_silent_data_co.md`
- **PB-9 (UB→UB `DataCopy` silent corruption) applies to arch 200 (dav-m200) TOO, not just 220x** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_180_pb_9_ub_ub_datacopy_silent_corruption_applies_to.md`
- **dav-m200 MTE `DataCopy` requires the **UB-side address** to be 32B-aligned (the GM side tolerates 4B granularity) — a tail block anchored at a row END touches UB at an unaligned offset and corrupts, a** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_184_dav_m200_mte_datacopy_requires_the_ub_side_addre.md`
- **`PipeBarrier<PIPE_ALL>()` between TBuf-resident MTE2 load and V compute → silent crash 507015 (V220-confirmed)** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_21_pipebarrier_pipe_all_between_tbuf_resident_mte2_lo.md`
- **MTE2 DataCopy on V220 CANN 9.0.0 has 32-byte (8 fp32 element) transfer limit per destination TBuf [V220]** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_22_mte2_datacopy_on_v220_cann_9_0_0_has_32_byte_8_fp3.md`
- **SIMT DCache shares UB → tiling MUST reserve 40KB or risk silent UB OOB [V351, simt-tiling]** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_32_simt_dcache_shares_ub_tiling_must_reserve_40kb_or.md`
- **`event_t(0)` for cube-internal pipe sync (`MTE1_M` / `M_FIX` / `MTE2_MTE1`) collides with AIC↔AIV CrossCoreSetFlag `FLAG_CANON_DONE` chain in `MIX_AIC_1_2` mode → silent hang [V220 + V351, mixed-mode-** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_35_event_t_0_for_cube_internal_pipe_sync_mte1_m_m_fix.md`
- **[ARCHIVED/DEPRECATED 2026-05-22] `DataCopy` GM->UB with non-zero `srcStride` for BSH->BNSD layout canonicalize on V220 produces wrong output [V220, layout-canonicalize, DataCopy-stride]** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_36_archived_deprecated_2026_05_22_datacopy_gm_ub_with.md`
- **UB-to-UB DataCopy Silent Data Corruption** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_9_ub_to_ub_datacopy_silent_data_corruption.md`
- **Reduce sharedTmpBuffer 须与计算 dtype 一致** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reduce_sharedtmpbuffer_dtype_must_match.md`
- **Round 的 sharedTmpBuffer 须为 uint8_t 视图** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/round_shared_tmp_buffer_uint8_view.md`
- **清 TBE 编译缓存不修复 blockDim 固化** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbe_cache_clear_does_not_fix_baked_blockdim.md`
- **TBE编译期blockDim不受coreNum与缓存影响** · 213 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbe_dynamic_compile_blockdim_ignores_corenum_and_cache.md`
- **TBuf 三参数 InitBuffer 触发 isTQue 静态断言** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbuf_initbuffer_uses_two_arguments.md`
- **宏生成 tiling 结构容量须实测 sizeof** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tcube_tiling_sizeof_probe_buffer_capacity.md`
- **TilingKey 扩展绕不过 aclnn 执行器缓存** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingkey_expansion_cannot_bypass_aclnn_cache.md`
- **tmpBuffer 分配步长≠访问步长触发 UB 越界 341** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tmpbuffer_ub_341.md`

下一页：page-009.md
