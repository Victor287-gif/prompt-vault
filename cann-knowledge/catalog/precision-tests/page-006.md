# precision-tests: 351–420 / 893

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **ARA 向量化 padding 三方案无效须标量回退** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ara_vectorized_padding_fixes_ineffective.md`
- **arch35 bf16 标量 GM 读写丢 nan 位模式** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_bf16_gm_scalar_nan_lost.md`
- **arch35 跨位宽 Cast 按较大位宽读源致错值** · 278 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_cast_cross_width_oob_read.md`
- **arch35 count Compare Select 异步VF 存读 RAW** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_count_compare_select_vf_raw.md`
- **arch35 非末维 S 写 V 读 UB 不一致未解** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_s_pipe_write_v_pipe_read_ub.md`
- **arch35 Select 超64 cmpmask 残缺判别** · 110 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_select_64_cmpmask.md`
- **arch35 Select cmpmask SPR 跨调用泄漏致非确定失败** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_select_cmpmask_spr.md`
- **arch35 Select 异步SIMT同步失效致分支错选** · 149 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_select_simt.md`
- **arch35 Select VSEL_CMPMASK_SPR 每 64 元素复用** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_select_vsel_cmpmask_spr_mask_reuse_count_over_64.md`
- **arch35 sigmoid 大负 z 下 exp 溢出 inf 致 sig=0** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_sigmoid_z_exp_inf_sig_0.md`
- **arch35 UB→GM DataCopyExtParams 拷贝写失效** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_ub_gm_datacopyextparams_write_fail.md`
- **arch35 VSEL_CMPMASK_SPR 标量批跑 mask 污染** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_vsel_cmpmask_spr_mask.md`
- **argmax NaN输出0或2^32：calIndex索引无效** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/argmax_nan_calindex_invalid_index.md`
- **Ascend950 DataCopy 写回 L2 驱逐丢对角线** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascend950_datacopy_l2.md`
- **Ascend950 向量管 FTZ 刷新次正规 exp 为零** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascend950_ftz_exp.md`
- **Ascend950 ReduceSum AR 多行归约逐行错** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascend950_reduce_sum_ar_multibundle_use_per_row.md`
- **AscendC::Abs 规范化 NaN payload 致逐位验收误报** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascendc_abs_nan_payload.md`
- **AscendC Muls cyclic-repeat 致分组 scale 错配** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascendc_muls_cyclic_repeat_scale.md`
- **at::zeros 异步零初始化跨 stream 覆盖 kernel 写出** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/at_zeros_stream_kernel.md`
- **b16常量零填充StoreAlign无需DIST_PACK_B32** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/b16_duplicate_storealign_plain_vs_cast_dist_pack_b32.md`
- **B16 交错 DINTLV 成对归零 — 转 fp32 后重排** · 239 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/b16_interleave_compute_should_promote_to_b32_layout.md`
- **特定 b16 load-cast-store 组合失真时在 LocalTensor 边界转换** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/b16_reg_cast_layout_should_convert_at_localtensor_boundary.md`
- **B16 结果大幅偏差 — UNPACK 后补 Reg Cast** · 352 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/b16_unpack_requires_numeric_cast_before_fp32_compute.md`
- **B16 比较先升到 B32 锁定索引 lane** · 252 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/b16_value_and_b32_index_masks_need_common_b32_compute.md`
- **半精度索引错位—拆分双 int32 掩码** · 545 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/b16_value_mask_must_split_for_two_int32_index_registers.md`
- **bf16 扩精度 Cast 误用 CAST_RINT 产垃圾值判别** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_cast_cast_rint.md`
- **bf16 极值输入 ε=0 NaN/Inf 语义分歧未解** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_extreme_input_nan_inf_divergence_unresolved.md`
- **bf16 边缘失败用 fair-oracle 探针判 oracle 不公** · 178 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_fair_oracle_oracle.md`
- **bf16全量程随机生成float32溢出改double域收窄** · 243 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_full_range_random_gen_float32_overflow.md`
- **BF16 gamma Cast 分阶段编排导致随机 NaN** · 260 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_gamma_cast_must_share_ordered_compute_stage.md`
- **ST造BF16输入误取fp32低16位致MERE=0.149** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_host_conversion_low_16bits.md`
- **bf16 多 matmul 链累积漂移判别** · 149 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_matmul.md`
- **bf16 量化极限的 MARE 略超阈三签名归类 — 阈值契约不可改时 safety_fallback 上报** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_quantization_limit_three_signature_safety_fallback.md`
- **bias 双路拷出偏移未 clamp，负偏移 uint32 回绕** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bias_dual_path_copyout_offset_clamp_uint32_wraparound.md`
- **bias 噪声证据失真：write_evidence 复用 input 真值** · 283 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bias_evidence_json_cross_output_fp64_truth_reuse.md`
- **ST bias mare 预算噪声自适应 max(10×th,10×noise)** · 427 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bias_reduction_mare_noise_adaptive_budget.md`
- **b16 running 经 fp32 回转会改写 NaN payload** · 231 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bit_exact_b16_nan_payload_needs_original_dtype_running_value.md`
- **黑盒 bf16-only 精度失败：dtype 分层判别** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/blackbox_dtype_stratified_precision_triage.md`
- **303 D2 末轴分块破坏 no-Cast DAG x2 加载** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/broadcast_303_d2_last_axis_no_cast_dag_x2_missing.md`
- **双边广播输出错乱：合轴 stride 误用输出维度累乘** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/broadcast_collapse_stride_must_accumulate_per_input.md`
- **int8 量化尾段 Cast 走固定 half 两步中转** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/cast_int8_output_two_step_half_pivot.md`
- **Cast SatMode::SAT 饱和致 fp16 NaN/Inf 丢失** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/cast_satmode_sat_fp16_nan_inf.md`
- **多核 ceil 尾块为负 — 每核按 remaining 截断** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ceil_partition_each_core_must_clamp_remaining.md`
- **Chebyshev 系数与 Clenshaw 约定不符输出小 4 个量级** · 354 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/chebyshev_coefficients_clenshaw_convention_mismatch.md`
- **除零兜底 cof=0.0f 致 var=0 而非 NaN** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/cof_0_0f_var_0_nan.md`
- **比较器重写后 max 分母公式回退复现失败 — 公式回退当回归 + 防回退用例** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/comparator_max_denominator_formula_regression.md`
- **比对阈值随 golden dtype 漂移：按算子输出 dtype 取阈** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/compare_threshold_bound_to_output_dtype.md`
- **b16 提升 fp32 比较后按存储宽度分块会漏掉后半 lane** · 145 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/compute_lane_count_must_follow_promoted_fp32_width.md`
- **卷积环形 lookback 写早于读覆盖历史状态** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/conv_ring_lookback_write_before_read_overwrites_history.md`
- **广播 CopyIn tile 跨段越界读判别** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/copyin_tile.md`
- **CopyOut 改 blockCount 批量搬运回归回退** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/copyout_blockcount_batch_regression.md`
- **count超4096仅前4288元算正确的精度阈值** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/count_4096_4288.md`
- **Max/Min 逐元素非标量广播 须 Duplicate 至同长 buffer** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/count_based_max_min_is_element_wise_not_scalar_broadcast.md`
- **V220 hand-written bare-mmad cube paths silently miscalculate on A5 (L0A transpose misplacement at blockM>=2) — route-guard them off and keep MatmulImpl (PB-43 cross-gen instance)** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/cube_manual_mmad_crossgen_001.md`
- **稀疏反卷积精度误差源隔离配方** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/cube_simt_fp32_precision_error_source_isolation.md`
- **DataCopy 非32字节对齐尾块丢字节** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopy_32.md`
- **DataCopy GM→UB 需 32B 对齐，非对齐 C 统计输出全零** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopy_gm_to_ub_non_32b_aligned_silent_zero.md`
- **DataCopy 不自动按 tile 推进须显式元素 offset** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopy_setglobalbuffer_no_auto_advance_tileoffset.md`
- **DataCopyPad padding须按输入dtype对齐** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_input_dtype_align_507035.md`
- **窄尾 tile 行距错位：补齐 UB gap** · 212 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_narrow_tile_needs_explicit_ub_row_gap.md`
- **DataCopyPad 尾块改逐行一维搬入** · 291 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_partial_2d_blocks_use_per_row_copy.md`
- **合并尾块的非 32B 行导致 PathB 数据损坏** · 271 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_rows_must_keep_32_byte_block_length.md`
- **DataCopyPad多行stride单位误用导致非对齐行输出NaN** · 373 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_stride_fields_use_32_byte_block_units.md`
- **DataCopyPad stride 表示 block 间隙** · 241 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_stride_is_inter_block_gap.md`
- **逐元素 DataCopyPad 做 strided gather 输出全零** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_strided_gather.md`
- **DataCopyPad 未填满强制对齐缓冲 — 小维规约读入脏尾部** · 201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_tail_must_initialize_full_reduction_extent.md`
- **门梯度整体错位 — CopyOut srcStride 误用行距** · 188 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_unified_segment_burst_srcstride_is_segment_gap.md`
- **DataCopy blockLen 以 32B 块计非字节** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopyparams_block_len_32b_block_units.md`
- **Ascend950 Reg 扫描按 256B 锁定 lane** · 139 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/dav3510_vector_register_width_is_256_bytes.md`
- **DeInterleave 两路写反——按偶索引到 dst0 修正** · 233 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/deinterleave_dst0_is_even_lane_output.md`

下一页：page-007.md
