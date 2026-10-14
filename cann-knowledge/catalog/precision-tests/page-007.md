# precision-tests: 421–490 / 893

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Depthwise 卷积小值域假失败 — 期望==实际打印一致指纹、阈值 max(表值,10×th) 与示例侧 CompareResults 双示例落地** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/depthwise_small_value_print_identical_example_compare.md`
- **MTE2跨流水未建依赖导致UB读到上次调用残值** · 282 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/direct_mte_slot_reuse_needs_bidirectional_hard_events.md`
- **2^-30 是双标杆比值标准阈值 — 单标杆保守用法对 fp32 舍入差过严的三前提 fallback** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/dual_benchmark_threshold_single_benchmark_fallback.md`
- **Duplicate 全量清零 padding 覆盖有效区 banned** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/duplicate_padding_banned.md`
- **EMA 公式 stat+m*(batch-stat) 在 inf 时 NaN** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ema_stat_m_batch_stat_inf_nan.md`
- **空batch归约：bias输出为(3H,)精确零非空张量** · 192 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/empty_batch_bias_output_zero_fill_not_empty_tensor.md`
- **exp 下界 clamp -50 破坏 sigmoid 大负数下溢** · 80 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/exp_clamp_50_sigmoid.md`
- **exp 输入 clamp 上界用 threshold 用户参数致溢出** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/exp_clamp_upper_bound_must_decouple_from_user_threshold.md`
- **Exp 默认 INTRINSIC FTZ 致 subnormal 残留失配** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/exp_intrinsic_ftz_subnormal.md`
- **exponential_ 0% 匹配须改 -log(1-u) 路径** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/exponential_0_log_1_u.md`
- **bias 归约 NaN/Inf 不匹配：±FLT_MAX 输入溢出非精度缺陷** · 261 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/extreme_range_bias_reduction_nan_inf_sum_order_mismatch.md`
- **FindSplitAxis 误传 total_slots 致 slot 翻倍** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/findsplitaxis_total_slots_slot.md`
- **Reg 固定地址跨块累加需逐 VF PipeBarrier** · 269 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fixed_address_vf_accumulator_needs_per_call_barrier.md`
- **flash在线softmax JIT重排致精度超阈** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/flash_softmax_jit.md`
- **base=1.000001f 落入 |base-1|<1e-6 容差分支** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/float_ulp_tolerance_branch_routing.md`
- **FloatToFp16 denormal 路径截断致 1-ULP 偏差** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/floattofp16_denormal_truncate_1ulp.md`
- **FloatToHalf 次正规阈值漏 exp102 致 mare=0.373** · 244 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/floattohalf_subnormal_threshold_exp102.md`
- **FP16 算子 Cast 提升-计算-回退链设计** · 463 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_cast_up_fp32_compute_cast_down_chain.md`
- **FP16 half 就地覆写 x1 破坏 y_second 读依赖** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_half_x1_y_second.md`
- **fp16大E近阈值MERE临界超出无修复** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_large_e_threshold_margin_unresolved.md`
- **fp16 大累加项数跨实现 MERE 越阈边界** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_large_term_accumulation_cross_golden_boundary.md`
- **FP16多步计算链误差累积超单步阈值：按步数×ULP预估** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_multistep_accumulated_error_budget.md`
- **fp16 小值输出 MARE 假失败的两条边界 — 元素豁免 allAbsOk（正）与大 K 事后放宽（反例红线）** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_small_value_mare_exemption_posthoc_loosen.md`
- **fp16 次正规 1-ULP 差被 mare 放大判别** · 398 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_subnormal_1ulp_mare_amplification_diagnosis.md`
- **fp16小值域MARE超标：golden未模拟存储精度** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_subnormal_golden_simulate_storage.md`
- **fp32-vs-double 机器精度差判真实超标 — 放宽 epsilon 属 banned 掩盖** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_double_real_exceed_banned_epsilon_loosen.md`
- **大归约维 fp32 golden 顺序累加致假精度失败** · 205 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_golden.md`
- **fp32→int32越界NPU饱和与numpy不一致** · 158 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_int32_cast_overflow_saturate_vs_indefinite.md`
- **fp32 kernel 对 double golden 的近零豁免界须按精度模型推导 — eps 下限、abs_floor 校准与 mare_raw 双向验证** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_kernel_double_golden_near_zero_floor_derivation.md`
- **fp32 常规值域 mare 跨分支同构未解** · 182 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_normal_range_mare_shared_chain_unresolved.md`
- **fp32 乘积相消 MARE 超阈值 — 用补偿乘积保留低位** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_rope_product_cancellation_two_product.md`
- **fp32 窄化舍入模式不一致触发稀疏 MARE 越线** · 297 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_to_fp16_cast_matches_rne_oracle.md`
- **fp32算子bit-exact仍败于fp64 golden比对** · 276 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp64_golden_fp32_cancellation_dtype_limit.md`
- **共享权重buffer被覆写致l≥1误用wc** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fused_kernel_shared_weight_ub_buffer_later_load_overwrites.md`
- **门连续存储遇 H8 步长读取错位改分段** · 193 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gate_segment_layout_h8_stride_mismatch.md`
- **gather 扁平偏移公式 dim 上方形状错配** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gather_dim.md`
- **多维 Gather 部分 case 错 — LUT 维序反了** · 220 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gather_lut_coordinate_decode_must_start_from_last_dimension.md`
- **Gather 非 dim 维缩小时折叠步长错位** · 617 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gather_non_axis_shape_shrink_requires_per_axis_source_strides.md`
- **整数gcd golden：INT_MIN回绕与无符号欧几里得** · 189 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gcd_int_min_wraparound_unsigned_euclid.md`
- **GEIR bf16 mere~1e11：近零误诊，量级比判别** · 273 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/geir_bf16_mere_explosion_engine_path_divergence.md`
- **GEIR 图模式 int dtype 用例静默逐位失配** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/geir_int_dtype_silent_bitwise_mismatch.md`
- **AscendC GELU 三系数 erfc 在大值域 fp32 精度不足** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gelu_erfc_polynomial_precision_budget.md`
- **GELU 负尾部用 erfc 形态避免大数相消** · 203 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gelu_negative_tail_erfc_avoids_cancellation.md`
- **GlobalTensor::SetValue 写丢失，改 DataCopyPad** · 277 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/globaltensor_setvalue_gm_tail_write_lost.md`
- **GM SetValue 写回后 GetValue 读到旧值** · 171 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gm_setvalue_getvalue_reads_stale_within_kernel.md`
- **golden BF16 rstd 归约误取 post-BF16 量化值** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_bf16_rstd_post_bf16.md`
- **golden 全量物化 scores OOM 改分块 gather** · 255 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_dense_scores_oom_chunked_gather.md`
- **golden 参考精度不足：fp32 中间计算升 fp64** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_intermediate_precision_above_kernel.md`
- **golden interpolate需显式recompute=True** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_interpolate_recompute_scale_factor.md`
- **golden三目求max对±0返回-0，需双零特判** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_max_negative_zero_signbit.md`
- **golden 未量化到输出 dtype，denormal 区误报** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_quantize_to_output_dtype.md`
- **golden 归约顺序与 kernel 不一致放大 near-zero MARE** · 167 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_reduce_order_near_zero_mare.md`
- **GridSampler 混合符号输入相消致近零输出 MARE 假失败 — 数据侧改非负输入 [0,1]** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/grid_sampler_mixed_sign_cancellation_nonnegative_input.md`
- **F.group_norm单元素组报错，golden改手工双遍** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/group_norm_single_element_group_manual_golden.md`
- **Group Phase2读y别名workspace与CopyOut竞态** · 223 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/group_phase2_workspace_alias_y_race.md`
- **fp32 GroupTailA mere 随归约长度 R 增长越阈** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/group_reduce_mere_grows_with_r_signature.md`
- **GroupNorm 归一化输出近零元素 — MARE/MERE 分离判别、混合分母与 torch 工程完整比对器** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/groupnorm_near_zero_hybrid_denominator_torch_compare.md`
- **gtsv2 回代缺 du2 第二上对角线 fill-in 致主元交换用例全错** · 239 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/gtsv2_backward_subst_missing_du2_fill_in.md`
- **手写 fp32→fp16/bf16 舍入未实现 RNE 偏 1 ulp** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/handwritten_float_cast_rne_roundbit_bias.md`
- **HiFloat8 SatMode::SAT Cast NaN 映射 0x00** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/hifloat8_satmode_sat_cast_nan_0x00.md`
- **偏置校正 β^t 用 powf 单精度与 golden double 漂移** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/host_powf_beta_pow_double_align.md`
- **Host 缺 SetDataSize 致 TilingData 截断** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/host_setdatasize_tilingdata.md`
- **InferShape 返回 identity 致空 kernel 假通过** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/infershape_identity_kernel.md`
- **原地Vector化MIX kernel输出全0** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/inplace_vectorization_mix_kernel_all_zero.md`
- **int32 fp32累加reduce binary_equal不可达** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int32_fp32_reduce_binary_equal.md`
- **int32经fp32中转丢精度，超2^24须原生比较** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int32_fp32_roundtrip_beyond_2p24_native_compare.md`
- **Reg int32→int64 Cast 源目 VL 错配留空索引** · 206 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int32_to_int64_reg_cast_must_respect_destination_lane_width.md`
- **int64+AR 动态 shape aclnn 直调跳过自定义 tiling** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int64_ar_dynamic_shape_aclnn_skip_tiling.md`
- **硬标签target固定按int32步长读取致精度全错** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int64_hard_label_int32_stride_misread.md`
- **Reg::Interleave 自定义重组 D=128 行两半互换** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/interleave_vf_full_register_row_half_swap.md`

下一页：page-008.md
