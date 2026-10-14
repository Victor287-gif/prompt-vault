# reduction-broadcast: 771–840 / 1263

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Float-AR ArgMax：ReduceMax calIndex 分块实现** · 366 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/argmax_float_ar_reduce_max_calindex_recipe.md`
- **dav-3510 混合 dtype 广播算子多核 kernel 骨架配方** · 485 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/broadcast_mixed_dtype_multicore_skeleton.md`
- **ChunkedR 在线两趟 softmax kernel 实现** · 380 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/chunked_r_online_two_pass_softmax_kernel.md`
- **ChunkedR ARA softmax 三趟 kernel 实现配方** · 523 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/chunkedr_ara_softmax_three_pass_recipe.md`
- **Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_39_cube_matmulimpl_mm_cfg_cfg_norm_rejected_matmulcon.md`
- **`Duplicate(dst, src, count)` broadcasts a 32B BLOCK (8 fp32 / 16 fp16), NOT a single scalar — compiles clean but produces garbage output** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_80_duplicate_dst_src_count_broadcasts_a_32b_block_8_f.md`
- **1D 拷贝循环替代 NDDMA 随路广播 CopyIn 失败** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/errcode_263_82_nddma_broadcast_1d_loop_banned.md`
- **postReduceUbSize 不足致 Phase2 读破坏** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/euclidean_norm_group_post_reduce_ub_size_phase2_chunk.md`
- **FullloadR softmax kernel AR/ARA 实现配方** · 586 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/fullloadr_softmax_ar_ara_kernel_recipe.md`
- **L0 op 须算归约后 shape（非复用输入）** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/l0op_stub_must_derive_reduce_shape_not_input.md`
- **FA-class FORWARD operators with softmax/attention require codified AscendC tile templates, not a naive single-spawn rewrite [V220; arch22→arch35 scope narrowed by OL-185]** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_159_fa_class_forward_operators_with_softmax_attention.md`
- **High-level reduce/broadcast primitives run on arch versions absent from their public arch-guard — they fall through to the common implementation; verify by compiling, don't avoid them on an "unlisted"** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_230_high_level_reduce_broadcast_primitives_run_on_arch.md`
- **UB budget gate for D×hwNum normalization/elementwise ops (error 271)** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_265_ub_budget_gate_d_hwnum_normalization.md`
- **host-tiling-bound high-level APIs (SoftmaxGrad, any primitive whose Compute path needs a host TilingFunc) are** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_414_host_tiling_bound_high_level_apis_softmaxgrad_an.md`
- **on dav-m200 WholeReduceSum(dst,src,mask,repeatTime,dstRepStride,srcBlkStride,srcRepStride)'s int32 mask is an** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_416_on_dav_m200_wholereducesum_dst_src_mask_repeatti.md`
- **vendor torch.add broadcast is NON-DETERMINISTIC on dav-m200** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_122_vendor_torch_add_broadcast_is_non_deterministic.md`
- **dav-m200 `WholeReduceMax` `dstRepStride` unit is a PAIR (value,idx,pad,pad) while `WholeReduceSum`'s is an ELEMENT — the two Level-2 WholeReduce APIs do NOT share a stride-unit convention and neither ** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_183_dav_m200_wholereducemax_dstrepstride_unit_is_a_p.md`
- **torch_npu `.contiguous()` / `.copy_()` of a transpose-to-INNERMOST permute HANGS (wedges) the device — a permute that keeps the last dim is fine** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_46_torch_npu_contiguous_copy_of_a_transpose_to_innerm.md`
- **Reduce sharedTmpBuffer 须与计算 dtype 一致** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reduce_sharedtmpbuffer_dtype_must_match.md`
- **Reduce work buf 额外占 UB 致 507035** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reduce_work_buf_ub_507035.md`
- **RmsNorm无bf16支持须手动组合实现** · 167 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rmsnorm_bf16_high_api_gap_manual_impl.md`
- **RNN 门控算子纯 AIV 路线：Mul+ReduceSum 矩阵乘** · 391 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rnn_gated_pure_aiv_mul_reducesum_route.md`
- **Softmax 手工尾填充会触发 Vector 异常** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/softmax_manual_tail_fill_route_vector_exception.md`
- **二进制数少于TilingKey组合数不等于缺实例** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tpl_binary_count_less_than_combos_normal.md`
- **ACL 507035 — 完整 a0 统计缓冲超过 UB** · 268 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ub_slices_reduction_axis_scratch_buffers_507035.md`
- **B=1 批量化公式与 GM 布局互为转置不可直搬** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/b1_batch_layout_transpose_mismatch.md`
- **301 非连续多核启用：preferMultiCore 守护** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/broadcast_301_nlast_transpose_multicore_reverse_search.md`
- **Hardware reduce intrinsics collapse instruction count, not element work — on dav-m200 the VEC pipe is element-throughput-bound, so swapping manual reduction trees for `vcadd`/`vcmax`/`vcgadd`/`vcgmax`** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_682_hardware_reduce_intrinsics_collapse_instruction.md`
- **Cost analysis of norm + activation fusion** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_69_cost_analysis_of_norm_activation_fusion.md`
- **每核最少行数地板压低 AIV 并行度—按可用核数切分** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/row_reduction_min_rows_floor_underutilizes_aiv.md`
- **950 循环内 ReduceMin 取索引致 vector core 异常** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/950_reducemin_vector_core.md`
- **FP16 NaN 用例丢 NaN：ACL fp16 转换位级规避** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/acl_fp16_conversion_nan_lost.md`
- **AR 归约读未初始化 pad 行 NaN 污染** · 165 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ar_reduce_pad_rows_identity_prefill.md`
- **Ascend950 ReduceSum AR 多行归约逐行错** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascend950_reduce_sum_ar_multibundle_use_per_row.md`
- **ST造BF16输入误取fp32低16位致MERE=0.149** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_host_conversion_low_16bits.md`
- **ST bias mare 预算噪声自适应 max(10×th,10×noise)** · 427 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bias_reduction_mare_noise_adaptive_budget.md`
- **303 D2 末轴分块破坏 no-Cast DAG x2 加载** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/broadcast_303_d2_last_axis_no_cast_dag_x2_missing.md`
- **双边广播输出错乱：合轴 stride 误用输出维度累乘** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/broadcast_collapse_stride_must_accumulate_per_input.md`
- **Max/Min 逐元素非标量广播 须 Duplicate 至同长 buffer** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/count_based_max_min_is_element_wise_not_scalar_broadcast.md`
- **DataCopyPad 未填满强制对齐缓冲 — 小维规约读入脏尾部** · 201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_tail_must_initialize_full_reduction_extent.md`
- **Exp 默认 INTRINSIC FTZ 致 subnormal 残留失配** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/exp_intrinsic_ftz_subnormal.md`
- **bias 归约 NaN/Inf 不匹配：±FLT_MAX 输入溢出非精度缺陷** · 261 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/extreme_range_bias_reduction_nan_inf_sum_order_mismatch.md`
- **flash在线softmax JIT重排致精度超阈** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/flash_softmax_jit.md`
- **FloatToFp16 denormal 路径截断致 1-ULP 偏差** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/floattofp16_denormal_truncate_1ulp.md`
- **FloatToHalf 次正规阈值漏 exp102 致 mare=0.373** · 244 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/floattohalf_subnormal_threshold_exp102.md`
- **fp16 次正规 1-ULP 差被 mare 放大判别** · 398 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_subnormal_1ulp_mare_amplification_diagnosis.md`
- **fp16小值域MARE超标：golden未模拟存储精度** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_subnormal_golden_simulate_storage.md`
- **fp32 常规值域 mare 跨分支同构未解** · 182 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp32_normal_range_mare_shared_chain_unresolved.md`
- **golden 归约顺序与 kernel 不一致放大 near-zero MARE** · 167 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/golden_reduce_order_near_zero_mare.md`
- **F.group_norm单元素组报错，golden改手工双遍** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/group_norm_single_element_group_manual_golden.md`
- **fp32 GroupTailA mere 随归约长度 R 增长越阈** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/group_reduce_mere_grows_with_r_signature.md`
- **GroupNorm 归一化输出近零元素 — MARE/MERE 分离判别、混合分母与 torch 工程完整比对器** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/groupnorm_near_zero_hybrid_denominator_torch_compare.md`
- **int32 fp32累加reduce binary_equal不可达** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int32_fp32_reduce_binary_equal.md`
- **scatter 归约逐次回写原生 dtype 放大舍入误差** · 450 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/low_precision_scatter_reduction_needs_fp32_accumulator.md`
- **bf16算子MERE超阈但失配占比极低的归约顺序指纹** · 186 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mere_over_threshold_tiny_bitwise_mismatch_reduction_order_fingerprint.md`
- **Reduction-tree fail-floor invariant — swapping per-stage reduction shape only moves which cases fail, not the total** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_110_reduction_tree_fail_floor_invariant.md`
- **A5 Subnormal handling via `LnConfig{algo}` for Ln/Sqrt/Rsqrt/Div/Reciprocal/Exp [V351, precision-config]** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_147_a5_subnormal_handling_via_lnconfig_algo_for_ln_sqr.md`
- **Mirror Phase O5 verifier plugin-skip and tier-normalization logic across the local-container and SSH paths — unmirrored divergence falsely rejects correct kernels at finalize** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_261_mirror_phase_o5_verifier_plugin_skip_and_tier_norm.md`
- **aten CPU-fallback scatter-reduce chain table on 310p: fp16/bf16 are fp32-shadow + ONE final RNE round (add AND multiply); fp32 is exact native; reduce=None is ascending last-wins** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_671_aten_cpu_fallback_scatter_reduce_chain_table_on.md`
- **Reg 归约结果直接 Duplicate 导致末轴 Softmax 偏差** · 277 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reduce_lane0_must_be_broadcast_before_vector_normalization.md`
- **ReduceSum复用源改写computeBuf致脏读** · 186 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reducesum_isreusesource_corrupts_source_buffer.md`
- **ReduceSum RA(c310)累加序：16宽平衡寄存器树** · 337 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reducesum_ra_pattern_tree_order_bitwise.md`
- **Reduce count 必须排除 DMA 对齐填充元素** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reduction_count_excludes_dma_padding.md`
- **Reg::Reduce 跨 chunk 覆写导致规约失真** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reg_reduce_per_chunk_result_must_accumulate_explicitly.md`
- **A3 RMSNorm 不要用分母正值保护吞掉 NaN 传播** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rmsnorm_nan_propagation_not_positive_guard.md`
- **AR 输出张量未初始化导致 Softmax 部分用例错** · 237 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_ar_output_buffer_must_be_initialized_from_input.md`
- **softmax 特殊值切片语义：含 +inf 全 NaN 非 one-hot** · 153 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_special_value_slice_semantics.md`
- **SoftMax srcK须等于缓冲区行步长** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_srck_row_stride.md`
- **SoftmaxV2 ARA与AR模板R向累加树不同致1ULP分歧** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmaxv2_ara_ar_r_1ulp.md`
- **tiling 按位置奇偶判 reduce 轴致非交替 axes 误减** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tiling_reduce_axes.md`

下一页：page-013.md
