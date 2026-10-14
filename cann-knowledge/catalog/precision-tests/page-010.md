# precision-tests: 631–700 / 893

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Reduce count 必须排除 DMA 对齐填充元素** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reduction_count_excludes_dma_padding.md`
- **Reg API int8 DIST_PACK_B32 交错零** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reg_api_int8_dist_pack_b32.md`
- **Reg::Reduce 跨 chunk 覆写导致规约失真** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reg_reduce_per_chunk_result_must_accumulate_explicitly.md`
- **Reg 尾块 masked Add 清零跨块累加器** · 257 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reg_tail_mask_zeroing_clears_cross_chunk_accumulator.md`
- **RegTensor 跨 VF 内循环累加失效 — 状态改存 UB 槽** · 306 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/regtensor_cross_chunk_state_should_persist_in_ub_slot.md`
- **FreeTensor早释使队列别名被下一轮DMA覆盖** · 556 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reinterpret_tensor_alias_must_outlive_queue_release.md`
- **重复目标随机写丢失 — 在 UB 内按序合并后 DMA 写回** · 235 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/repeated_overwrite_updates_need_ordered_ub_merge.md`
- **A3 RMSNorm 不要用分母正值保护吞掉 NaN 传播** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rmsnorm_nan_propagation_not_positive_guard.md`
- **递归算子ST权重尺度须随fan-in缩放** · 139 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rnn_st_weight_scale_fan_in_xavier.md`
- **RoPE DIST_INTLV_B16 store mask 致零值** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rope_dist_intlv_b16_store_mask.md`
- **RoPE interleaved链白盒精度劣于half链** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rope_interleaved_whitebox_precision_regression.md`
- **dav-2201 Rsqrt 精度不足 — 改用 Sqrt 加标量倒数** · 172 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rsqrt_fast_approximation_use_sqrt_reciprocal.md`
- **GetBool 返回指针，隐式转 bool 致 hasInverse 恒 1** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/runtime_attrs_getbool_pointer_implicit_bool.md`
- **单标量输出向量链丢值：回读标量侧再写** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scalar_output_v_mte3_loss_readback.md`
- **标量 S pipe 误用 TQue 导致输出全常数** · 308 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scalar_s_pipe_requires_explicit_mte_dependencies.md`
- **Scatter 非散射维可缩小，更新量须按 indices 计算** · 258 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scatter_update_count_must_follow_indices_shape.md`
- **Select cmpmask 不偏移致分支反转诊断** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/select_cmpmask.md`
- **Select CMPMASK_SPR 模式致混合分支误选** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/select_cmpmask_spr.md`
- **Select dst==src 别名破坏张量输入值** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/select_dst_src.md`
- **Select张量-标量模式索引不更新，须TT模式** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/select_tensor_scalar_mode_tensor_operand_no_update.md`
- **Select VSEL_CMPMASK_SPR count>64 掩码错位判别** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/select_vsel_cmpmask_spr_count_64.md`
- **共享 UB 跨流水线复用须按方向显式同步** · 352 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/shared_ub_cross_pipeline_per_direction_sync.md`
- **AR 跨块循环放在 __simd_vf__ 只执行首块** · 389 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/simd_vf_chunk_loop_must_be_driven_by_aicore_wrapper.md`
- **__simd_vf__多rep截断致大D尾部为零** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/simd_vf_multi_rep_truncation_large_d.md`
- **perm 化简 size-1 强制 identity 破坏不变量误路由** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/simplify_perm_size1_identity_invariant.md`
- **SIMT 切分未 coalesce 致 block 边界错位** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/simt_coalesce_block.md`
- **SIMT VF 栈数组动态索引致输出全零** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/simt_vf.md`
- **骨架占位输出全 0：标量直算先打通正确性** · 244 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/skeleton_placeholder_output_zero_scalar_conv.md`
- **AR 输出张量未初始化导致 Softmax 部分用例错** · 237 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_ar_output_buffer_must_be_initialized_from_input.md`
- **softmax 特殊值切片语义：含 +inf 全 NaN 非 one-hot** · 153 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_special_value_slice_semantics.md`
- **SoftMax srcK须等于缓冲区行步长** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_srck_row_stride.md`
- **SoftmaxV2 ARA与AR模板R向累加树不同致1ULP分歧** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmaxv2_ara_ar_r_1ulp.md`
- **softplus_v2_grad 极端 FLT_MAX 与 aten 不可对齐** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softplus_v2_grad_flt_max_aten.md`
- **softplus_v2_grad 极值 inf 发散源于 aten 乘除顺序** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softplus_v2_grad_inf_aten.md`
- **SoftplusV2Grad 非默认θ线性区分支缺失致超差** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softplusv2grad.md`
- **SoftplusV2Grad VF Form B exp 溢出判别** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softplusv2grad_vf_form_b_exp.md`
- **专用 kernel 路径输出不变时先核对共享库是否真的包含目标符号** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/specialized_npu_kernel_symbol_presence_probe.md`
- **Reg 同 wrapper 回读 UB 引发尺寸相关漂移** · 524 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/split_multi_pass_vf_at_ub_producer_consumer_boundaries.md`
- **ST 连续运行失败而独立 repro 全对的二分** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/st_continuous_run_fail_independent_repro_pass.md`
- **红基线 mere~1e33＝poison 保持：stub 未写输出** · 234 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/st_poison_red_baseline_unwritten_output_signature.md`
- **PyTorch 移植 ST 随机流不一致：逐位移植 RNG** · 206 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/st_port_rng_stream_mismatch.md`
- **近零 golden 相对误差 — 分母改用 max(|g|, th)** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/stat_rel_err_near_zero_floor_denominator.md`
- **算子属性默认值 stringstream 截断致 FP32 标量误差** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/stringstream_fp32.md`
- **SwiGLU 高级 API 在非32B对齐 slice 上精度失败** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/swiglu_adv_api_fails_non_32b_aligned_slice.md`
- **SyncAll 修复尾元素丢写无效反致误差恶化** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/syncall_not_fix_scalar_lost_write.md`
- **裸 TBuf 跨 pipe 依赖需 PipeBarrier<PIPE_ALL>** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tbuf_cross_pipe_dependency_requires_pipe_all.md`
- **TBuf 跨 MTE2 与 Vector 须队列交接** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tbuf_mte2_to_vector_requires_queue_handoff.md`
- **同一 TBuf 两次 Get 是别名：先算后覆盖** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tbuf_repeated_get_alias_reorder.md`
- **模板内 float→float Cast 使 fp32 分支全错** · 359 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/templated_float_input_must_bypass_same_type_cast.md`
- **TensorList 非连续支持两条路线均失败已降级** · 205 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tensorlist_noncontiguous_two_routes_fail.md`
- **TK1 偏移未随 M/K 输出序语义同步致下标互换** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tiling_mk_semantics_swap_offsets.md`
- **tiling 按位置奇偶判 reduce 轴致非交替 axes 误减** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tiling_reduce_axes.md`
- **标量属性入 TilingData 致同 shape 跨调用陈旧** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tilingdata_shape.md`
- **TilingFunc 未写 TilingData 致输出全 0** · 247 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tilingfunc_stub_not_writing_tilingdata_zero_output.md`
- **TopK BF16 tie 索引顺序与 CPU 不一致** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/topk_bf16_tie_cpu.md`
- **k==E 输出垃圾值：SoftMax buffer 污染未解** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/topk_k_eq_e_output_garbage_unresolved.md`
- **torch.topk 最小值 NaN 不排前：golden 需对齐** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/topk_nan_largest_false_golden_alignment.md`
- **TopK padding 区须 SetValue 填极值** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/topk_padding_extreme_value_fill.md`
- **TopK输出行步长kAlign与kPad不一致** · 172 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/topk_row_stride_misalignment_507035.md`
- **numpy复刻torch sum序：16行块级联bitwise等golden** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_cpu_sum_dim0_16row_block_cascade_bitwise.md`
- **torch.cummin平局索引为last-occurrence** · 218 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_cummin_tie_last_occurrence.md`
- **torch fp32 sum(dim=0)逐列序：大N稳定为casc16级联** · 179 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_fp32_sum_dim0_casc16_column_order.md`
- **fp16 1-ULP失配：torch多维softmax不升fp32** · 185 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_multidim_fp16_softmax_no_internal_upcast.md`
- **半精度 shadow 跨流竞态 — 转换与 kernel 两侧做设备同步** · 384 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_npu_async_temporaries_need_completion_before_raw_kernel.md`
- **torch.unique ±0 符号不稳定致 bit-exact 失败** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_unique_negzero_sign_normalize.md`
- **TTK bf16阈值键破碎回退fp32档2^-13** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ttk_bf16_dtype_key_broken_threshold_fallback.md`
- **expect_error 用例被判 FAIL：host 拒收非缺陷** · 224 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ttk_expect_error_negative_case_invalid_tiling_fail.md`
- **TTK stat_rel_err 默认阈值 fp32 2^-13** · 219 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ttk_stat_rel_err_default_threshold_table.md`
- **TTK 输入无固定 seed，同用例两跑 mare 可不同** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ttk_unseeded_random_input_mare_instance_dependence.md`
- **2 字节逐元素 GM RMW 大规模丢写判别（mare≈1）** · 214 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/two_byte_gm_rmw_large_scale_lost_write.md`

下一页：page-011.md
