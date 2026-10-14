# other-operator-knowledge: 1121–1190 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Fan-In Threshold for Sort Decision (CANN-derived)** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p39_fan_in_threshold_for_sort_decision_cann_derived.md`
- **vec4 vectorization path enablement condition** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p3_vec4_vectorization_path_enablement_condition.md`
- **Sort Algorithm Selection Decision Tree** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p43_sort_algorithm_selection_decision_tree.md`
- **Radix Sort Twiddle Transform for Float/Signed Types** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p44_radix_sort_twiddle_transform_for_float_signed_ty.md`
- **Quantize Cast Chain (fp32 → int8)** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p46_quantize_cast_chain_fp32_int8.md`
- **Per-Block Private Histogram in UB (atomicAdd Reduction)** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p48_per_block_private_histogram_in_ub_atomicadd_redu.md`
- **Dynamic block size** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p4_dynamic_block_size.md`
- **Lift `nblk` to runtime AIV count for V220 portability** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p51_lift_nblk_to_runtime_aiv_count_for_v220_portabil.md`
- **Use AscendC `Power` VEC API for Pow; do not decompose manually** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p55_use_ascendc_power_vec_api_for_pow_do_not_decompo.md`
- **LAUNCH_BOUND + LAUNCH_CHECK** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p5_launch_bound_launch_check.md`
- **AscendC Sort ASC tie-break direction is REVERSED from PyTorch stable ASC (ANTI-PATTERN)** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p60_ascendc_sort_asc_tie_break_direction_is_reversed.md`
- **Determinism-preserving patterns (positive, recommended)** · 199 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p61_determinism_preserving_patterns_positive_recomme.md`
- **`ArithProgression` for vectorized arange / index-sequence generation** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p63_arithprogression_for_vectorized_arange_index_seq.md`
- **PyTorch-UB-class scatter-overwrite — NPU reference is non-deterministic, kernel cannot chase** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p67_pytorch_ub_class_scatter_overwrite_npu_reference.md`
- **grid_y and thread count consistency** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p6_grid_y_and_thread_count_consistency.md`
- **Activation-layout convention disambiguation via CANN op-parameter default fingerprinting** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p71_activation_layout_convention_disambiguation_via.md`
- **Dynamic int8 per-token quant tail — generic primitive chain** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p72_dynamic_int8_per_token_quant_tail_generic_primit.md`
- **Op-signature-as-public-formula recognition (META workflow for fused activation ops)** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p73_op_signature_as_public_formula_recognition_meta.md`
- **Multi-AIC partition-dispatch via host-precomputed segment offsets** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p74_multi_aic_partition_dispatch_via_host_precompute.md`
- **Per-iter output via `TQue<VECOUT, depth=2>` (the TQue side of the OL-94 decision)** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p77_per_iter_output_via_tque_vecout_depth_2_the_tque.md`
- **Row-parallel data unfolding to expose AIV parallelism upstream of single-AIC cube call** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p78_row_parallel_data_unfolding_to_expose_aiv_parall.md`
- **#pragma unroll scope** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p7_pragma_unroll_scope.md`
- **Vectorized index-compression emit via GatherMask for predicate-driven kernels** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p80_vectorized_index_compression_emit_via_gathermask.md`
- **Runtime-bounded loop cap for top-K-style merge / scan with constexpr K_MAX buffer** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p81_runtime_bounded_loop_cap_for_top_k_style_merge_s.md`
- **Deterministic cross-AIV tournament merge of P partial top-K buffers via composite key (value DESC, orig_idx ASC)** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p82_deterministic_cross_aiv_tournament_merge_of_p_pa.md`
- **Column-major intermediate output + pybind transpose to eliminate per-element scalar pack in predicate-driven N-D coord emit** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p83_column_major_intermediate_output_pybind_transpos.md`
- **`AscendC::TopK` adv_api primitive — use this instead of hand-rolled chunked-Sort + scalar 2-pointer merge for k-selection on arch 3510/5102/3003/3113** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p85_ascendc_topk_adv_api_primitive_use_this_instead.md`
- **Empirical-measurement-required for fused-op gap analysis — analytical-only decomposition is reward-hacking on the diagnostic side** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p86_empirical_measurement_required_for_fused_op_gap.md`
- **Hand-rolled transcendental via Cephes-form range reduction with fp32-grade primitives — when an end-to-end transcendental kernel hits a precision ceiling AND vendor source confirms the algorithmic ref** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p88_hand_rolled_transcendental_via_cephes_form_range.md`
- **GM workspace contract for fused ops — public outputs stay separate; opaque scratch is one aligned byte workspace sliced by host offsets** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p89_gm_workspace_contract_for_fused_ops_public_outpu.md`
- **Host benchmark best practice** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p8_host_benchmark_best_practice.md`
- **V220→V351 (arch22→arch35) — surgical strip pattern for op_kernel port** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p90_v220_v351_a3_a5_arch35_surgical_strip_pattern_fo.md`
- **Multi-variant kernel organization — sibling `<op>_<variant>.h` files dispatched by tiling-key from a thin `.cpp`** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p91_multi_variant_kernel_organization_sibling_op_var.md`
- **Quant-op CPU reference MUST `.clamp(low, high).to(int_dtype)` to match NPU hardware clamp** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p93_quant_op_cpu_reference_must_clamp_low_high_to_in.md`
- **`__VEC_SCOPE__ { RegTensor / MaskReg / MicroAPI::* }` — canonical A5 L2 inner-loop shape** · 124 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p96_vec_scope_regtensor_maskreg_microapi_canonical_a.md`
- **Mask-free additive piecewise decomposition — replace MicroAPI Select-based activation forms with `Maxs/Mins` branch gates so each domain reduces to 0 outside its active range, then sum** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p97_mask_free_additive_piecewise_decomposition_repla.md`
- **SIMD vs SIMT selection** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p9_simd_vs_simt_selection.md`
- **重构/升级后防止 regression 的方法论** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/regression_methodology_3way_oracle.md`
- **Rope 类算子族速查** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/rope_family.md`
- **VF 融合、存活寄存器与循环展开的取舍** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/vector/fusion_liveness_and_unrolling.md`
- **OP_CHECK 断言条件写反致 161002 拒合法输入** · 428 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/161002_op_check_condition_inverted.md`
- **507015—Execute预写workspace对kernel不可见** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507015_device_workspace_prewrite_invisible.md`
- **507035 — aten 与自定义命名空间路径对照** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_aten_dispatch.md`
- **507035：模板参数死路径全 N 分配超 UB** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_dead_tpl_path_full_n_alloc.md`
- **507035 — 三队列 float32 tile 超 UB** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_float32_tile_ub.md`
- **Gather L2 整寄存器读 offset 越界致 507035** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_gather_l2_offset_overread.md`
- **507035 全量超时：GetInputStride 全 0 误判** · 198 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_getinputstride_unfilled_misjudge.md`
- **507035：每中间量全尺寸 TBuf 总量超 UB** · 196 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_per_intermediate_full_size_tbuf.md`
- **507035 小 innerSize：整块拷贝+对齐向量混合路径** · 256 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_small_inner_size_hybrid_path.md`
- **507035 — standalone ccec launch 失败** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_standalone_ccec_launch.md`
- **507035 errcode 271：TopK 保留 UB 预算漏记** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_topk_reserved_ub_budget.md`
- **507035：ubFormer 超 blockFormer 尾块越界** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_ub_former_exceeds_block_former.md`
- **561103 同名算子签名冲突链接绑内置版** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103.md`
- **561103 aclIntArray 布局误解致属性读取垃圾值** · 269 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_aclintarray_layout_misread.md`
- **561103: beta 属性按 float 读致 GRAPH_FAILED** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_beta_float_graph_failed.md`
- **561103—GetWorkspaceSize阶段host缓冲为空** · 122 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_host_workspace_null_getworkspacesize.md`
- **561103 — 非连续输入经 Contiguous 仍被拒** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_noncontiguous_input_rejected.md`
- **561103 可选输入 nullptr 须 shape=[0] 占位** · 201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_optional_input_nullptr_placeholder.md`
- **561103 — vendor 未注册 load_priority 不被加载** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_vendor_load_priority_not_registered.md`
- **ACL BOOL 测试数据 vector<bool> 取 data() 编译失败** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/acl_bool_vector_bool_data_deleted.md`
- **复用 aclOpExecutor 致 double free 崩溃** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/acl_op_executor_reuse_double_free.md`
- **aclCreateTensorList 是两参签名，旧多参写法编译失败** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclcreatetensorlist_two_param_signature.md`
- **aclIntArray 无 At()，元素访问用下标** · 213 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclintarray_no_at_use_subscript.md`
- **ACLNN 161001 — ascend910_93 缺算子内核配置** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_161001_ascend910_93.md`
- **沿用内置算子名被系统库抢占报 EZ9903** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_361001_ez9903_builtin_symbol_preemption.md`
- **ACLNN 561000 非连续张量需 AutoContiguous** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_561000_autocontiguous.md`
- **561103 自定义aclnn非fp32调用环境相关失败** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_561103_vendor_env_dtype_partition.md`
- **aclnn 检查宏形态：OP_CHECK 三参与判空宏归属** · 139 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_check_macro_signature_header_ownership.md`
- **aclnn 直调符号望文生义三处编译失败** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_direct_call_symbol_names.md`
- **aclnn 直连复现骨架 + dtype 全编码往返探针** · 182 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_direct_repro_skeleton_noop_probe.md`

下一页：page-018.md
