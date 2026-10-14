# precision-tests: 491–560 / 893

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **kernel 输出=输入：binary 不随源码变化即守卫吞体** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/kernel_binary.md`
- **位真模拟器先过跨 case 保真校验再断案——模拟器自身 bug 会伪装成 kernel 精度缺陷** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/kernel_sim_fidelity_check_001.md`
- **Gather 双加 LocalTensor 基址导致交错旋转乱码** · 321 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/localtensor_gather_base_address_not_double_counted.md`
- **LocalTensor SetValue 参数颠倒导致索引错乱** · 205 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/localtensor_setvalue_uses_offset_then_value.md`
- **LocalTensor 混用 Vector/Scalar 写致输出全错** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/localtensor_vector_scalar_mixed_write.md`
- **scatter 归约逐次回写原生 dtype 放大舍入误差** · 450 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/low_precision_scatter_reduction_needs_fp32_accumulator.md`
- **批次行索引未对 M 回绕致整行置零判别** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/m.md`
- **Matmul workspace 预偏移 totalN 回读 banned** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/matmul_workspace_totaln_banned.md`
- **Maxs 下限 clamp 阻止 exp 自然下溢残留** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/maxs_clamp_exp.md`
- **bf16算子MERE超阈但失配占比极低的归约顺序指纹** · 186 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mere_over_threshold_tiny_bitwise_mismatch_reduction_order_fingerprint.md`
- **MERE 过而 MARE 爆炸：golden 近零与精度模型不一致** · 121 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mere_pass_mare_explode_near_zero_golden.md`
- **未分配 TBuf 导致浮点索引随机错乱** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/missing_tbuf_allocation_causes_undefined_ub_index_scratch.md`
- **DataCopyPad 后未先同步 Cast 导致低精度 NaN** · 364 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mte2_barrier_must_precede_vector_cast.md`
- **MulAddDst FMA 单次舍入与 golden Mul+Add 双舍入偏差** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/muladddst_fma_golden_mul_add.md`
- **多输出 TensorList 地址经 op 属性传 kernel** · 254 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multi_output_tensorlist_addr_op_attr.md`
- **多核共享首段行区间导致部分输出未写** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multiblock_launch_must_derive_row_range_from_block_index.md`
- **多核非对齐 DataCopyPad 写回边界互相覆盖** · 194 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multicore_non_aligned_datacopypad_rmw_overlap.md`
- **多核 scatter 同目标非原子 RMW 丢更新，改单核** · 196 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multicore_nonatomic_rmw_same_target_lost_updates.md`
- **多核 S 管线标量写跨核不可见** · 290 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multicore_scalar_gm_write_cross_core_invisible.md`
- **多遍扫描漏 CopyInChunk，DeQue 读到残数据** · 200 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multipass_scan_missing_copyin_stale_deque.md`
- **多张量min范数合并读未写workspace零值** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multitensor_min_uninitialized_partial_slot_zero.md`
- **MX-FP4 NaN 块 device 写 e8m0=0xFF、fp4=0x0** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mx_fp4_nan_device_e8m0_0xff_fp4_0x0.md`
- **2D NDDMA 小内维落位错位致精度失配** · 174 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/nddma_2d_small_inner_dim_misplacement.md`
- **NDDMA 快速路径 C=1 中间轴漏维判别** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/nddma_c_1.md`
- **NDDMA多维dstStride致data搬运偏移的判别** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/nddma_dststride_data.md`
- **NDDMA 最内层 srcStride=0 广播读越界** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/nddma_srcstride_0.md`
- **自建比较器近零兜底 epsilon 用独立魔法数与阈值脱节 — 灾难性消除产物仍走相对误差** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/near_zero_bucket_epsilon_decoupled_from_threshold.md`
- **NEG_INF 掩码常量按单一 Dk 标定致漏检** · 208 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/neg_inf_mask_constant_decouple_from_dk.md`
- **inf 输入 Newton 开方 inf/inf 传播 NaN 需早退** · 161 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/newton_sqrt_inf_nan_early_exit.md`
- **fp32 UB 非 VL 拷贝须用带掩码 Reg** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/non_vl_fp32_ub_copy_needs_masked_reg_load_store.md`
- **非连续输入结果错：tiling 无 stride 须 L2 Contiguous** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/noncontig_tiling_no_stride_l2_contiguous.md`
- **跨行子块误作连续 slab 导致分块结果错** · 179 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/noncontiguous_row_subtiles_require_per_row_copy.md`
- **noTrans extract 与 NDExtLayoutPtn 方向不匹配** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/notrans_extract_ndextlayoutptn.md`
- **多核 numPerCore 须对齐 Philox group 4** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/numpercore_philox_group_4.md`
- **numpy legacy 标量提升致 lr_adj 1-ULP 偏差** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/numpy_legacy_lr_adj_1_ulp.md`
- **官方小值域分支的两种比对器落地 — max(errorLimit,|golden|) FTZ 容忍型与固定 2^-16 绝对阈型** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/official_small_value_branch_two_implementations.md`
- **[RETRACTED 2026-04-29 — depends on retracted OL-99]** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_100_retracted_2026_04_29_depends_on_retracted_ol_99.md`
- **PyTorch fp32 linspace is not closed-form replicable in kernel — precompute on CPU host and ship as input** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_101_linspace_precompute_host_aux_table.md`
- **fp16/bf16 composite ops must replicate CPU's per-op rounding chain (cast back to T between each mul and each add)** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_102_fp16_bf16_per_op_rounding_chain.md`
- **NPU hardware transcendentals are ~fp16 piecewise-polynomial — kernels using them cannot pass fp32 thresholds vs CPU libm** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_103_npu_transcendentals_fp16_precision.md`
- **Parallel-vs-serial conflict — modify the BENCHMARK reference to a CPU serial impl aligning with the kernel's algorithm** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_104_modify_benchmark_reference_cpu_serial.md`
- **Software fp32 sigmoid Python prototype passes; AscendC SIMD port has bit-manipulation lowering bugs — debug in standalone test kernel before integrating** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_105_software_fp32_sigmoid_python_prototype_passes_asce.md`
- **KB-sufficiency audit on SIMT scatter — op#21_Scatter doesn't apply P-P67/OL-67 (KB knows, archive doesn't)** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_107_kb_sufficiency_audit_on_simt_scatter_op_21_scatter.md`
- **SIMT precision-gap taxonomy — distinct from OL-102 chain rounding** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_108_simt_precision_gap_taxonomy.md`
- **Two-tier precision verdict with fail-closed independent-truth boundary** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_109_two_tier_precision_verdict.md`
- **Reduction-tree fail-floor invariant — swapping per-stage reduction shape only moves which cases fail, not the total** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_110_reduction_tree_fail_floor_invariant.md`
- **On-device pilot before applying any probe directive — CPU-fp32 simulation does not bit-equal NPU-fp32 unit ordering** · 70 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_111_on_device_pilot_before_applying_any_probe_directiv.md`
- **fp32 op-order is load-bearing — literal translation beats algebraic-equivalent rewrite for cancellation-prone chains** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_112_fp32_op_order_is_load_bearing.md`
- **Lane host dir is NOT auto-synced from project workspace — `deploy_to_npu_lane.sh` reads stale source unless caller manually copies first** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_113_lane_host_dir_is_not_auto_synced_from_project_work.md`
- **Reference output dtype comes from Model.forward body, not input dtype — read the source before allocating** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_118_reference_output_dtype_from_forward.md`
- **Use bit-pattern -inf (0xFF800000), not -FLT_MAX literal, for fp32 reduction-mask sentinels** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_119_bit_pattern_neg_inf_sentinel.md`
- **fp32-internal compute + single CAST_RINT end-cast is the canonical fp16/bf16 path when the reference is torch_npu.<op> (CANN fused)** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_120_fp32_internal_single_cast_rint.md`
- **SIGTERM does NOT propagate through `ssh + docker exec` chain — wrap remote workloads with `timeout` to bound them** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_121_sigterm_does_not_propagate_through_ssh_docker_exec.md`
- **V220 TBuf→MTE3 DataCopy reads stale data even after VEC write — use TQue<VECOUT> for output [V220]** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_128_v220_tbuf_mte3_datacopy_reads_stale_data_even_afte.md`
- **Path A canonical-evaluator §4.5.3 small-value fallback degenerates when the reference equals CPU truth** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_138_path_a_small_value_fallback_degenerates.md`
- **A5 narrow-float datatype family — FP8 / HiFloat8 / FP4x2 / fp8_e8m0_t scale [V351, narrow-float]** · 110 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_144_a5_narrow_float_datatype_family_fp8_hifloat8_fp4x2.md`
- **A5 Subnormal handling via `LnConfig{algo}` for Ln/Sqrt/Rsqrt/Div/Reciprocal/Exp [V351, precision-config]** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_147_a5_subnormal_handling_via_lnconfig_algo_for_ln_sqr.md`
- **Conditional-write kernels: bit-exact verify must mask the uninit tail the kernel skips** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_161_conditional_write_mask_uninit_tail.md`
- **Bit-exact negation needs Sub(z, zero_buf, x); Muls(x, -1) diverges on signed zero** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_166_bit_exact_negation_sub_not_muls.md`
- **Cold-start recovery — restore from `.opgen_backups` when a workspace wipe lost only runner/wrapper artifacts but the validated baseline kernel was preserved [V351+V220, ALL_MODES, workflow]** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_171_cold_start_recovery_restore_from_opgen_backups_whe.md`
- **TPipe must be a class member — local destructor frees TQue buffers before async MTE3 writes complete** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_179_tpipe_must_be_a_class_member_local_destructor_free.md`
- **FA-class finalize-gate matrix — 7 finalize-time gates wired in benchmark plugin (PR #125)** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_183_fa_class_finalize_gate_matrix_7_finalize_time_gate.md`
- **V351 forward FA — cube-MatmulImpl P@V is required to match CANN precision; vec-only scalar-broadcast P@V drifts +5-25pp mare at large n_kv_tile** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_186_v351_fa_cube_matmul_pv_required.md`
- **port_a3 V220-pure — emit ALL host-dispatched algorithm variants and replicate the dispatch rule; single-variant ports leak shape-dependent sub-LSB diffs that mimic hw-floor** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md`
- **Every code change MUST be followed by BOTH precision AND performance verification** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_18_every_code_change_must_be_followed_by_both_precisi.md`
- **Magnitude test-inputs must be dtype-aware, and nan/inf-reference cases are DEGENERATE (unscoreable), not FAIL** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_191_magnitude_test_inputs_must_be_dtype_aware_and_nan.md`
- **V220→V351 port discipline — explicit `V→MTE3` sync required between Cast/V-compute and DataCopy store; `PipeBarrier<PIPE_V>` is NOT sufficient** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_193_v220_v351_port_discipline_explicit_v_mte3_sync_req.md`
- **When a generation step's correct output value is fixed by an upstream design artifact, ENFORCE it with a deterministic post-emit gate — do not rely on the (LLM) generator reproducing it** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_199_when_a_generation_step_s_correct_output_value_is_f.md`
- **Never mark issues "confirmed safe" without code change** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_1_never_mark_issues_confirmed_safe_without_code_chan.md`
- **An alternate-oracle / CPU-reference precision probe must gate on the dtype rounding floor with AND, never max_abs ≤ T1 OR mean ≤ T2** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_203_alt_oracle_probe_and_gate_dtype_floor.md`

下一页：page-009.md
