# precision-tests: 561–630 / 893

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Establish bug-reproducibility BEFORE building a proxy-investigation — a non-reproducible defect's proxy-refutations are not bug-relevant** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_207_establish_bug_reproducibility_before_building_a_pr.md`
- **A deterministic-test discipline is only as valid as its target — the proxy itself can be refuted; keep the protocol, hold the target as falsifiable** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_208_a_deterministic_test_discipline_is_only_as_valid_a.md`
- **grep the op's own call-graph before attributing a defect to a subsystem; NO_FIRE ≠ clean; don't revive a grep-refuted root by assertion** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_209_grep_the_op_s_own_call_graph_before_attributing_a.md`
- **Single-core-first testing strategy — halve the search space before multi-core debugging [V351, ALL_MODES, testing-methodology]** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_214_single_core_first_testing_strategy_halve_the_searc.md`
- **At a finalize gate that re-measures claimed counts, emit a concrete pass count for EVERY pass the runner can reliably reproduce — minimize N/A claims so one transient measurement can't sink a correct ** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_218_at_a_finalize_gate_that_re_measures_claimed_counts.md`
- **NPU tensor lifecycle management in serial multi-case kernel verification — del tensors + empty_cache between cases prevents silent data corruption** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_222_npu_tensor_lifecycle_management_in_serial_multi_ca.md`
- **A multiplicative output gate amplifies the upstream backward gradient — the most-accumulated grad overflows fp16 first; classify as a dtype-range edge, not an algorithm bug, when fp32/bf16 hold and th** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_227_gated_backward_fp16_overflow_dtype_range_edge.md`
- **Verify a hand-derived analytic backward against autograd on a PURE-fp64 forward — a reference forward that internally upcasts (.float()) injects a spurious ~1e-7 round-off that masks whether the deriv** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_228_verify_analytic_backward_on_pure_fp64_forward.md`
- **Compute a dtype precision-floor by rounding the kernel's actual low-precision INPUT through the reference math — NOT by rounding the fp64 truth OUTPUT down to the dtype** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_243_precision_floor_round_input_not_output.md`
- **`pipe_barrier(PIPE_ALL)` is REQUIRED between VEC compute (Add/Muls) and MTE2 DataCopy when both touch the same UB buffer — silent precision corruption without it** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_247_pipe_barrier_pipe_all_is_required_between_vec_comp.md`
- **Mirror Phase O5 verifier plugin-skip and tier-normalization logic across the local-container and SSH paths — unmirrored divergence falsely rejects correct kernels at finalize** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_261_mirror_phase_o5_verifier_plugin_skip_and_tier_norm.md`
- **Closing a coverage gap can surface failures rooted in the truth-source reference, not the kernel — a suite over-fit to the trivial path masks reference bugs and can invert a correct kernel's verdict** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_262_closing_a_coverage_gap_can_surface_failures_rooted.md`
- **Disambiguate a mode-defaulting reference capture empirically against the captured data, not from kernel source or a cross-project default** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_269_reference_mode_disambiguate_empirically.md`
- **A reference model's `forward()` must accept the harness case-dict kwargs shape, not only the flat positional signature — else native/cpu-truth provisioning fail-closes silently and grading degrades to** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_270_a_reference_model_s_forward_must_accept_the_harnes.md`
- **Compute a Gaussian-CDF / probit tail from the erfc direct form, not 1+erf (fp32 cancels where erf→−1)** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_271_gaussian_cdf_tail_erfc_direct_form.md`
- **A raw hand-rolled cube is deterministic ALONE, but a multi-cube pipeline self-poisons (silent wrong-result, not a hang) unless it uses a managed cube lifecycle — and a stub-AIV MIX co-schedule perturb** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_275_a_raw_hand_rolled_cube_is_deterministic_alone_but.md`
- **A Cube fp32 matmul precision residual on a cancellation case is `baseK`-tiling-INVARIANT — it is reference-ub (accumulation order vs the vendor Mmad), not a tunable, so do not sweep tiles to fix it** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_276_a_cube_fp32_matmul_precision_residual_on_a_cancell.md`
- **prove a shared fa_class-template edit is "FA-neutral" with a 3-gate codegen-diff — `.text`-section md5 of the FA `.o` before/after (determinism-validated) + live bit-exact + downstream-compile — NOT l** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_280_prove_a_shared_fa_class_template_edit_is_fa_neutra.md`
- **identify the correct OPTIMIZATION VEHICLE before spending — some ops ship a dedicated custom test rig, NOT the standard `orch --optimize` O2.5 pipeline; running the wrong vehicle burns quota stuck at ** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_282_identify_the_correct_optimization_vehicle_before_s.md`
- **Audited CPU PyTorch is the numerical specification** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_28_audited_cpu_pytorch_is_the_numerical_specification.md`
- **Expert feedback must result in code changes, never just documentation** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_2_expert_feedback_must_result_in_code_changes_never.md`
- **SIMD performance optimization must not come at the cost of precision** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_30_simd_performance_optimization_must_not_come_at_the.md`
- **NPUKernelBench — PyTorch wrapper is CANN delegation (PROHIBITED)** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_36_npukernelbench_pytorch_wrapper_is_cann_delegation.md`
- **the CPU-truth reference must replicate the kernel's out-of-range index policy: scatter/index kernels that SKIP** · 21 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_408_the_cpu_truth_reference_must_replicate_the_kerne.md`
- **ops whose inputs are forward-pass-derived statistics (mean/rstd/var) MUST have those test inputs computed from** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_411_ops_whose_inputs_are_forward_pass_derived_statis.md`
- **a reference model hardcoding fp64 compute produces an fp64-disguised native_capture.pt (bit-identical to cpu_t** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_418_a_reference_model_hardcoding_fp64_compute_produc.md`
- **ACLRT_LAUNCH_KERNEL in pybind11 is ASYNC — aclrtSynchronizeStream(stream) before any local at::Tensor goes out** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_546_aclrt_launch_kernel_in_pybind11_is_async_aclrtsy.md`
- **a GM-delta RMW design must initialize the delta from the (widened) base tensor — zero-init + post-add merge co** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_547_a_gm_delta_rmw_design_must_initialize_the_delta.md`
- **bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + nan_to_n** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_549_bit_exact_comparison_vs_a_reference_with_legitim.md`
- **host-derived scalar coefficients: compute in DOUBLE, cast ONCE to the tensor dtype** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_550_host_derived_scalar_coefficients_compute_in_doub.md`
- **exact zero/nonzero predicates over 16-bit float tiles on narrow-VEC SoCs (dav-m200)** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_551_exact_zero_nonzero_predicates_over_16_bit_float.md`
- **scatter-accumulate numeric semantics are dtype-asymmetric and SoC-specific — bit-probe the reference's per-dty** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md`
- **a vendor-native acceptance golden can be internally non-deterministic — run the golden twice and self-diff it** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_556_a_vendor_native_acceptance_golden_can_be_interna.md`
- **A prose/simulator invariant divergence is itself a defect carrier — regenerate prose invariants from the simulator's formula** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_565_a_prose_simulator_invariant_divergence_is_itself.md`
- **A captured ground truth speaks only for the chip it was captured on — audit the capture AND its device provenance before encoding semantics from it or from any advisory** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_628_a_captured_ground_truth_speaks_only_for_the_chip.md`
- **When a lane has TWO graders with DIFFERENT goldens, measure golden-vs-golden distance per failing case BEFORE kernel iteration — a both-pass intersection smaller than the case set is a structural bloc** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_667_when_a_lane_has_two_graders_with_different_golde.md`
- **A composite grader verdict ANDs independent sub-checks — when the statistical metric passes but the verdict fails, the per-element bound is the binding constraint; decompose the verdict before iterati** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_668_a_composite_grader_verdict_ands_independent_sub.md`
- **aten CPU-fallback scatter-reduce chain table on 310p: fp16/bf16 are fp32-shadow + ONE final RNE round (add AND multiply); fp32 is exact native; reduce=None is ascending last-wins** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_671_aten_cpu_fallback_scatter_reduce_chain_table_on.md`
- **bf16 reference for PURE-COPY ops (pad/tile/repeat/broadcast) can legally run the NPU fp32-roundtrip** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_679_bf16_reference_for_pure_copy_ops_pad_tile_repeat.md`
- **A fp32 "precision FAIL" against a fp64 golden on a multi-term-reduction op is a TRIAGE problem before it is a kernel bug — bisect the cancellation floor first, then (on vendor-suite lanes) declare the** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_681_a_fp32_precision_fail_against_a_fp64_golden_on_a.md`
- **bf16 Adds with a zero scalar may hit PB-4** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_74_bf16_adds_with_a_zero_scalar_may_hit_pb_4.md`
- **NPU fp16/bf16 division by zero produces inf — bit-exact match with PyTorch CPU** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_79_npu_fp16_bf16_division_by_zero_produces_inf_bit_ex.md`
- **API existence MUST be looked up in the catalog — never invent a workaround from memory or guess** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md`
- **bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_81_bf16_fp16_cast_default_should_be_cast_rint_not_cas.md`
- **1-ULP boundary mismatch signature — probe must be exhaustive, cannot be waived directly** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_83_1_ulp_boundary_mismatch_signature_probe_must_be_ex.md`
- **Worker brief attention budget — secondary quality dimensions must not dilute primary** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_84_worker_brief_attention_budget_secondary_quality_di.md`
- **Logic-first precision fix — anti-overfitting, anti-reward-hacking (CRITICAL)** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md`
- **Before spawning aog-kernel-worker, preflight benchmark reference run — schema-invalid inputs poison downstream evidence** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_87_before_spawning_aog_kernel_worker_preflight_benchm.md`
- **Reference-op run-to-run non-determinism — Phase O2.5 pre-flight + scope-by-construction** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_88_reference_op_run_to_run_non_determinism_phase_o2_5.md`
- **When the active reference is non-deterministic, look for a docstring-embedded Python decomposed reference and use IT as math ground truth** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_89_when_the_active_reference_is_non_deterministic_loo.md`
- **PyTorch-UB-class ops with non-deterministic reference — detection + verifier-side alt-ref mitigation** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md`
- **API static-template-arg vs runtime-member ambiguity — trace the source to find the actual driver** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_92_api_static_template_arg_vs_runtime_member_ambiguit.md`
- **Stale `LOCAL_TASK` staging masquerades as runtime non-determinism — `diff workspace/{op}/kernel LOCAL_TASK/kernel` before declaring a stuck signature** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_96_stale_local_task_staging_masquerades_as_runtime_no.md`
- **Precision evidence: CPU truth and source-arch behavior capture** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_97_precision_target_cpu_truth_vs_cann_bit_match.md`
- **PyTorch torch.linspace enforces endpoint-exact — step*(n-1) does NOT equal end in fp32** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_98_linspace_endpoint_exact_fp32.md`
- **[RETRACTED 2026-04-29 — premise was wrong, see retraction note below]** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_99_retracted_2026_04_29_premise_was_wrong_see_retract.md`
- **OpDef 属性截断 double β 致 NaN 失配** · 233 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/opdef_scalar_attr_double_truncation_nan.md`
- **累计极值 NaN 语义须用 self-NE 合成选择掩码** · 369 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ordered_compare_alone_cannot_propagate_nan_prefix_state.md`
- **per-core zero-out 经 grid-stride ZeroOutS** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/per_core_zero_out_grid_stride_zeroout_simt_overshoot.md`
- **philox offset 推进量须固定 12 而非元素数** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/philox_offset_12.md`
- **Philox offset 须放 counter[2:3] 而非 [0:1]** · 92 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/philox_offset_counter_2_3_0_1.md`
- **PhiloxRandom 默认 7 rounds 与原生 10 不匹配** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/philoxrandom_7_rounds_10.md`
- **跨批首行漏比较—续批从 row 0 扫描** · 224 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/prefix_scan_continuation_batch_must_compare_first_row.md`
- **RANK大于5时NDDMA漏维只解释accum错误** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rank_5_nddma_accum_var.md`
- **低 rank 零填充致 count=0 用 td_->rank 非 RANK** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rank_count_0_td_rank_rank.md`
- **rank>5 广播缺外层 Flat loop 致大面积失配** · 413 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/rank_gt5_nddma_outer_loop_missing.md`
- **Reciprocal 低精度致 sigmoid 比对失败** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reciprocal_sigmoid.md`
- **Reg 归约结果直接 Duplicate 导致末轴 Softmax 偏差** · 277 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reduce_lane0_must_be_broadcast_before_vector_normalization.md`
- **ReduceSum复用源改写computeBuf致脏读** · 186 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reducesum_isreusesource_corrupts_source_buffer.md`
- **ReduceSum RA(c310)累加序：16宽平衡寄存器树** · 337 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reducesum_ra_pattern_tree_order_bitwise.md`

下一页：page-010.md
