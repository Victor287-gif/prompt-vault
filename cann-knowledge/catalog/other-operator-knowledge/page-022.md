# other-operator-knowledge: 1471–1540 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Pybind padding wrapper when an upstream kernel uses one input's dim as buffer extent for differently-shaped tensors** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_162_pybind_padding_wrapper_asymmetric_shape.md`
- **model_new_ascendc.py `forward()` MUST exercise the kernel — cache-replay / digest-lookup / subprocess-CPU-offload / PyTorch-fallback are reward-hacking anti-patterns [V351+V220, ALL_MODES, safety-net]** · 110 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_165_model_new_ascendc_py_forward_must_exercise_the_ker.md`
- **Reference-spec drift between archive date and cold-restart silently invalidates prior bit-exact verdicts — mtime pre-flight before treating archived kernel as starting material [V351+V220, ALL_MODES, ** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_168_reference_spec_drift_between_archive_date_and_cold.md`
- **ModelNew.forward output count MUST mirror Model.forward output count — pybind aux outputs are discarded at the ModelNew wrapper boundary [V351+V220, ALL_MODES, safety-net + reference-contract-parity]** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_172_modelnew_forward_output_count_must_mirror_model_fo.md`
- **`__CCE_AICORE__` macro values per Ascend arch (when reading CANN source)** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_173_cce_aicore_macro_values_per_ascend_arch_when_readi.md`
- **Defensive-guard refusal (kernel won't run → 0-output) is the highest tier of failure — NEVER package as "scope skip" / "separate DEBT" / "different limit" [V351+V220, ALL_MODES, agent-discipline + ant** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_175_defensive_guard_refusal_kernel_won_t_run_0_output.md`
- **bisheng multi-branch Simt::VF_CALL causes 507035 crash** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_17_bisheng_multi_branch_simt_vf_call_causes_507035_cr.md`
- **AscendC kernel .so loading requires CANN environment initialization** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_180_cann_env_init_required_for_so_load.md`
- **Per-case taxonomy with evidence_strength grading required before claiming fix coverage** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_182_per_case_taxonomy_with_evidence_strength_grading_r.md`
- **`_OutOfScope` sentinel convention for FA-class `model_new_ascendc.py`** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_184_outofscope_sentinel_convention_for_fa_class_model.md`
- **507015 aicore exception under NPU contention — distinguish a device-load crash from a kernel ADDR_MISALIGN bug** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_189_507015_aicore_exception_npu_contention.md`
- **torch_npu does not surface a kernel's foreign GM writes into a returned raw-byte workspace tensor — probe via TYPED OUTPUT tensors** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_192_torch_npu_does_not_surface_a_kernel_s_foreign_gm_w.md`
- **`--port-a3` CLI op_classification seed MUST include algorithm-class tags + complexity, not just the mode tag — otherwise FA-class detection cannot fire for FA-class ports** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_194_port_a3_cli_op_classification_seed_must_include_al.md`
- **V220 manual per-fractal / 3D-im2col cube operand loads are A5-invalid — port_a3→A5 must use a single 2D `LoadData2DParamsV2` (mStep-encoded)** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_197_v220_manual_per_fractal_3d_im2col_cube_operand_loa.md`
- **A workspace-reset (cold-start) contract must evict EVERY in-workspace backup/snapshot dir, not just one naming convention — any surviving snapshot is a restore-target that the next generator agent wil** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_198_a_workspace_reset_cold_start_contract_must_evict_e.md`
- **GlobalTensor::SetValue() unreliable in SIMD AIV-only mode** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_19_globaltensor_setvalue_unreliable_in_simd_aiv_only.md`
- **STARS2.0 is the hardware mechanism behind A5 multi-core core-split + cross-core sync — Group scheduling (≤8 Groups, by-Die L2-affinity), compute-partition (AIC/AIV/SDMA into ≤16 resource pools), and a** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_211_stars2_0_is_the_hardware_mechanism_behind_a5_multi.md`
- **Diagnostic matrix — symptom-to-root-cause lookup for common AscendC kernel failures [V351, ALL_MODES, diagnostic]** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_216_diagnostic_matrix_symptom_to_root_cause_lookup_for.md`
- **AscendC documentation is JS-rendered — use dev-browser plugin to access** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_22_ascendc_documentation_is_js_rendered_use_dev_brows.md`
- **A shell `*` glob does NOT match dotfiles — cleanup of a SHARED workspace must explicitly remove dotfile markers, or a prior op's stale marker silently contaminates the next op** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_232_a_shell_glob_does_not_match_dotfiles_cleanup_of_a.md`
- **Generic flat signature-based input-gen produces degenerate rank-1 tensors for ops with inter-tensor shape constraints — route any op whose inputs have interdependent ranks/shapes to SCHEMA-based input** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_233_generic_flat_signature_based_input_gen_produces_de.md`
- **A finalize `phase_o5_mismatch` ("pass claimed but not measured") with genuinely-correct worker counts is usually a stale interpreter/env pointer for the active container — never reduce counts to "fix"** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_238_a_finalize_phase_o5_mismatch_pass_claimed_but_not.md`
- **Per-dtype GM stride alignment for multi-dtype output tensors — each output uses its own dtype's alignment unit, not a uniform stride** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_259_per_dtype_gm_stride_alignment_for_multi_dtype_outp.md`
- **Member variable shadowing in AscendC `Init()` — `type name_ = expr` creates a LOCAL that shadows the class member, leaving the member at its initializer value (typically 0)** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_264_member_variable_shadowing_in_ascendc_init_type_nam.md`
- **AscendC API symbols are split between global scope and the `AscendC::` namespace — when a qualified or unqualified symbol isn't found, try the OTHER scope; the `did you mean simply X?` hint means the ** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_267_ascendc_api_symbols_are_split_between_global_scope.md`
- **Research official docs + CANN source before optimization attempts** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_26_research_official_docs_cann_source_before_optimiza.md`
- **Backward-mode op-gen uses the pybind `.so` paradigm end-to-end — the `op_host/` build contract (PB-33) and the Phase-O5 workspace resync BOTH have backward-specific carve-outs the worker must handle i** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_272_backward_mode_op_gen_uses_the_pybind_so_paradigm_e.md`
- **Size a dtype-templated kernel's UB for the WIDEST dtype and run the full multi-dtype verify (507035)** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_273_dtype_templated_ub_widest_dtype_507035.md`
- **Before escalating a 507035 on a freshly-built kernel as infra, run a read-only vendor-op probe on the SAME device** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_274_507035_vendor_op_probe_before_infra.md`
- **A verification script must be data-flow-independent of the verdict artifact it re-derives — emit diagnostics to stderr and ONE summary object to stdout; the caller (not the verifier) materializes the ** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_277_a_verification_script_must_be_data_flow_independen.md`
- **MIX dispatch runs ONE binary on BOTH AIC and AIV — every core-type-specific op (TQue/InitBuffer/Mmad/UB) must sit behind an `ASCEND_IS_AIC`/`ASCEND_IS_AIV` guard, and cross-stage `SyncAll<false>()` ba** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_278_mix_dispatch_runs_one_binary_on_both_aic_and_aiv_e.md`
- **Fixpipe L0C→GM `dstStride` unit is PER-FORMAT — elements for NZ2ND (`CFG_ROW_MAJOR`, the default) vs 32-byte blocks for NZ — and Nd2Nz `srcDValue` is a GM row stride in elements [910B2C / 220x / CANN ** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_279_fixpipe_l0c_gm_dststride_unit_is_per_format_elemen.md`
- **a self-heal / auto-provision that stages a Python entrypoint MUST provision its FULL dependency closure (the imported package + relative-path deps), not just the entrypoint file — else fresh-container** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_281_a_self_heal_auto_provision_that_stages_a_python_en.md`
- **Edge-case discovery methodology (quantization ops)** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_29_edge_case_discovery_methodology_quantization_ops.md`
- **a captured ground truth speaks only for the chip it was captured on: audit the capture AND its device provenan** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_404_a_captured_ground_truth_speaks_only_for_the_chip.md`
- **dav-m200 (Ascend310P3) compute-dtype scope is fp32/fp16 ONLY: the ascend310p AICoreConfig registers {DT_FLOAT** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_412_dav_m200_ascend310p3_compute_dtype_scope_is_fp32.md`
- **harness flat-N datasets store every input at full length — host tiling must replicate the reference model's sh** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_417_harness_flat_n_datasets_store_every_input_at_ful.md`
- **int64 index tensor in SIMT — direct __gm__ read works** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_41_int64_index_tensor_in_simt_direct_gm_read_works.md`
- **Cast(bf16→f32→bf16) roundtrip is not necessarily bit-exact** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_44_cast_bf16_f32_bf16_roundtrip_is_not_necessarily_bi.md`
- **in-place CANN ops (aclnnApply) mutate their inputs through the torch_npu wrapper** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_542_in_place_cann_ops_aclnnapply_mutate_their_inputs.md`
- **allow_internal_format=True reorders within the inner dimension (NZ/fractal) — for element-ORDER-dependent ops** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_543_allow_internal_format_true_reorders_within_the_i.md`
- **One consuming GM write per TQue EnQue/DeQue cycle — a dequeued tensor held across multiple DataCopy issues silently drops all but the first few writes; the per-cycle cost (~0.5µs) sets the batching fl** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_560_one_consuming_gm_write_per_tque_enque_deque_cycl.md`
- **An inter-stage dtype quantize between two raw kernel launches belongs IN-KERNEL — host-staged `.to(at::kCPU)` materialization is a P149-rejected CPU-offload shape and independently pays a D2H+cast+H2D** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_564_an_inter_stage_dtype_quantize_between_two_raw_ke.md`
- **Cast-to-Int32 workspace for non-float atomicAdd** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_58_cast_to_int32_workspace_for_non_float_atomicadd.md`
- **Encode the documented aclnn error-code contract in UT FIRST and let the red UT drive wrapper structure — semantic param checks belong in `CheckParams` (161002/161001), not inside the graph builder (56** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_662_encode_the_documented_aclnn_error_code_contract.md`
- **GEIR attr element indexing must be diffed across ALL host entry points — infershape, tiling, and the aclnn wrapper can silently diverge on attr conventions, and a host-computed-shape wrapper never exe** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_663_geir_attr_element_indexing_must_be_diffed_across.md`
- **torch::zeros on NPU is NOT stream-ordered with custom kernels** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_66_torch_zeros_on_npu_is_not_stream_ordered_with_cust.md`
- **dav-m200 TQueBind<VECIN,VECOUT> is UNSAFE for scalar-consumer RMW** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_675_dav_m200_tquebind_vecin_vecout_is_unsafe_for_sca.md`
- **Every acceptance-script family member must self-source its CANN environment — sibling-script env asymmetry produces a "precision works, perf crashes" split-brain** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_686_every_acceptance_script_family_member_must_self.md`
- **Some torch_npu fused ops are unavailable on Ascend950PR** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_68_some_torch_npu_fused_ops_are_unavailable_on_ascend.md`
- **Restart container before experiments (zombie processes)** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_6_restart_container_before_experiments_zombie_proces.md`
- **Hardware-accelerator typed-config struct — use the typed config, not the generic policy** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_91_hardware_accelerator_typed_config_struct_use_the_t.md`
- **When the reference path's output depends on RNG global state, both ref and cand must agree on a seed** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_95_when_the_reference_path_s_output_depends_on_rng_gl.md`
- **op_api 补数据内容校验：读 device 数据验 Sum=A** · 226 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_api_data_content_validation_aclrtmemcpy.md`
- **op_api UT 引入 runtime API 须补 libacl_rt** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_api_ut_link_libacl_rt.md`
- **OpAttrDef 无 Int64()，int64 属性用 Int()** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_attr_def_integer_attr_use_int_not_int64.md`
- **OP_CHECK条件反转误报161001** · 228 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_check_condition_inversion_161001.md`
- **OP_CHECK 第三参花括号逗号拆参致编译失败** · 191 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_check_macro_brace_comma_arg_split.md`
- **OP_CHECK_NOTNULL 内嵌 int 返回，指针函数编译错** · 205 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_check_notnull_int_return_pointer_function.md`
- **op_host 编辑破坏块结构报 graphStatus 误导错** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_host_brace_break_graphstatus_misleading.md`
- **op_host 直连 libruntime 内部符号致安装拒装** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_host_extern_c_libruntime_undefined_symbol.md`
- **op_host fwk 编译链拒 C++20 写法** · 139 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_host_fwk_modules_rejects_cxx20.md`
- **op_host 层 OP_CHECK 未定义须用 OP_CHECK_IF** · 181 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_host_op_check_if_macro_name.md`
- **op_host util 头不互含，未声明报错补 include** · 216 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_host_util_headers_require_explicit_include.md`
- **OP_LOGE 首参传 nullptr 触发编译错** · 216 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_loge_first_param_nullptr_compile_error.md`
- **缺 OP_PROTO_LIB 致空 axes 命中内置 InferShape** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_proto_lib_axes_infershape.md`
- **OP_WORKSPACE 常量覆盖 tiling 动态 workspace** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_workspace_constant_overrides_dynamic_ws.md`
- **op_def 属性链写 Required() 编译失败** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opattrdef_no_required_setter.md`
- **OpAttrDef 无 SetDefault/RequiredListInt** · 152 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opattrdef_no_setdefault_required_overload.md`
- **OpDef 改动不生效：ops-info 与 autogen py 不刷新** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opdef_change_stale_ops_info_autogen_py.md`

下一页：page-023.md
