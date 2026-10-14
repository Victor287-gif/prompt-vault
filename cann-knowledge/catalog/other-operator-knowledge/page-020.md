# other-operator-knowledge: 1331–1400 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **fp32 `-inf` sentinel must be `0xFF800000` (true IEEE -inf), NOT `-FLT_MAX`** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_28_fp32_inf_sentinel_must_be_0xff800000_true_ieee_inf.md`
- **bisheng (310p/dav-m200 backend) rejects plain namespace-scope `constexpr` in aicore code — use `static constexpr` (internal linkage)** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_290_bisheng_310p_dav_m200_backend_rejects_plain_name.md`
- **`Muls(dst, src, tensorOperand, n)` fails to compile — the `-s`-suffix VEC ops take a SCALAR operand; the tensor×tensor product is `Mul`** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_291_muls_dst_src_tensoroperand_n_fails_to_compile_th.md`
- **template-derived kernel classes must qualify dependent-base members with `this->` — bare `pipe_.InitBuffer(...)` fails name lookup (C++ two-phase lookup)** · 22 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_292_template_derived_kernel_classes_must_qualify_dep.md`
- **dav-m200 Level-0 binary repeat-form `mask` is a PER-REPEAT element count — fp32 silently clamps at 64; long axes must be chunked or use the Level-2 contiguous form** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_297_dav_m200_level_0_binary_repeat_form_mask_is_a_pe.md`
- **`SortConfig` device-side is 2-field, not 4-field (host tiling header mismatches)** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_29_sortconfig_device_side_is_2_field_not_4_field_host.md`
- **`GM_ADDR` needs typed pointer cast** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_2_gm_addr_needs_typed_pointer_cast.md`
- **`Select(dst, mask, src0, src1)` mask polarity — mask bit=1 → src0 (NOT src1)** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_31_select_dst_mask_src0_src1_mask_polarity_mask_bit_1.md`
- **Advanced `Sort` 6-arg overload with `srcIndexTensor` triggers aivec 343 on Ascend950PR** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_33_advanced_sort_6_arg_overload_with_srcindextensor_t.md`
- **`const LocalTensor<T>` variable blocks Sort / write-through VEC API overload resolution** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_34_const_localtensor_t_variable_blocks_sort_write_thr.md`
- **AIV device binary does not link libm transcendentals — split SIMT + SIMD for per-element trig/exp/log** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_35_aiv_device_binary_does_not_link_libm_transcendenta.md`
- **`Cast<T, T>` same-dtype is a no-op — destination stays uninitialised (silent data corruption)** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_36_cast_t_t_same_dtype_is_a_no_op_destination_stays_u.md`
- **K2 workspace must be pre-zeroed when K1 cores may skip their slot (fused scatter+reduce)** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_37_k2_workspace_must_be_pre_zeroed_when_k1_cores_may.md`
- **ASCENDC_API_CATALOG.md miss ≠ API doesn't exist — must `ls` adv_api/ headers before falling back to manual decomposition** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_38_ascendc_api_catalog_md_miss_api_doesn_t_exist_must.md`
- **`LAUNCH_BOUND` value exceeds 512** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_3_launch_bound_value_exceeds_512.md`
- **AscendC autogen K_TYPE single-match-per-cpp — mixing AIV and AIC kernels in one .cpp silently mis-registers all but the first** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_42_ascendc_autogen_k_type_single_match_per_cpp_mixing.md`
- **bisheng `half` macro undefined in `aic_obj` compile target — direct-include `kernel_operator.h` in .cpp** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_43_bisheng_half_macro_undefined_in_aic_obj_compile_ta.md`
- **`deploy_to_npu.sh` TARGET case statement rejects `a3-ds` and other alias targets** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_44_deploy_to_npu_sh_target_case_statement_rejects_a3.md`
- **Container `CA_cann_9_b2_kevin` is compile-only — stub CANN libraries, `aclInit` returns 100039** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_45_container_ca_cann_9_b2_kevin_is_compile_only_stub.md`
- **Explicit `kernel_operator_<adv>_intf.h` include fails — adv_api headers are pulled transitively via `kernel_operator.h`** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_46_explicit_kernel_operator_adv_intf_h_include_fails.md`
- **`ToFloat<>` static_assert "only support bfloat16_t/hifloat8_t/fp8_*/fp4_*" after BF16 guard removal [V351, port_a3_to_a5]** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_47_tofloat_static_assert_only_support_bfloat16_t_hifl.md`
- **`Exec format error: bisheng` / `OSError: [Errno 8]` — bisheng wrapper script missing shebang [V351, build-toolchain]** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_48_exec_format_error_bisheng_oserror_errno_8_bisheng.md`
- **`simt_compat.h` conflicts in NPU mode** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_4_simt_compat_h_conflicts_in_npu_mode.md`
- **Target prior-art JSON may contain duplicate bin_filename keys — regenerate a valid task-owned schema** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_50_upstream_config_json_may_contain_duplicate_bin_fil.md`
- **ops-nn-port cmake fails `Could not find a package configuration file provided by "ASC"` — ASCEND_CANN_PACKAGE_PATH not auto-exported** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_51_ops_nn_port_cmake_fails_could_not_find_a_package_c.md`
- **A3→A5 port — PR4778 ship artifacts (`<op>_apt.cpp`, `arch35/<op>.h`) placed under `kernel/` instead of `op_kernel/` → `build_ascendc.py` glob picks them up and compile fails with `unknown type name '<** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_54_a3_a5_port_pr4778_ship_artifacts_op_apt_cpp_arch35.md`
- **Pybind module import fails with a torch symbol — link libtorch_python before diagnosing ABI** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_55_pybind_module_import_fails_with_undefined_symbol_t.md`
- **Small NPU tensors (<32 bytes) trigger torch_npu Slice kernel crash 507035 — pad tiling buffer to ≥256 bytes** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_56_small_npu_tensors_32_bytes_trigger_torch_npu_slice.md`
- **Phase O5 re-measurement disagrees with worker `pass_a_runner.py` `n_pass` count → orchestrator infinite finalize→await_worker loop when worker iter_cap exhausted [V220+V351, port_a3_to_a5, orchestrato** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_59_phase_o5_re_measurement_disagrees_with_worker_pass.md`
- **`static_cast<float>(bfloat16_t)` fails in bisheng** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_5_static_cast_float_bfloat16_t_fails_in_bisheng.md`
- **ACLRT_LAUNCH_KERNEL with blockDim=0 causes ACL_ERROR_RT_PARAM_INVALID (107000) on V220** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_60_aclrt_launch_kernel_with_blockdim_0_causes_acl_err.md`
- **V220 `op_kernel/*_base.h` `#define bfloat16_t int16_t` fallback silently disables bf16 on A5 — author local A5-safe copy under `workspace/kernel/`** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_65_v220_op_kernel_base_h_define_bfloat16_t_int16_t_fa.md`
- **Wrapper input materialization required — kernel reads zero on layout-conversion tensors (mechanism provisional)** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_66_wrapper_input_materialization_required_kernel_read.md`
- **`__VEC_SCOPE__` for-loop induction variable MUST be `uint16_t`** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_67_vec_scope_for_loop_induction_variable_must_be_uint.md`
- **A kernel that owns its workspace via `GetUserWorkspace` under an `ACLRT_LAUNCH` host stub must call `SetSysWorkspaceForce(workspaceGM)` FIRST, or temporaries land out-of-range → MTE "DDR addr out of r** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_68_a_kernel_that_owns_its_workspace_via_getuserworksp.md`
- **Variable names `AT`, `BT`, `CT`, `WT` collide with AscendC builtin enums in `cce_aicore_intrinsics.h`** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_69_variable_names_at_bt_ct_wt_collide_with_ascendc_bu.md`
- **`using namespace AscendC::Simt` causes `GetBlockIdx` ambiguity** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_6_using_namespace_ascendc_simt_causes_getblockidx_am.md`
- **`FixpipeParams<float>` member-field form (`fp.nSize=`/`fp.mSize=`) does NOT compile on V220 (arch22) — use the positional `FixpipeParamsV220(...)` ctor + templated `Fixpipe<dstT, srcT, CFG_ROW_MAJOR>`** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_72_fixpipeparams_float_member_field_form_fp_nsize_fp.md`
- **A device helper called from a `__simt_vf__` kernel must be `__simt_callee__` on CANN ≥9.1.T500 — plain `__aicore__ inline` (which EC-1 prescribes) no longer suffices** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_74_a_device_helper_called_from_a_simt_vf_kernel_must.md`
- **`.ascendc_env` `A5_CANN_PATH` pointed at the toolkit symlink-root (only `latest` + `set_env.sh` symlinks, no `tools/tikcpp`) → cmake `ascendc_kernel_cmake does not exist` — point it at the complete `.** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_75_ascendc_env_a5_cann_path_pointed_at_the_toolkit_sy.md`
- **A chunked scan's cross-chunk carry fold needs a FULL `PipeBarrier<PIPE_ALL>` (not `PIPE_V`) before the wide intra-chunk scan reads it — at large state width the carry write has not drained** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_77_a_chunked_scan_s_cross_chunk_carry_fold_needs_a_fu.md`
- **SIMD elementwise kernel segfault (exit 139) with blockDim > 1 on A5 — multi-core-first diagnostic, NBLK=1 as fallback** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_78_simd_elementwise_kernel_segfault_exit_139_with_blo.md`
- **`Simt::atomicAdd` — wrong namespace** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_7_simt_atomicadd_wrong_namespace.md`
- **bare `is_same<>` unresolved in a standalone bisheng/ccec kernel TU despite `using namespace AscendC` — use `std::is_same` + `#include <type_traits>`** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_83_bare_is_same_unresolved_in_a_standalone_bisheng_cc.md`
- **uninitialized `RunInfo<true>` (isInfer) fields read UNCONDITIONALLY in `CalcS1Coord` → stack-garbage query/kv-row shift — LATENT in the shared fa_class template until an inference-mode op instantiates** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_86_uninitialized_runinfo_true_isinfer_fields_read_unc.md`
- **Missing `#include <kernel_operator.h>`** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_8_missing_include_kernel_operator_h.md`
- **Missing `namespace ascendc_ops {}` wrapper** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_9_missing_namespace_ascendc_ops_wrapper.md`
- **空 Ascend C kernel 骨架需显式标注 Vector 执行类型** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/empty_kernel_requires_explicit_vector_type.md`
- **empty分支SetBlockDim(0)与aUbFactor误式判别** · 231 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/empty_setblockdim_0_aubfactor.md`
- **EXEC_NPU_CMD 参数序：inputs → attrs → 输出 tensor 必须最后——按 msopgen 自动生成的 GetWorkspaceSize 签名对齐** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/exec_npu_cmd_arg_order_inputs_attrs_output.md`
- **UT ExecuteTestCase {0} 重载歧义需显式 vector** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/execute_test_case_brace_init_overload_ambiguity.md`
- **ExtendCfgInfo 设 jitCompile：enable/0 非法** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/extendcfginfo_jitcompile_values_invalid.md`
- **ExtendCfgInfo 无点键致打包 ValueError** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/extendcfginfo_undotted_key_valueerror.md`
- **extern 预编译 kernel 自定义包致 aclnn 161001** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/extern_kernel_aclnn_161001.md`
- **EZ3003 V2d/V2D 与 _camel_to_snake 数字拆分** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ez3003_camel_to_snake_digit_split_op_type.md`
- **EZ9999 找不到 infer 回调 — 补 IMPL_OP 注册** · 379 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ez9999_infer_impl_op.md`
- **advance_step in-place kernel FATBIN wrap** · 239 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/fatbin_wrapper_param_count_matches_opdef_input_plus_output.md`
- **FloorDiv 混用 int64 与 int 字面量推导冲突** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/floordiv_int64_int_literal_deduction_conflict.md`
- **fusion pass SetAttr output_padding 非根因** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/fusion_pass_setattr_output_padding.md`
- **Gather 双线性插值 kernel — DAV_3510 编译级配方** · 661 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gather_resize_bilinear_dav3510_recipe.md`
- **Gather 偏移表形参固定 LocalTensor<uint32_t>** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gather_srcoffset_requires_uint32_tensor.md`
- **GE 图模式 1343242270：缺 InferDataType 注册** · 298 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_1343242270_missing_inferdatatype.md`
- **GE图模式263: 自定义VF Select被禁** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_263_vf_select.md`
- **ge::DataType 为 unscoped 枚举勿手写副本** · 175 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_datatype_unscoped_enum_ut.md`
- **GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999** · 213 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_graph_tbe_dynamic_compile_blockdim_ez9999.md`
- **GE 头文件多目录布局致 GEIR 示例编译失败** · 231 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_header_multidir_include_path.md`
- **GE IR fp32 命中 te_ JIT 而非预编译 binary** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_ir_fp32_te_jit_kernel_selection.md`
- **ge.jit_compile=2图级闸门挡预编译复用** · 181 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_jit_compile_2_blocks_prebuilt_binary_reuse.md`
- **GE SetAttr 字符串属性链接失败，改用 AttrValue** · 266 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_setattr_string_undefined_reference.md`
- **GEInitialize 新 ABI 重载未导出改 AscendString** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geinitialize_cxx11_abi_undefined_ref.md`

下一页：page-021.md
