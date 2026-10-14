# other-operator-knowledge: 1261–1330 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **CANN 内置无 GRU Cell 融合反向算子（9.0.0-beta.2）** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/check_builtin_op_proto_before_porting.md`
- **CheckAttrs 误传 weight 致维度语义错读 161002** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/checkattrs_weight_as_x_161002.md`
- **CheckNotNull 占位 nullptr 致 161001 两形态** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/checknotnull_nullptr_placeholder_161001.md`
- **aclnn 返回裸 1：校验函数 bool 被当状态码** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/checkparams_bool_ret_bare_error_1.md`
- **重装算子包前清理 OPP vendors 旧安装残留** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/cleanup_vendors_before_reinstall.md`
- **CMake arch 子目录隔离 dav-3510 专属 API** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/cmake_arch_dav_3510_api.md`
- **CommonOpExecutorRun 加销毁循环 double free** · 278 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/common_op_executor_run_double_free.md`
- **Compare/Select 模板参错位编译报错** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/compare_select_template_param_order.md`
- **CPack 失败—binary/config 需 binary 聚合目标** · 190 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/cpack_binary_config_requires_binary_target.md`
- **Cube 直调 retCode=0x26：AIV 误跑 Cube API 越界** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/cube_retcode_0x26_aiv_cube_api.md`
- **Cube tile k/n 超 L1/L0 上限报 INVALID_TILE** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/cube_tile_k_n_l1_l0_invalid_tile.md`
- **custom_opp 装 vendors/ 而 ST 查 opp/vendors** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/custom_opp_install_path_vendors_layout.md`
- **Data 节点缺 index 未收集致 AddGraph 失败** · 161 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/data_node_index_inputs_addgraph_null.md`
- **DAV_2201 element-wise unary kernel 骨架 多核** · 226 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/dav2201_elementwise_unary_kernel_skeleton_relu.md`
- **def.cpp 缺 AddConfig 致 SOC_NOT_SUPPORT** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/def_cpp_addconfig_soc_not_support.md`
- **def dtype 集合与 TPL 实例化须双向同步** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/def_dtype_set_must_match_tpl_instantiation.md`
- **Depthwise 卷积 Vector Axpy 直接累加配方** · 355 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/depthwise_conv2d_vector_axpy_recipe.md`
- **AscendC 标量类型误加命名空间导致模板分发编译失败** · 248 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/device_half_bfloat16_types_are_global.md`
- **Device helper 必须保留 GM 地址空间** · 212 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/device_helper_preserves_gm_address_space.md`
- **设备侧 kernel 编译不支持 lambda 调用** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/device_kernel_no_lambda_calls.md`
- **bisheng 不接受 device 侧 sqrtf—改用向量 Sqrt** · 460 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/device_sqrt_uses_vector_api.md`
- **RegTensor 使用 c10 低精度类型导致 TypeGet 未定义** · 208 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/device_templates_use_native_types_not_torch_wrappers.md`
- **launch配置内static_cast致尖括号解析错位** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/direct_launch_config_static_cast_parse_error.md`
- **SIMT 核类型推导失败 — 补 __vector__** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/direct_launch_kernel_requires_vector_qualifier.md`
- **连续直调异常可用 stream(true) 做同环境 A/B** · 253 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/direct_launch_requires_empty_task_queue_stream.md`
- **直调优化实现忠实集成回注册工程配方** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/direct_optimized_integrate_back_registered.md`
- **DTYPE_X 未定义：dtype 宏名随输入名生成** · 234 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/dtype_macro_name_follows_spec_input_name.md`
- **Duplicate 支持类型分档：arch22 无 int8/int64** · 191 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/duplicate_dtype_support_arch22_arch35.md`
- **DynamicCompileStaticFlag 传参错位致输出全 0** · 274 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/dynamic_compile_static_arg_misalign_507015.md`
- **E80003 属性 Python 类型须与 OpDef 逐例一致** · 218 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/e80003_attr_python_type_strict_isinstance.md`
- **eager直调GetUserWorkspace恒返回保留区常量** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/eager_direct_launch_getuserworkspace_constant.md`
- **aclrtlaunch Undefined Reference (Linker)** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_10_aclrtlaunch_undefined_reference_linker.md`
- **bash bad substitution `${310P_EXTRA_LD_LIBRARY_PATH:-}` — bash identifiers cannot start with a digit; never derive env-var names from target tokens** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_120_bash_bad_substitution_310p_extra_ld_library_path.md`
- **plain `static inline` file-scope helper functions called from `__aicore__` context are dropped by the device link (ld.lld) — use SDK-form `__aicore__ inline` (non-static), with a host-side empty-`#def** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_127_plain_static_inline_file_scope_helper_functions.md`
- **an unbacked `LocalTensor` class member used as a VEC destination/source writes through a garbage UB address — silent, channel-selective corruption that neither the compiler nor the static checks flag** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_129_an_unbacked_localtensor_class_member_used_as_a_v.md`
- **`block_num` / `block_index` macro collision in parameter names** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_12_block_num_block_index_macro_collision_in_parameter.md`
- **pybind11 launcher: a `const at::Tensor&` operand rejects an explicit Python `None` (torch caster has no None conversion) — declare None-able operands `std::optional<at::Tensor>`; and `Tensor::is_npu()** · 22 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_130_pybind11_launcher_a_const_at_tensor_operand_reje.md`
- **`TQue<..., 0>` — depth must be >= 1** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_14_tque_0_depth_must_be_1.md`
- **RESERVED_WORKSPACE is an arch-conditional LADDER, not a constant** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_170_reserved_workspace_is_an_arch_conditional_ladder.md`
- **Sub-alignment chunk overwrite in compact (tightly-packed) output** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_17_sub_alignment_chunk_overwrite_in_compact_tightly_p.md`
- **Forward-overwrite data race in multi-block non-aligned DMA** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_18_forward_overwrite_data_race_in_multi_block_non_ali.md`
- **Duplicate<int8_t> has no dav-m200 overload** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_191_duplicate_int8_t_has_no_dav_m200_overload.md`
- **dav-m200 (Ascend310P3): use the FULL using namespace AscendC at kernel-namespace scope; selective using Ascend** · 352 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_192_dav_m200_ascend310p3_use_the_full_using_namespac.md`
- **__aicore__ functions forbid a direct unsigned-integral→floating static_cast ("cast between floating and unsign** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_193_aicore_functions_forbid_a_direct_unsigned_integr.md`
- **flat-index → multi-dim divmod chain: each level's remainder must use THAT level's stride, NOT the innermost (=** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_195_flat_index_multi_dim_divmod_chain_each_level_s_r.md`
- **floorf/floor unavailable in the __aicore__ kernel TU on dav-m200 (ccec compile: "undeclared identifier 'floorf** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_196_floorf_floor_unavailable_in_the_aicore_kernel_tu.md`
- **CANN 9.0.0 (dav-m200) VEC/repeat-form signatures DIFFER from newer-CANN source-kernel forms: repeat-form binar** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md`
- **220x-family entry-point _kernels.cpp: bare TPipe does not resolve; qualify AscendC::TPipe. The entry .cpp is a** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_198_220x_family_entry_point_kernels_cpp_bare_tpipe_d.md`
- **libtorch C++ API has no at::kUInt8; the uint8 dtype enum is at::kByte (matches torch.uint8/legacy ByteTensor).** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_199_libtorch_c_api_has_no_at_kuint8_the_uint8_dtype.md`
- **Missing `__aicore__` on helper function** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_1_missing_aicore_on_helper_function.md`
- **fmaf / __builtin_fmaf is host-only on dav-m200 aicore (compile "host function from aicore" + link undefined sy** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_201_fmaf_builtin_fmaf_is_host_only_on_dav_m200_aicor.md`
- **ACLRT_LAUNCH_KERNEL arg count is arch-dependent: on dav-m200 (__CCE_AICORE__=200) the launcher takes NO tiling** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_203_aclrt_launch_kernel_arg_count_is_arch_dependent.md`
- **fp32→fp16 Cast on dav-m200 supports ONLY CAST_NONE/CAST_ODD** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_205_fp32_fp16_cast_on_dav_m200_supports_only_cast_no.md`
- **element-level input strides from the host are ALREADY in element units (each = product of lower input dims); a** · 110 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_206_element_level_input_strides_from_the_host_are_al.md`
- **dav-m200 rejects half-typed arithmetic in ANY instantiated branch** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_207_dav_m200_rejects_half_typed_arithmetic_in_any_in.md`
- **lambdas are HOST-ONLY under bisheng** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_209_lambdas_are_host_only_under_bisheng.md`
- **at::Tensor → py::object return needs explicit py::cast** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_210_at_tensor_py_object_return_needs_explicit_py_cas.md`
- **runtime if on a templated kernel's compile-time dtype flag type-checks BOTH branches** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_211_runtime_if_on_a_templated_kernel_s_compile_time.md`
- **half is a GLOBAL typedef, NOT AscendC::half** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_212_half_is_a_global_typedef_not_ascendc_half.md`
- **on the CANN 9.1.0 ascendc cmake flow the generated host launch stub is aclrtlaunch_<kernel> returning uint32_t** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_214_on_the_cann_9_1_0_ascendc_cmake_flow_the_generat.md`
- **python setup.py bdist_wheel ships an EMPTY wheel unless the cmake-built .so is staged into build/lib.<plat>** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_215_python_setup_py_bdist_wheel_ships_an_empty_wheel.md`
- **SortConfig must be global constexpr for template NTTP** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_24_sortconfig_must_be_global_constexpr_for_template_n.md`
- **`Duplicate<int8_t>` has no dav-m200 overload — compile-time `SupportType` static_assert; duplicate through an int16 view (the vendor's own idiom)** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_259_duplicate_int8_t_has_no_dav_m200_overload_compil.md`
- **Duplicate / VEC op on non-aligned UB offset → runtime error 340** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_26_duplicate_vec_op_on_non_aligned_ub_offset_runtime.md`
- **ops-nn UT `add_modules_ut_sources` splits test modules by FILENAME GLOB — a test cpp mixing tiling + infershape + aclnn concerns links only against its glob-module's libs → undefined references at ld ** · 22 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_281_ops_nn_ut_add_modules_ut_sources_splits_test_mod.md`
- **case-runner template `bestUs` unit-mix — `ms < bestUs*1000.0` compares ms against us (almost always true) → the timing field reports the LAST repeat, not the min [ops-math case-pack runner template — ** · 22 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_282_case_runner_template_bestus_unit_mix_ms_bestus_1.md`
- **`CREATE_EXECUTOR` (opdev executor-registration macro) needs an explicit `#include "opdev/make_op_executor.h"` — it arrives transitively via `cast.h` in some ops and not others, so a new op that omits ** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_283_create_executor_opdev_executor_registration_macr.md`
- **`--cann-op-gen <repo>/<cat>/<op> --resume` re-seeds `.opgen_state.json` in BENCHMARK shape — the resume child-command rebuild only replays port_pytorch/port_cuda, so an ops_math_commercial workspace l** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_284_cann_op_gen_repo_cat_op_resume_re_seeds_opgen_st.md`
- **`<c10/npu/NPUStream.h>` is not on the pybind TU include path on this build chain — use `<torch_npu/csrc/core/npu/NPUStream.h>`** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_288_c10_npu_npustream_h_is_not_on_the_pybind_tu_incl.md`
- **Vendor benchmark source `32_AdaptiveAvgPool3d.py get_input_groups()` crashes on every load — `for idx, case in cases:` unpacks single-key-per-line JSON dicts** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_289_vendor_benchmark_source_32_adaptiveavgpool3d_py.md`

下一页：page-020.md
