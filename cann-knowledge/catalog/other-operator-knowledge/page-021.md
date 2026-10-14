# other-operator-knowledge: 1401–1470 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **GEIR check-route标记缺失：CANN日志差异验V2路由** · 145 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_check_route_cann_v2.md`
- **GEIR 路由 DESCRIPTOR_ERROR：适配器端口未对齐 OpDef** · 289 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_descriptor_error_adapter_port_mismatch.md`
- **GEIR 示例混用 ACL 清理致退出段错误** · 192 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_example_acl_ge_finalize_segfault.md`
- **GEIR负向SUPPRESSED：InferDataType缺输出dtype校验** · 194 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_inferdatatype_output_dtype_validation.md`
- **GEIR 路由 NOT_UNIQUE：RunGraph 失败非歧义** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_route_not_unique_probe_rungraph_failure.md`
- **GEIR 图模式 V2 推理注册与构建接线配方** · 425 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_v2_infer_shape_datatype_impl_op_wiring.md`
- **GenSimplifiedKey 签名 UINT32 不在 ge 命名空间** · 197 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gensimplifiedkey_uint32_global_scope.md`
- **gert 属性读取返回 const 指针须解引用** · 283 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gert_attrs_getters_return_const_pointer.md`
- **gert 无 TensorDesc：GetInputDesc 用 auto 接** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gert_no_tensordesc_getinputdesc_auto.md`
- **gert::Shape 无 ToString，日志用 GetDimNum** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gert_shape_no_tostring_use_getdimnum.md`
- **StorageShape 需解包为 Shape 再用维度 API** · 268 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gert_storageshape_unwrap_dim_api.md`
- **GetBlockNum对齐numCores未修复GE路径EZ9999** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/getblocknum_align_numcores_not_fix_ge_ez9999.md`
- **GetCoreNumAiv运行时少于INI核数，分支边界漂移** · 226 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/getcorenumaiv_runtime_below_ini_core_count.md`
- **GetOutputViewTensor 私有致 op_host 编译失败** · 70 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/getoutputviewtensor_op_host.md`
- **GetPhyAddr const __gm__ 返回不可去 const** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/getphyaddr_const_gm_const.md`
- **GetTPipePtr 错加 AscendC 命名空间导致编译失败** · 419 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gettpipeptr_is_global_function.md`
- **GetUbBlockSize 返回 32 字节而非元素数** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/getubblocksize_bytes_not_elements.md`
- **__global__ 核函数 extern C 失效致符号未定义** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/global_vector_stub_ignores_extern_c_mangled_name.md`
- **GlobalTensor 无 GetAddr，辅助函数收引用** · 333 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/globaltensor_no_getaddr_pass_reference.md`
- **GM_ADDR host侧cast被拒或静默剥离属性** · 242 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gm_addr_cast_from_void_uses_toolchain_cast.md`
- **CCE 禁止 __gm__ 指针转 void*，打印用 uint64_t** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gm_pointer_cast_generic_banned.md`
- **GMM A8W4 561000 缺 binary entry 无 JIT 回退** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gmm_a8w4_561000_binary_entry_jit.md`
- **带主元 Thomas 求解器 SIMT 直访 GM kernel** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gtsv2_partial_pivoting_thomas_simt_direct_gm_kernel.md`
- **奇数 H 致 B 视图 UB 地址非 32B 对齐** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/h_b_ub_32b.md`
- **非对齐hiddenSize致507035需全链路8对齐** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/hidden_size_non_multiple_8_alignment_507035.md`
- **自定义算子host的OP_LOGI不进stdout与plog** · 214 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/host_op_logi_stdout_plog.md`
- **Host workspace 需先 CPU 暂存再 H2D** · 228 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/host_workspace_uses_cpu_staging_before_h2d.md`
- **向量函数内HWLoop归纳变量必须uint16_t** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/hwloop_induction_variable_uint16.md`
- **BiSheng bf16 类型转换失败——隔离模板实例化** · 338 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/if_constexpr_excludes_unsupported_bf16_cast.md`
- **bf16 未编译剔除触发 backend Cannot select** · 171 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/if_constexpr_excludes_unsupported_bf16_compare.md`
- **IMPL_OP_INFERSHAPE 双展开重定义须链式注册** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/impl_op_infershape_double_expansion_redefinition.md`
- **InitOutput GM 专用，UB 清零用 Duplicate** · 102 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/initoutput_gm_ub_duplicate.md`
- **手改 kernel JSON int64Mode 无效且被重装覆盖** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/int64mode_json_patch_ineffective.md`
- **interleaved RoPE 分支 kernel 完整实现配方** · 307 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/interleaved_rope_branch_kernel_recipe.md`
- **IR 属性名误改判别：IR 名与 aclnn ABI 名两层对照** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ir_attr_name_vs_aclnn_abi_name_layers.md`
- **K_MAX_SHAPE_DIM=0须先于首个include定义** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/k_max_shape_dim_zero_before_first_include.md`
- **kernel entry not implement — 入口须随算子名** · 306 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_entry_snake_case_follows_op_type.md`
- **dav_3510 核数用 GetBlockNum 非 GetBlockDim** · 117 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_getblocknum_not_getblockdim.md`
- **kernel_printf.h 不存在且编译错误被构建吞掉** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_printf_header_missing_error_swallowed.md`
- **直调 kernel-type 裸赋值标记无效** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_type.md`
- **ViewCopy/Contiguous 未声明：在 l0op 命名空间** · 117 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/l0op_contiguous_viewcopy_namespace.md`
- **l0op::Reshape 变通 int64 AR 触发 561103 回退** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/l0op_reshape_int64_ar_workaround_561103.md`
- **L1/L0 tensor 须用 TPosition 构造非 SetAddr** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/l1_l0_tensor_tposition_setaddr.md`
- **L2_DFX_PHASE_1 只收 3 参：多输入并 DFX_IN** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/l2_dfx_phase_1_multi_input_single_dfx_in.md`
- **ld.lld cannot open _mix_aiv 报错判别** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ld_lld_cannot_open_mix_aiv.md`
- **UT 缺 libascend_hal.so：source set_env.sh** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/libascend_hal_runpath_indirect_dep_set_env.md`
- **l0op::Contiguous未定义符号致安装校验失败** · 372 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/libcust_opapi_l0op_contiguous_undefined_symbol.md`
- **MIX kernel 507015：AIC L1+AIVec 双侧异常诊断** · 245 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mix_507015_aic_aiv_dual_side_anomaly.md`
- **Mix合图编译超时致 MLA decode 不可用** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mix_mla_decode.md`
- **MIX模式TPL DATATYPE_DECL不实例化改UINT_DECL** · 215 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mix_mode_tpl_datatype_decl_not_instantiated.md`
- **混合 SIMT 启动边界用 GM_ADDR 承接地址空间** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mixed_simt_launch_boundary_uses_gm_addr.md`
- **CANN 9.0.1 SIMT 入口类型无法派生—改混合 VF** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mixed_simt_requires_vector_shell_and_simt_vf.md`
- **多 vendor 环境 find 盲选错 opapi 库致符号错** · 222 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/multi_vendor_find_head1_wrong_opapi_lib.md`
- **kernel 内 numeric_limits 不可用—AscendC 替代** · 165 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/numeric_limits_kernel_use_ascendc.md`
- **README-driven doc updates** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_10_readme_driven_doc_updates.md`
- **Hardware doc said Ascend910_9589 but actual SOC is Ascend950PR_9589** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_11_hardware_doc_said_ascend910_9589_but_actual_soc_is.md`
- **Use file-CONTENT signal, not file-EXISTS-AND-NONEMPTY signal, for build-status wait loops** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_122_use_file_content_signal_not_file_exists_and_nonemp.md`
- **CANN API surface shrinks across versions — check API availability before using A5-KB patterns on older CANN [V220, CANN 9.0.0]** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_127_cann_api_surface_shrinks_across_versions_check_api.md`
- **V220 `Abs<half>` / `Abs<bfloat16_t>` requires raw `__ubuf__*` TBuf pointer — use Cast→fp32 TBuf→Abs→Cast back for fp16/bf16 abs [V220]** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_129_v220_abs_half_abs_bfloat16_t_requires_raw_ubuf_tbu.md`
- **V220 A3 container has a libascendcl.so SONAME conflict between CANN 9.0.0 and ascend-toolkit 8.3** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_130_v220_libascendcl_soname_conflict.md`
- **STOP and search existing skills before inventing workarounds** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_13_stop_and_search_existing_skills_before_inventing_w.md`
- **Host-side W8 staging without a working kernel is misleading scaffolding [V351, port_a3_to_a5]** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_140_host_side_w8_staging_without_a_working_kernel_is_m.md`
- **Target `op_kernel/arch35/` reference is advisory prior art — never skip task-owned generation or independent truth [arch22→arch35]** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_141_target_op_kernel_arch35_reference_is_advisory_prio.md`
- **MicroScaling (MX) format — mxfp8 / mxfp4 cube path with fp8_e8m0_t shared scale [V351, microscaling]** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_145_microscaling_mx_format_mxfp8_mxfp4_cube_path_with.md`
- **CastTrait constants for A5 narrow-float quantization — full reference table [V351, microapi-cast]** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_146_casttrait_constants_for_a5_narrow_float_quantizati.md`
- **arch22 → arch35 (220x → 351x) hardware delta reference — single-pointer authoritative summary [V351, hardware-arch]** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_149_arch22_arch35_220x_351x_hardware_delta_reference_s.md`
- **`using namespace AscendC::Simt;` causes GetBlockIdx ambiguity** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_14_using_namespace_ascendc_simt_causes_getblockidx_am.md`
- **SIMT helper APIs — AtomicAdd / UintDiv / `__local_mem__` UB access [V351, simt-l3]** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_151_simt_helper_apis_atomicadd_uintdiv_local_mem_ub_ac.md`
- **L4 STRUCTURAL signature — kw cannot single-shot port these; route to researcher FIRST [V351, port_a3_to_a5_escalation]** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_156_l4_structural_signature_kw_cannot_single_shot_port.md`
- **Safety-net file-name coupling is an architectural anti-pattern — every new op-gen mode MUST emit canonical entry-point names, never invent its own [V351+V220, ALL_MODES, arch_invariant]** · 93 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md`

下一页：page-022.md
