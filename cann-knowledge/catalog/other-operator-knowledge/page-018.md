# other-operator-knowledge: 1191–1260 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **aclnn_exclude 头文件自动生成覆盖源码致签名不符** · 248 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_exclude.md`
- **aclnn 显式 l0op::Contiguous 支持非连续输入** · 121 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_explicit_l0op_contiguous_input.md`
- **aclnn 全绿但 GE 图模式 error 263 分层判别** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_ge_error_263.md`
- **ACLNN GetWorkspaceSize 缺 L0 调用致输出全零** · 161 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_l2_get_workspace_size_requires_l0_launcher_list.md`
- **L2 传 L0 输出张量需 const_cast** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_l2_l0_output_const_cast.md`
- **多输出算子 aclnn 适配器签名须逐参对齐生成头** · 244 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_multi_output_adapter_signature_autogen_header.md`
- **aclnn bfloat16.h 内嵌 fp16_t.h 编译失败** · 149 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_opdev_bfloat16_h_nested_include_path.md`
- **ACLNN 507035 — opType 与内置 ops_legacy 同名碰** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_optype_collision_built_in_ops_legacy_507035.md`
- **aclnn 冒烟测试以 autogen 生成头为签名源** · 520 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_signature_source_autogen_header.md`
- **自定义算子 ACLNN 独立示例工程搭建配方** · 376 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_standalone_example_project_recipe.md`
- **aclnn String 属性 char* 传 const 编译失败** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_string_attr_char_ptr_const_compile.md`
- **ACLNN_SUCCESS 公开头未导出须自定义宏** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_success_not_in_public_headers.md`
- **aclDestroyTensorList 连带销毁成员致 double free** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_tensorlist_destroy_members_double_free.md`
- **aclnn 测试链接的是 OPP 安装目录的 libcust_opapi.so** · 194 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_test_links_installed_vendors_opapi.md`
- **aclTensor 类型在 aclnn/acl_meta.h 不在 acl.h** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_types_require_acl_meta_h.md`
- **aclnn 无条件 Contiguous 致连续大张量挂起** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_unconditional_contiguous_large_tensor_hang.md`
- **变长 TensorList 输出 aclnn 的 PyTorch 适配** · 303 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_variable_output_tensorlist_pytorch_adapter.md`
- **ACLNNTYPE aclnn autogen 与手写同名符号冲突** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnntype_aclnn_autogen.md`
- **aclrtFinalize 不存在，收尾用 aclFinalize** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclrtfinalize_undeclared_use_aclfinalize.md`
- **ACL 无 aclGetTensorAddr，D2H 直传 Malloc 指针** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/acltensor_no_device_addr_getter.md`
- **aclTensor 无 IsContiguous 成员（CANN 9.1.0）** · 208 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/acltensor_no_iscontiguous_op_check_null.md`
- **aclTensorList 无 GetTensor，用 operator[]** · 238 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/acltensorlist_api_surface_no_gettensor.md`
- **AscendC 算子新增输入 dtype 五处联动清单** · 349 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/add_input_dtype_five_place_linkage_checklist.md`
- **AddConfig 产品变体名致 SoC 未配置构建失败** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/addconfig_product_variant_name_invalid.md`
- **Adds 写输出队列张量挂起 — 改写 TBuf** · 185 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/adds_dst_output_queue_tensor_hang.md`
- **__aicore__ 内 std::min/max 编译失败** · 409 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aicore_avoid_host_std_minmax.md`
- **aicore 禁止 const char* cast 到 __gm__ T*** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aicore_const_char_cast_gm_t.md`
- **aicore函数内double字面量触发编译错** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aicore_double_literal_promotion_banned.md`
- **核函数调用未标__aicore__的纯算术辅助函数** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aicore_helper_requires_device_qualifier.md`
- **__aicore__ 禁止 void* 转 __ubuf__ 桥接** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aicore_void_ubuf.md`
- **AICPU launcher 缺 OP_ATTR_NAMES 编译错** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aicpu_launcher_op_attr_names.md`
- **arch35 后端 64 位 div/mod 报 bitwidth>32** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_64_div_mod_bitwidth_32.md`
- **arch35 __aicore__ 调 LocalMemBar 报 507035** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_aicore_localmembar_507035.md`
- **arch35 Arange/Adds 拒 uint32，int32 生成重解释** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_arange_adds_uint32_banned.md`
- **arch35 Cube 内联函数误入 Vector 通道编译失败** · 181 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_cube_intrinsics_split_core_guard.md`
- **arch35 GetValue/SetValue 标量bypass禁用** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_getvalue_setvalue_bypass.md`
- **arch35 kernel 上板报 507035 实为 arch 不匹配** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_kernel_507035_arch.md`
- **arch35 小 per-channel 输出读旧值判别与规避** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_per_channel.md`
- **arch35 分块 Select 掩码偏移非 32B 对齐崩溃 507035** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_select_32b_507035.md`
- **ASC_VDEQ_SIZE 未声明 — 勿直引 vector_compute.h** · 122 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/asc_simd_umbrella_header_for_c_api_symbols.md`
- **dim3 未声明：显式引入 SIMT 公共头** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/asc_vf_call_launch_config_requires_simt_header.md`
- **asc_vf_call VF内非对齐访存致507035** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/asc_vf_call_unaligned_access_507035.md`
- **ASCEND350 非 SocVersion 枚举成员致编译失败** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascend350_socversion.md`
- **Ascend950 DMA 32B 块最小传输粒度** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascend950_dma_32b_minimum_block_granularity.md`
- **Ascend950 Reg 命名空间无 Erf/Tanh 封装** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascend950_reg_erf_tanh.md`
- **Ascend950PR 的 soc_version 确认为 ascend950** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascend950pr_soc_version_ascend950.md`
- **AscendC:: 前缀致全局符号编译失败** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc.md`
- **ASCENDC_CUBE_ONLY 入口须 __aicore__** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_cube_only_aicore.md`
- **设备侧头文件用 kernel_operator.h 而非 AscendC.h** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_h_not_found_use_kernel_operator_h.md`
- **AscendC Log签名与设计文档不符：按已装头文件核对** · 289 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_log_signature_design_doc_mismatch.md`
- **ASCENDC_TPL_DATATYPE_SEL 须每块单 dtype** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_datatype_sel_single_dtype_per_block.md`
- **kernel.h 须 include struct.h 才能识别 TPL** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_kernel_h_must_include_struct_h.md`
- **ASCENDC_TPL 宏未定义，须用 ASCENDC_TPL_SEL** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_macro_undefined_use_tpl_sel.md`
- **缺 ASCENDC_TPL_SEL 致运行时分发静默 no-op 跳过** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_sel_no_op.md`
- **AscendString 不能从 std::string 隐式构造** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendstring_no_std_string_ctor_map_init.md`
- **workspace 与 aclnn 保留名冲突：按孪生先例改名 storage** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aten_workspace_name_collides_aclnn_reserved.md`
- **autogen 头缺 TensorList/GetOutputCount 声明** · 163 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/autogen_header_tensorlist_getoutputcount_missing.md`
- **裸 make 不编 kernel binary 且默认按 910b 编译** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bare_make_no_kernel_binary_soc_910b.md`
- **bf16 枚举实名 DT_BF16：全称 DT_BFLOAT16 不存在** · 293 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bf16_enum_name_dt_bf16.md`
- **binary.json bin_filename 空致多 dtype kerne** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/binary_json_bin_filename_empty_overwrites_multi_dtype_o.md`
- **device 侧代码无法使用 lambda 表达式：lambda 无法操作带 device 标记的 API——把函数体提升为 __aicore__ inline 成员函数** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bisheng_device_lambda_no_device_objects.md`
- **bisheng 动态 shape 编译段错误 — extern C + DTYPE** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bisheng_extern_c_dtype_dynamic_shape_segfault.md`
- **bisheng 单模板跨字宽分支 segfault 判别** · 102 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bisheng_segfault.md`
- **BiSheng 不重编 TPL_UI_LIST 第二实例化** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bisheng_tpl_ui_list_second_instantiation_stale.md`
- **C1 Cube FP32 输出阻断 950 Mix合图 CV 通路** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/c1_cube_fp32_950_mix_cv.md`
- **C-API asc_simd.h 编译缺 include 根** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/c_api_asc_simd_h_include.md`
- **CastTrait 按值 NTTP 在 C++17 编译失败** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/casttrait_nttp_uses_reference_in_cxx17.md`
- **Reg CastTrait RoundMode 命名空间编译失败** · 381 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/casttrait_roundmode_is_in_ascendc.md`
- **__ubuf__模板成员辅助函数编译失败改宏** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ccec_member_template_ubuf_expected_unqualified_id.md`
- **CeilDiv 反推组数过冲容量阈值，UT 期望按取整方向写** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ceildiv_group_count_overshoots_capacity.md`

下一页：page-019.md
