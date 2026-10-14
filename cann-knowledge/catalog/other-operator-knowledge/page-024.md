# other-operator-knowledge: 1611–1680 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **白盒pytest走aten内置绕过自定义aclnn** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pytest_aten_aclnn.md`
- **qgmm_mx FP4 fatbin 缺失致 107000** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/qgmm_mx_fp4_fatbin_107000.md`
- **rank>2 展平批块搬运 gather 配方** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rank_over_2_flatten_batch_block_gather_recipe.md`
- **共享只读 CANN 上 .run 安装权限失败重定向** · 163 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/readonly_cann_opp_install_redirect.md`
- **f32→s64 Cast 使用 CAST_NONE 触发 vcvt 静态断言** · 163 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_cast_f32_to_s64_uses_supported_rounding.md`
- **Reg Cast 默认 UNKNOWN mask 被拒 — 显式 ZEROING** · 303 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_cast_requires_explicit_zeroing_trait.md`
- **CastTrait 文件域声明触发命名空间编译失败** · 328 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_casttrait_define_inside_simd_vf.md`
- **Reg::Cast 的 trait 需要静态存储期** · 308 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_casttrait_requires_static_storage.md`
- **Reg Compare 的 CMPMODE 命名空间导致编译失败** · 250 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_compare_cmpmode_is_in_ascendc.md`
- **Reg CreateMask 缺类型参数导致编译失败** · 250 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_createmask_requires_explicit_type.md`
- **Duplicate 广播枚举与模板顺序不兼容** · 246 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_duplicate_highlow_precedes_merge_mode.md`
- **LoadDist 错用 DIST_NORM_B32 导致编译失败** · 153 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_load_and_store_use_distinct_dist_enums.md`
- **Reg LoadAlign 不接受 const __ubuf__ 指针** · 404 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_loadalign_requires_nonconst_ubuf_pointer.md`
- **LoadAlign/StoreAlign 源类型须匹配 __ubuf__ 指针** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_loadalign_source_type_matches_ubuf_pointer.md`
- **Reg::Move 三参带 mask 且不接受 const** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reg_move_add_mask_non_const.md`
- **注册工程转直调split构建与eager启动配方** · 346 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/registered_to_direct_launch_split_recipe.md`
- **RegTensor 累加须用 Add 而非 Adds** · 220 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/regtensor_add_uses_add_not_adds.md`
- **ReinterpretCast缺template关键字编译失败被旧二进制掩盖** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/reinterpret_cast_dependent_template_keyword.md`
- **多层 weight_ih 须 flat 无补零拼接防 161002** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rnn_weight_ih_concat_flat_layout_161002.md`
- **RoPE cos/sin BSND 与 BNSD 反算配对** · 237 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rope_cos_sin_bsnd_bnsd_stride_pairing.md`
- **RoPE half BSND kernel 平面分解实现** · 434 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rope_half_bsnd_planar_kernel_recipe.md`
- **.run 安装 chmod 噪音与建目录权限失败判别** · 107 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/run_install_chmod_noise_vs_vendor_mkdir_denied.md`
- **.run 重装只读 scripts 目录 rm 权限拒绝** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/run_package_reinstall_readonly_scripts_chmod.md`
- **RunOpApi 延迟回调需按值捕获局部 launch vector** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/runopapi_callbacks_capture_launch_state_by_value.md`
- **同进程先跑正常用例后属性校验被跳过** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/same_process_attr_validation_order_sensitive_skip.md`
- **OpDef 骨架 ge::InferShape 无字面，绑定走 IMPL_OP** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/setinfershape_ge_notation_not_literal_symbol.md`
- **SetLocalWorkspace 507015：transLength 恒 0** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/setlocal_workspace_507015_translength_zero.md`
- **SetOpSupportInfo 回调在 aclnn 直调不被调用** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/setopsupportinfo_aclnn_not_called.md`
- **标量输入被赋非标量 shape 致 561002 判别** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/shape_561002.md`
- **simd_vf 互调导致 helper 编译失败** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/simd_vf_helpers_use_simd_callee.md`
- **__simd_vf__ if constexpr 致 hoisted 报错** · 325 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/simd_vf_if_constexpr_lowering_banned_exit_70.md`
- **SIMT API 未进入 AscendC 命名空间导致编译失败** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/simt_api_is_in_ascendc_simt_namespace.md`
- **SIMT 内读块号致 Bisheng 崩溃—移到核入口** · 228 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/simt_move_block_metadata_to_vector_shell.md`
- **SIMT __ubuf__ 静态数组 UB 占用不随分支释放** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/simt_ubuf_ub.md`
- **SIMT helper 误标 __simt_vf__ 导致编译失败** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/simt_vf_helpers_use_simt_callee.md`
- **跨 slice 批量 vector 源地址非 32B 对齐触发 507035** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/slice_vector_32b_507035.md`
- **950 无 Sort API：unique 改自实现排序** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/sort_api_unavailable_950_self_radix_sort.md`
- **稀疏注意力 Vector 主路径 kernel 全 UB 实现** · 918 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/sparse_attention_vector_main_path_full_ub.md`
- **split.axis 未随 PadAndSqueeze 偏移致 507035** · 158 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/split_axis_padandsqueeze_507035.md`
- **split-load 多 tile UB 循环 kernel 实现骨架** · 279 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/split_load_multi_tile_ub_loop_kernel.md`
- **UT 构造 StorageShape 须用 initializer_list** · 316 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/storage_shape_ctor_initializer_list_only.md`
- **负步长 end 开区间致输出 shape 不符报 561103** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/strided_slice_negative_stride_end_open_interval_561103.md`
- **输出恒为哨兵位型：入口分支stub未接线** · 198 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/stub_branch_sentinel_output_entry_wiring.md`
- **TBE fusion 不扫自定义 vendor：三种修复均无效** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbe_fusion_binary_scan_list_excludes_vendor.md`
- **kernel 类成员裸 TBuf 须写 TBuf<> 空模板实参** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbuf_class_member_requires_empty_template_args.md`
- **TBuf Get 提前到搬运前致 kernel 间歇挂死** · 214 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbuf_get_before_copy_hangs_kernel.md`
- **独立 TBuf.Get 返回重叠 UB 地址 dav-2201** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbuf_get_ub_dav_2201.md`
- **TBuf 不能直接传给 LocalTensor 形参** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tbuf_gets_typed_localtensor_before_helper_call.md`
- **TEFUSION二进制匹配走全局OPP非staging** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tefusion_binary_search_uses_global_opp_not_staging.md`
- **x86_64 主机模板 aarch64 硬编码两处失效** · 334 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/template_aarch64_hardcode_x86_64_host.md`
- **复制模板工程改名的四形态映射与五项门禁** · 163 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/template_project_rename_gates.md`
- **Tensor operator[] 临时值不能绑定非 const 引用** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tensor_subview_cannot_bind_nonconst_lvalue_ref.md`
- **三元返回 B16 让 cast 失效触发 i64 setcc** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ternary_b16_return_penetrates_float_cast.md`
- **自定义算子内 .to(FP8) 多流跨流竞争 banned** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/to_fp8_banned.md`
- **TopK 高阶 API 在 DAV_3510 编译可用（产品表未列）** · 279 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/topk_highlevel_api_dav3510_compile_verified.md`
- **TopK 整数 dtype 走 fp32 排序 Cast 通路配方** · 364 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/topk_integer_dtype_cast_sort_recipe.md`
- **TORCH_LIBRARY 多输出返回须 std::tuple 非 vector** · 242 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/torch_library_multi_output_return_tuple_not_vector.md`
- **Meta 校验误传 data 致 ACLGraph 浮点全失败** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/torch_meta_param_misalignment_aclgraph.md`
- **torch_npu 只保留一族 ACL 头文件** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/torch_npu_uses_single_acl_header_family.md`
- **直调TPipe kernel同步507015：AIVEC magic补丁** · 249 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tpipe_eager_launch_507015_aivec_magic_patch.md`
- **TPL_BOOL_DECL 取值须整型 0/1 非 false/true** · 191 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tpl_bool_decl_requires_integer_literals.md`
- **TilingKey 声明与 kernel 模板四处签名须一致** · 174 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tpl_decl_kernel_template_signature_mismatch.md`
- **TPL 双 bool 模板入口与 sub-kernel 实例化验证** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tpl_template_entry_subkernel_instantiation_verify.md`
- **TQue AllocTensor 作临时分配致 kernel 挂起** · 213 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_alloctensor_temp_alloc_hangs.md`
- **TQue 位置参数须用 AscendC::TPosition 枚举** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_ascendc_tposition.md`
- **TQue 重构不修复 at::zeros 跨 stream 覆盖** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_at_zeros_stream.md`
- **TQue 输入未释放导致循环 AllocTensor 卡死** · 290 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_dequeue_must_end_with_freetensor.md`
- **ACL 507035 — 空 TQue DeQue 触发 B16 路径异常** · 277 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_dequeue_requires_prior_enqueue_507035.md`
- **TQue 单核稳定排序去重 unique kernel 骨架** · 333 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_single_core_stable_sort_dedup_kernel_skeleton.md`
- **TQue tensor 所有权只能 DeQue 一次** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_tensor_ownership_single_dequeue.md`

下一页：page-025.md
