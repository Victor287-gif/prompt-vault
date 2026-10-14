# tiling: 351–420 / 686

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **非连续 TilingKey 死代码：三层连续化拦载** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/noncontig_tilingkey_dead_path_three_layers.md`
- **arch22->arch35 port: peer-tiling-registry edits when the op-family shares a tiling-template library** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_157_peer_tiling_registry_edits.md`
- **Host↔device tiling struct layout mismatch from mixed-width fields — use explicit packing** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_178_host_device_tiling_struct_layout_mismatch_from_mix.md`
- **An aclnn tiling-error code (561002 class) is a HOST tiling-function REJECTION, not a missing-kernel-binary — read `ASCEND_GLOBAL_LOG_LEVEL=3` FUNC/FILE/LINE first; AND before concluding "the vendor op** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_219_an_aclnn_tiling_error_code_561002_class_is_a_host.md`
- **AscendC TilingData: struct-def / host-getter / kernel-register must all use the SAME registration style** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_268_tilingdata_two_registration_styles_mismatch.md`
- **host-tiling-bound high-level APIs (SoftmaxGrad, any primitive whose Compute path needs a host TilingFunc) are** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_414_host_tiling_bound_high_level_apis_softmaxgrad_an.md`
- **GM tiling struct cannot be copied wholesale — must scalar-read each field** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_77_gm_tiling_struct_cannot_be_copied_wholesale_must_s.md`
- **OP_ADD 占位 TilingParse 先注册挤掉自定义回调** · 201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/op_add_placeholder_tilingparse_precedence.md`
- **optiling 内调 TilingFunc 须引 AscendC 空间** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/optiling_using_namespace_ascendc.md`
- **CMake `COMPUTE_UNIT` ≠ `TILING_DIR` list length → `find_value_by_key` FATAL_ERROR [V351, build-system]** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_30_cmake_compute_unit_tiling_dir_list_length_find_val.md`
- **SIMT DCache shares UB → tiling MUST reserve 40KB or risk silent UB OOB [V351, simt-tiling]** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_32_simt_dcache_shares_ub_tiling_must_reserve_40kb_or.md`
- **host UT 链接 PlatformAscendC 需静态库+dlog** · 242 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/platformascendc_host_ut_link_static_tiling_api.md`
- **移植tiling误拒rank=0标量输入** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ported_tiling_rank0_scalar_rejected.md`
- **rebase 引入新 kernel 文件旧 tiling 类型致编译失败** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rebase_kernel_tiling.md`
- **507035 host tiling workspace 缺 RESERVED** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/register_none_tiling_reserved_workspace_507035.md`
- **REGISTER_NONE_TILING 致 UnboundLocalError** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/register_none_tiling_unboundlocalerror.md`
- **TilingData 注册核验：strings .o 查 mocker 符号** · 163 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/register_tiling_default_kernel_entry_mocker_probe.md`
- **REGISTER_TILING_DEFAULT 取小档致 NaN** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/register_tiling_default_nan.md`
- **REGISTER_TILING_DEFAULT 多 struct 冲突** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/register_tiling_default_struct.md`
- **softplus_v2_grad tiling rank 校验拒标量广播** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/softplus_v2_grad_tiling_rank.md`
- **框架从 op_tiling/lib/（而非 op_api/lib/）加载 host tiling 库——残留旧 opmaster 伪装成 GetTilingData null** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/stale_op_tiling_opmaster_get_tiling_data_null.md`
- **旧算子包遮蔽工程模板致新 TilingKey 不识别** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/stale_vendor_pkg_shadows_tiling_key_template.md`
- **Stub tiling padding 不解 aclnn Segfault** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/stub_tiling_padding_aclnn_segfault.md`
- **sub-kernel 后缀 _N 是 tilingKey 位编码非序号** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/sub_kernel_n_tilingkey.md`
- **TCubeTiling 双定义：kernel 侧禁宿主类型与头** · 334 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tcube_tiling_host_kernel_dual_definition.md`
- **宏生成 tiling 结构容量须实测 sizeof** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tcube_tiling_sizeof_probe_buffer_capacity.md`
- **te_fusion 未调 TilingPrepare 致 GEIR FPE** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/te_fusion_tilingprepare_geir_fpe.md`
- **TILING_DATA_DEF 宏须包在 optiling 命名空间** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_data_def_macros_need_optiling.md`
- **TILING_DATA_DEF 迁移 rebase 字段失配判别** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_data_def_rebase.md`
- **TilingData加成员函数致507035** · 204 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_data_member_functions_507035.md`
- **tiling扩A超UB致507035：UT小核数掩盖** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_expand_over_ub_corenum_507035.md`
- **共享 tiling.h 宏范式 DT 宏冲突改纯 POD** · 190 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_h_macro_dt_conflict_plain_pod.md`
- **TILING_KEY_IS 须数值宏，constexpr 分支不注册** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_key_is_requires_numeric_macro.md`
- **attrs 分支算子 TILING_KEY_IS 单 binary 分发** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_key_is_runtime_dispatch_single_binary.md`
- **tilingKey 编码与 TPL 位打包不一致致路由错误** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_key_tpl_bit_encoding_mismatch.md`
- **手动 tiling 头遮蔽自动生成类致 no member 判别** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_no_member.md`
- **OpDef SetTiling 与 IMPL_OP_OPTILING 二选一** · 298 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_registration_settiling_vs_impl_op_optiling.md`
- **tiling 派生参数须补范围守卫防 UB 溢出回绕** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_shape_param_range_guard_ub_uint32.md`
- **RmsNormTiling 与 CANN 内置类型冲突 — 改名隔离** · 189 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_type_name_does_not_shadow_cann_builtin.md`
- **Tiling 属性用 GetAttrs 而非 GetAttrsPointer** · 272 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingcontext_getattrs_not_getattrspointer.md`
- **TilingData 编入构建四探针：opParaSize=8+sizeof** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_compiled_into_build_four_probes.md`
- **TilingUT 悬垂指针—CreateCap 缓冲需 fixture 持有** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_createcap_dangling_pointer.md`
- **TilingData 结构体落地——类内初始化器与注册兼容** · 418 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_dual_struct_nsmdi_registration.md`
- **融合算子 TilingData 内嵌 SoftMax/TopkTiling** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_embed_highlevel_api_struct.md`
- **TilingData 须 host 与 kernel 各定义一份** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_host_kernel.md`
- **TilingData 追加数据超额致 INVALID_TILING** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_invalid_tiling.md`
- **TilingData 改布局须重编 kernel 并清 kernel_meta** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_layout_change_rebuild_kernel_meta.md`
- **TilingFunc 未声明依赖读 axes 致 segfault** · 313 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingfunc_axes_segfault.md`
- **TilingKey 扩展绕不过 aclnn 执行器缓存** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingkey_expansion_cannot_bypass_aclnn_cache.md`
- **vendor 缺 op_host 头致 DYN 重编译失败** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vendor_dyn_recompile_missing_op_host_tiling_data_h.md`
- **tiling魔数au上限与UB方程欠计：均衡au求解** · 213 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/host_tiling_magic_au_cap_ub_equation_rebalance.md`
- **upstream tiling-key variant sprawl is PERF-only, not correctness — for a mathematically-uniform op (pure copy/** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_410_upstream_tiling_key_variant_sprawl_is_perf_only.md`
- **Tiling-data array fields (`TILING_DATA_FIELD_DEF_ARR`) surface kernel-side as a plain C++ array read via `tiling->field[i]` — there is NO generated `get_field()`; `set_field(arr)` copies the FULL decl** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_665_tiling_data_array_fields_tiling_data_field_def_a.md`
- **按行均分 tiling 致 nnz 倾斜时约半数核空闲** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/tiling_nnz.md`
- **aclnn TilingData 缓存致 runtime attr 陈旧** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/aclnn_tilingdata_runtime_attr.md`
- **Host 缺 SetDataSize 致 TilingData 截断** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/host_setdatasize_tilingdata.md`
- **int64+AR 动态 shape aclnn 直调跳过自定义 tiling** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/int64_ar_dynamic_shape_aclnn_skip_tiling.md`
- **非连续输入结果错：tiling 无 stride 须 L2 Contiguous** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/noncontig_tiling_no_stride_l2_contiguous.md`
- **port_a3 V220-pure — emit ALL host-dispatched algorithm variants and replicate the dispatch rule; single-variant ports leak shape-dependent sub-LSB diffs that mimic hw-floor** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_187_port_a3_emit_all_tiling_dispatch_variants.md`
- **TK1 偏移未随 M/K 输出序语义同步致下标互换** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tiling_mk_semantics_swap_offsets.md`
- **tiling 按位置奇偶判 reduce 轴致非交替 axes 误减** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tiling_reduce_axes.md`
- **标量属性入 TilingData 致同 shape 跨调用陈旧** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tilingdata_shape.md`
- **TilingFunc 未写 TilingData 致输出全 0** · 247 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tilingfunc_stub_not_writing_tilingdata_zero_output.md`
- **expect_error 用例被判 FAIL：host 拒收非缺陷** · 224 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ttk_expect_error_negative_case_invalid_tiling_fail.md`
- **507035 超时排障法：强制 tiling 路径隔离 + plog 探针** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/507035_tiling_path_bisection_plog.md`
- **带标量属性 elementwise 算子 tiling 骨架配方** · 545 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/elementwise_scalar_attr_tiling_skeleton.md`
- **空张量分类 H==0 必须先于 B==0 判定** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/empty_tiling_classify_h_zero_before_b_zero_div0.md`
- **TilingFunc 骨架：校验→空→组→基与收尾三件套** · 270 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/multi_branch_tilingfunc_skeleton_validation_order.md`
- **GetCoreNumAiv 实测≈56-57，group 路由分界 28** · 270 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/tiling_route_threshold_runtime_getcorenumaiv_not_ini.md`
- **大C_out用例超时：tiling切分退化仅N核并行** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/tiling_total_tasks_degenerate_n_cores.md`

下一页：page-007.md
