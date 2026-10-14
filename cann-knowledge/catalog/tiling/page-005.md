# tiling: 281–350 / 686

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **对角线网格置换与边缘窗口** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/diagonal_grid_permutation_and_edge_windows.md`
- **直调 Add 的分核覆盖与 tile 地址** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/direct_add_tile_offsets_and_coverage.md`
- **直调单目模板的分区、UB 与类型合同** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/direct_unary_template_partition_and_buffer_contracts.md`
- **DSL 转换的分核、尾部与阶段 ABI** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/dsl_partition_and_stage_abi.md`
- **动态 UB Broadcast 的重载与存储约束** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/dynamic_broadcast_overloads_and_storage_contracts.md`
- **Elementwise fallback 字节、核数与精度字段** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/elementwise_fallback_bytes_cores_and_precision_fields.md`
- **GMM 不规则分组任务与资源单位** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/grouped_matmul_ragged_tasks_and_resource_units.md`
- **分层分块、参数复用与 Foreach 余数** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/hierarchical_tiles_parameter_reuse_and_foreach_remainders.md`
- **Tiling 整数范围证明与守卫语义** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/integer_range_proofs_and_guard_semantics.md`
- **线性切块的单位、核数与尾部范围** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/linear_partition_units_and_tails.md`
- **Matmul 分层 Tiling、Batch 广播与 Split-K 边界** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/matmul_hierarchy_batch_and_split_k_corrections.md`
- **MC2 fallback 尾块大小与候选搜索边界** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/mc2_fallback_tail_size_and_candidate_search_limits.md`
- **多轴尾块、池化索引与阶段归属** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/multiaxis_tails_pool_indices_and_phase_ownership.md`
- **多轴任务组、Stream-K 与尾块再分裂** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/multiaxis_work_groups_streamk_and_tail_split_limits.md`
- **MX Tiling Stub 的字段关系与整数边界** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/mx_stub_fields_capacity_and_integer_bounds.md`
- **一维切分的完整覆盖与尾部搬运** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/one_dimensional_coverage_and_tail_transfers.md`
- **OneDim 广播tile数与标量驻留** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/onedim_broadcast_tile_counts_and_scalar_residency.md`
- **注册 Add 的分区、空输入与队列缓冲语义** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/registry_add_partition_empty_and_queue_buffer_semantics.md`
- **形状特化的分发与共享资源边界** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/shape_specialization_dispatch_and_shared_resources.md`
- **Softmax Host容量公式与空维失效条件** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/softmax_host_capacity_formulas_and_empty_shape_failures.md`
- **Tiling 搜索参数角色与语义约束** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/tiling_search_parameter_roles_and_semantic_constraints.md`
- **直调 Add 的 Tensor 合同与 Tiling 生命周期** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/direct_add_tensor_and_tiling_lifetime.md`
- **Matmul 静态 Tiling、部分常量化与运行期 shape** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/matmul/static_tiling_and_runtime_shapes.md`
- **TILING_KEY bit-encoding convention: pack orthogonal flags into one uint64, extend never bit-steal** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_136_tiling_key_bit_encoding_convention.md`
- **Exhaust host-contract reconstruction before concluding a deeper device root** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_204_faithful_copy_host_tiling_before_disasm.md`
- **Restricted launch-lib link line makes a matmul-library cube kernel structurally unbuildable — this is the harness-linkage reason cube-class ports default to manual Mmad** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_235_cube_tcubetiling_unbuildable_restricted_link.md`
- **Tiling-tensor cache + C++ convenience wrapper — cut host-side overhead in pybind11 kernels** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_250_tiling_tensor_cache_cpp_convenience_wrapper.md`
- **2D Tiling for Large Embedding Scatter** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p41_2d_tiling_for_large_embedding_scatter.md`
- **Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p68_single_aic_gemm_with_constexpr_static_tiling_on.md`
- **多 vendor 同名 TilingFunc 符号冲突 561002** · 178 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561002_multi_vendor_tilingfunc_symbol_collision.md`
- **561002：TilingParse 未生效，CompileInfo 为空** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561002_tilingparse_compileinfo.md`
- **561002：torch_npu GE 初始化干扰 aclnn tiling** · 369 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561002_torch_npu_ge_init_tiling_interference.md`
- **561103: isGroup=-1 哨兵致 tilingKey 路由失配** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/561103_isgroup_1_tilingkey.md`
- **361001 非零 tilingKey 须逐 key 注册** · 163 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_361001_register_tiling_for_tilingkey_per_key.md`
- **ACLNN 361001 — tilingKey 与 funcEntry 映射失** · 210 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_361001_tilingkey_funcentry_sel_position.md`
- **ACLNN 507015 — tiling UB 漏算 kIdxBuf 下溢** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_507015_tiling_ub_kidxbuf.md`
- **ACLNN 507035 — stub kernel tiling 数据未初始化** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_507035_stub_tilingfunc_must_init_tiling_buffer.md`
- **aclnn L2不传导tiling workspace致507035** · 152 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_l2_tiling_workspace_gap_507035.md`
- **纯向量算子tiling误用GetCoreNumAic** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aiv_only_tiling_getcorenumaic_mismatch.md`
- **单 kernel 多 TilingKey：dtype×策略模板整合配方** · 454 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_dtype_strategy_tilingkey_recipe.md`
- **ASCENDC_TPL 多参数 TilingKey 声明与全量实例化** · 333 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_multi_param_tilingkey.md`
- **ASCENDC_TPL 模板路由与 TilingKey 实测编码规则** · 216 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_routing_tilingkey_encoding.md`
- **TilingData 注册：BEGIN_TILING_DATA_DEF 已废弃** · 253 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/begin_tiling_data_def_deprecated.md`
- **模板 kernel + tilingKey 派发：分支启用只改 apt 接线** · 186 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/class_template_tilingkey_dispatch_wiring_only.md`
- **直调不可用 GET_TILING_DATA_WITH_STRUCT 宏** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/direct_invoke_unavailable_get_tiling_data_with_struct.md`
- **按 dtype 分配 UB 必须匹配 tiling 预算** · 254 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/dtype_conditional_ub_allocation_matches_tiling_budget.md`
- **PadTiling name conflict with CANN built-in** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_19_padtiling_name_conflict_with_cann_built_in.md`
- **unknown type name '<Op>TilingData' flagged on an upstream op_kernel/.h line means the tiling struct is defined** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_200_unknown_type_name_op_tilingdata_flagged_on_an_up.md`
- **memset(0) on an optiling tiling struct (TCubeTiling/SoftMaxTiling, or any host struct embedding them) zeroes t** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_202_memset_0_on_an_optiling_tiling_struct_tcubetilin.md`
- **Tiling CPU→NPU copy must happen AFTER all fields finalized** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_20_tiling_cpu_npu_copy_must_happen_after_all_fields_f.md`
- **helpers in a shared tiling header called from aicore code must be `__aicore__ inline` — a host-flavored `inline` helper fails the device compile** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_298_helpers_in_a_shared_tiling_header_called_from_ai.md`
- **Cube tiling host POD size mismatch — `TCubeTiling` is 50 int32 (200 B), not 51** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_40_cube_tiling_host_pod_size_mismatch_tcubetiling_is.md`
- **A5 port `<op>_apt.cpp` `fatal error: '<op>_tiling.h' file not found` — must not `#include` the host-side tiling header** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_52_a5_port_op_apt_cpp_fatal_error_op_tiling_h_file_no.md`
- **`GET_TILING_DATA` macro unresolved (`use of undeclared identifier 'tilingData'`) in a GENERATED arch35 kernel compiled by the workspace verify-build → define a POD TilingData struct before the algorit** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_71_get_tiling_data_macro_unresolved_use_of_undeclared.md`
- **elementwise host tiling 切分配方** · 254 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/elementwise_host_tiling_ub_budget_recipe.md`
- **空tensor tiling：EMPTY_A清零+EMPTY_R重映射** · 376 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/empty_tiling_returns_empty_field_remapping.md`
- **TilingKey 缓存 correction 致 tilingdata 复用** · 153 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_tilingkey_cache_reuse_per_call_attr_pollution.md`
- **GEIR 图路径不消费 TilingParse/GenSimplifiedKey** · 265 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/geir_tilingparse_gensimplifiedkey_not_consumed.md`
- **GET_TILING_DATA 首参传类型名致变量遮蔽类型** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/get_tiling_data.md`
- **GET_TILING_DATA_WITH_STRUCT 宏不接受模板类型** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/get_tiling_data_with_struct.md`
- **GET_TILING_DATA_WITH_STRUCT 限入口函数** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/get_tiling_data_with_struct_kernel_entry_only.md`
- **GM_ADDR 标量不可解引用为 tiling 结构体指针** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/gm_addr_tiling.md`
- **高阶 API tiling 头传递依赖须补 include** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/highlevel_api_tiling_header_adv_api_include.md`
- **host 编译找不到 tiling_struct.h：补 include 路径** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/host_compile_tiling_struct_include_path.md`
- **Host Tiling UT 基础设施故障诊断** · 320 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/host_tiling_ut_infrastructure_failures.md`
- **工程化 op-build 路径 host tiling 的 ws[0] = usrWorkspace + sysWorkspace（求和），绝不能取 max——否则 MTE DDR 越界** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/host_tiling_workspace_sum_not_max.md`
- **kernel .o 文件名 hash 不随源码变化，以 MD5 判重编** · 158 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_o_filename_hash_tilingkey_not_source.md`
- **kernel printf 引用 tiling 形参编译报错** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_printf_tiling_param_namespace_conflict.md`
- **Conv2d list 属性经 OP_ATTR→TilingContext 传递** · 242 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/list_attr_op_attr_tiling_passing.md`
- **混合精度 scales 用 TilingData 标志双绑定** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mixed_dtype_scales_tilingdata_flag_dual_tensor.md`

下一页：page-006.md
