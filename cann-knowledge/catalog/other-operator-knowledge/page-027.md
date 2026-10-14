# other-operator-knowledge: 1821–1890 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **cann-bench op_plugin算子适配配方** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/cann_bench_op_plugin_adaptation_recipe.md`
- **conv3d_weight 不支持非对称 pads，golden 报维度错** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/conv3d_weight_golden_asymmetric_pads_limit.md`
- **CPU golden 对齐 torch：RNE 舍入与 rint** · 203 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/cpu_golden_align_torch_rne_rint.md`
- **自定义 aclnn 算子 PyTorch 接入 ST 测试工程配方** · 723 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/custom_aclnn_pytorch_st_harness_recipe.md`
- **自定义算子 host 侧 UT 工程配方（无 NPU）** · 660 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/custom_operator_host_ut_recipe_no_npu.md`
- **全空输出用例不构成红基线：空张量比对短路恒过** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/empty_output_cases_cannot_form_red_baseline.md`
- **空 tensor 分支 TTK 结构覆盖轴补齐配方** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/empty_tensor_branch_ttk_coverage_axis_supplement.md`
- **bf16比对.numpy()报错，栈被ERR99999掩盖** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/err99999_bf16_numpy_compare_double_bridge.md`
- **L2 异常用例判据：两级 aclnn 报错即 PASS** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/expect_error_dual_stage_aclnn_error_pass.md`
- **expectError但kernel成功：CANN不强制异常约束** · 293 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/expect_error_kernel_succeeded_not_enforced.md`
- **fork 隔离 35 例后驱动耗尽：分批父进程** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/fork_isolation_acl_driver_accumulation.md`
- **check-results PASS 不等于用例通过（防作弊）** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/geir_check_results_weak_pass_semantics.md`
- **GEIR 负向拒绝签名 — RunGraph rc=255 EZ9999** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/geir_negative_reject_rungraph_rc255_ez9999.md`
- **golden bf16 输出用 ml_dtypes 而非 torch** · 148 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/golden_bf16_ml_dtypes_astype_rne_aligns_cast_rint.md`
- **ST golden 须用 dtype 转换后数据** · 226 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/golden_input_dtype_aligns_npu_input.md`
- **审计计算逻辑是否全在 Kernel 而非 Host 侧** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/host_kernel_compute_placement_audit.md`
- **is_available=True 仍 507033：NPU 可用性须实测探测** · 199 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/is_available_true_507033_probe.md`
- **L2 异常用例非法性须物化：Mock 过 Real 返 0** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/l2_exception_invalidity_not_materialized.md`
- **aclnn L2 负向须真实违约并锁错误码** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/l2_negative_true_violation_lock_error_code.md`
- **L2 负向意外 PASS：合法输入误归类，非洗绿** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/l2_unexpected_pass_false_negative_triage.md`
- **registry-invoke（非直调）MC2 自定义算子的 runtime env 契约：ASCEND_CUSTOM_OPP_PATH 绝对路径 + 真实 HCCL 通信组/comm 名** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/mc2_runtime_env_opp_path_comm_group.md`
- **arch22→arch35 移植与反向生成算子的 NPU 验证** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/npu_ut_validation.md`
- **op_host UT 工程：faker+源码 .so+全链旧 ABI** · 792 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/op_host_ut_faker_so_old_abi.md`
- **op_host UT 覆盖率：给 dlopen 的 .so 插桩 gcov** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/op_host_ut_gcov_cov_so_swap.md`
- **UT 报 not found 算子：单目标 make 漏建 .so** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/op_host_ut_single_target_make_missing_so.md`
- **算子交付审计：验证真 AscendC kernel 与 ST 实测覆盖** · 124 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/operator_delivery_audit_csv_st_reconciliation.md`
- **纯搬运算子 ST 字节级 golden 与 bitwise 比对** · 212 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/pure_movement_op_byte_level_golden.md`
- **scatter dim=0 手写期望写反维度：值对调指纹** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/scatter_dim0_expected_value_dim_swap.md`
- **自主 AscendC kernel 静态核查配方** · 223 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/self_ascendc_kernel_static_audit_grep.md`
- **ST axes输入按目标dtype宽度逐元素序列化** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_axes_input_dtype_width_serialization.md`
- **ST 批跑 507014 偶发与状态连带：隔离重跑判别** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_batch_507014_isolated_rerun_triage.md`
- **ST 批量 exit 137：容器 cgroup 限额与 NPU 清理等待** · 386 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_batch_exit_137_cgroup_limit_npu_cleanup_wait.md`
- **ST 二进制 double free 未定位改 torch 路线** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_binary_double_free_unlocated_torch_fallback.md`
- **多分支 ST 共用 case 目录 golden 被覆盖** · 122 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_branch_case_dir_golden_overwrite.md`
- **st-design 生成用例输出维数独立采样需显式约束** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_design_output_dimensions_explicit_constraint.md`
- **用例 shape 全 nan：约束表达式名不匹配被静默跳过** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_design_shape_nan_expression_prefix_mismatch.md`
- **ST 塞 dummy 可选输入切换 kernel 路径，掩盖未测分支** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_dummy_optional_input_switches_kernel_path.md`
- **ST 进程退出码 137：区分 host OOM 与 kernel 未完成** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_exit_137_timeout_kill_not_oom.md`
- **ST golden 多 dtype 扩展：位级转换与 RNE 量化镜像** · 304 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_golden_multi_dtype_bits_rne.md`
- **aclnn ST 大 shape 宿主 oracle 超时，同路径缩 shape** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_host_oracle_timeout_shape_policy.md`
- **MERE 阈值 AND（MARE 阈值 OR 逐元素兜底）混合判定** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_hybrid_judgment_mere_mare_elementwise_fallback.md`
- **无 NPU 的 ST Mock 模式：CPU golden 前置验证** · 609 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_mock_mode_cpu_golden_harness.md`
- **ST 构造真非连续 aclTensor 验证 Contiguous 路径** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_noncontiguous_acltensor_strides_padding.md`
- **ST 非连续 tensor 构造：padded 存储加 strides view** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_noncontiguous_padded_strides_tensor.md`
- **非连续输入 ST 构造：转置 stride 张量+gather golden** · 210 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_noncontiguous_stride_view_construction.md`
- **ST 红基线输出 poison 防 stub 巧合命中** · 312 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_red_baseline_poison_prefill_exact_zero.md`
- **ST 回归归因 worktree 三历史点复跑** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_regression_worktree_three_point_attribution.md`
- **ST报告通过复测失败的四步分诊** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/st_report_pass_retest_fail_triage.md`
- **TilingUT 基线假通过：stub 常量撞上负向期望** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/stub_constant_collides_negative_expectation.md`
- **自定义 aclnn 算子 PyTorch 适配与批量 ST 配方** · 475 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/torch_adapter_aclnn_batch_st_recipe.md`
- **torch_adapter 封装 aclnn 两段式接入配方** · 550 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/torch_adapter_aclnn_two_stage_recipe.md`
- **自定义 aclnn 算子 PyTorch 适配层配方** · 295 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/torch_adapter_privateuse1_op_api_common_base.md`
- **TTK e2e 自定义 aclnn 算子 Torch 路由配方** · 519 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/torch_route_ttk_e2e_dual_mode_recipe.md`
- **aclnn CSV attributes 键用 ABI 名 hasBias** · 298 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_aclnn_csv_keys_follow_deployment_header_abi.md`
- **TTK aclnn golden插件输出位须按C签名序换算** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_aclnn_golden_output_index_c_signature.md`
- **aclnn golden插件bf16需视图转换防崩** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_aclnn_golden_plugin_bf16_view_bridge.md`
- **L2 异常用例 PARAM_COUNT_MISMATCH 为预期失败** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_aclnn_l2_exception_param_count_mismatch.md`
- **TTK 黑盒失败最小复现：子集 CSV+dump** · 148 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_blackbox_failure_min_repro_workflow.md`
- **白盒精度失败为过期产物：TTK compare 钩子生效判别** · 147 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_compare_hook_stale_result_csv_four_checks.md`
- **TTK CSV attributes 需 Python 字面量，JSON 被拒** · 268 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_csv_attributes_python_literal_not_json.md`
- **TTK 白盒 CSV op_name 蛇形与 golden 注册键不一致** · 182 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_csv_op_name_snake_case_golden_registration_key.md`
- **TTK e2e 按 api_name 注册 golden 插件** · 197 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_e2e_custom_api_name_golden_plugin.md`
- **GEIR 图执行 EZ9999/95 MTE DDR 越界与挂起** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_geir_hang_proc_timeout_triage.md`
- **TTK golden插件被模板重实例化覆盖致崩** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_golden_plugin_template_overwrite.md`
- **TTK空attrs用例golden插件须补默认超参** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_kernel_golden_plugin_default_attrs.md`
- **TTK kernel模式torch golden的numpy桥接插件** · 185 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_kernel_golden_plugin_numpy_torch_bridge.md`
- **动态编译产物落 CWD，TTK 查根 kernel_meta** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_kernel_meta_cwd_path_mismatch.md`
- **TTK 采样将 nan→0、inf→±max，CSV 非有限值域不物化** · 202 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_random_data_digitizes_inf_nan_ranges.md`
- **TTK result.csv 多状态列：聚合计数取错列** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_result_csv_multi_status_column.md`
- **TTK TestSpec golden 函数不得带 self 形参** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_testspec_golden_no_self_unbound_call.md`

下一页：page-028.md
