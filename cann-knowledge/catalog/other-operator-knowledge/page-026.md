# other-operator-knowledge: 1751–1820 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Recurrent ops are super-speedup targets for AscendC** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_76_recurrent_ops_are_super_speedup_targets_for_ascend.md`
- **Always verify expert claims with hardware test** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_7_always_verify_expert_claims_with_hardware_test.md`
- **Oversubscription helps unsorted bwd (-28%) but hurts sorted bwd (+17%)** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_8_oversubscription_helps_unsorted_bwd_28_but_hurts_s.md`
- **host-path-first perf on slow-host raw-launch kernels** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p202_host_path_first_perf_on_slow_host_raw_launch_ker.md`
- **predicate/mask cast chain on dav-m200** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p203_predicate_mask_cast_chain_on_dav_m200.md`
- **regime split for duplicate-writer scatter by output span M vs UB×cores** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p204_regime_split_for_duplicate_writer_scatter_by_out.md`
- **When the cancellation-free identity has NO library primitive, SYNTHESIZE it by range reduction — the reduced form is branch-free precisely because `k=0` covers the dangerous band** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p272_when_the_cancellation_free_identity_has_no_libra.md`
- **AscendC data-rearrangement vectorization — DataCopy(VECIN) load + byte-offset Gather reorder + K-group aligned store** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p276_ascendc_data_rearrangement_vectorization_datacop.md`
- **Multi-core data-movement partitioning — K-aligned row groups + DataCopy-only stores (NO scalar GM store)** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p277_multi_core_data_movement_partitioning_k_aligned.md`
- **Slab-window k-partition for dn>=1 scatter/gather-class ops** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p279_slab_window_k_partition_for_dn_1_scatter_gather.md`
- **Host-precompute dim-only fp64 tables and carry them via a tiling data array — never compute dim-only pow/exp tables on-device in fp32 when a Ratio-vs-fp32-competitor gate applies** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p280_host_precompute_dim_only_fp64_tables_and_carry_t.md`
- **随机 Gather 每元素 DMA 过慢—改 SIMT 直访 GM** · 268 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/random_gather_per_element_dma_should_switch_to_simt_direct_gm.md`
- **scatter 亿级元素 RMW 灾难性慢（非挂死）** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/scatter_hundred_million_rmw_slow_not_hung.md`
- **稀疏不规则输出算子 SIMD 标量退化诊断** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/simd.md`
- **上板首跑warmup偏慢：TTK取后N-1次中位数** · 165 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/task_prof_warmup_first_run_median_of_rest.md`
- **通算流水 tileCnt>1 性能反降 banned** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/tilecnt_1_banned.md`
- **VF收拢与lane累加消除scalar共bound** · 435 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/vf_lane_accumulate_row_loop_consolidation.md`
- **507035崩溃：负模运算产生UB负偏移** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/507035_negative_modulo_ring_index_ub_offset.md`
- **507035 向量核异常的 plog errcode 定位法** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/507035_plog_errcode_triage.md`
- **507035 崩溃：多核分发未拦截多余核越界** · 285 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/507035_unguarded_extra_cores_oob_dispatch.md`
- **aclDestroyTensorList 拥有成员勿重复销毁** · 180 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/acldestroytensorlist_owns_member_tensors.md`
- **Separate-mode kernel 的 AIC/AIV 角色必须在编译期用 ASCEND_IS_AIC/AIV 宏隔离——禁止运行期动态判别** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/aic_aiv_role_compile_time_isolation.md`
- **AscendC PRINTF 路由：TTK 查 plog 且防流水线扰动** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ascendc_printf_plog_routing.md`
- **多 vendor 同 hash kernel 冲突的检测与排除** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/dual_vendor_same_hash_kernel_conflict.md`
- **foreach TensorList 动态输入算子全栈实现模板** · 807 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/dynamic_input_tensorlist_full_stack_template.md`
- **空batch归约kernel：bias零填充+全核早退配方** · 389 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/empty_branch_kernel_empty_a_early_exit_empty_r_zero_fill.md`
- **group 2D 分核：对齐 aLoop 与 SetScheduleMode** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/group_2d_core_split_numblocks_align_schedule_mode.md`
- **GroupTailA 分核：合并分发+tail-A 助手复用** · 290 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/group_tail_a_merged_dispatch_helper_reuse.md`
- **GRU workspace 五平面布局：平面 4 为 hn+隐藏偏置** · 271 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/gru_workspace_five_plane_layout.md`
- **Host 预写 workspace 不可行：CreateVecIndex** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/host_no_workspace_write_api_createvecindex.md`
- **kernel 垃圾值分层探针：常量写入→回读→向量拷贝→参数回显** · 200 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/kernel_output_garbage_layered_probe_chain.md`
- **kernel 侧打印定位崩溃：AscendC::PRINTF 宏** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/kernel_printf_ascendc_macro_ascendc_dump_bisect.md`
- **内核运行期故障初判：先读 plog 提取 errorStr/errorcode 再二分 kernel** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/kernel_runtime_fault_plog_first_triage.md`
- **kernel直绑aclnn workspace入参即507035** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/kernel_workspace_param_direct_bind_507035.md`
- **Mish 数值稳定链：Log/Tanh 禁重叠与 tmpBuf 轮转** · 331 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/mish_softplus_stable_chain_log_tanh_overlap.md`
- **MIX_AIC_1_2：SetBlockDim 传 AIC 数（框架按 1:2 扩展发射），printf 缺失不等于核没跑** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/mix_set_block_dim_aic_count_printf_flush.md`
- **多 vendor 共存 LD 路径选错致符号缺失** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/multi_vendor_ld_path_wrong_lib.md`
- **507035排障：plog定性+跳层二分+对账** · 223 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/plog_aivec_exception_diagnosis_507035.md`
- **PRINTF 扰动时序：去 PRINTF 复验再定论** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/printf_perturbs_timing_verify_without_printf.md`
- **纯 AIV 写 workspace 507035 errcode 95** · 196 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/pure_aiv_workspace_write_507035.md`
- **rowIdx 直索引 GM 未校验，按 drop 模式补行数上界** · 172 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/rowidx_gm_index_range_guard_drop_mode.md`
- **tail-R 切分 rUbFactor 实切最内 R 才 FloorAlign** · 229 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/tail_r_rubfactor_flooralign_guard.md`
- **GetBlockNum 对齐后 TBE 多核执行仍 EZ9999** · 153 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/tbe_baked_blockdim_getblocknum_align_insufficient.md`
- **tile-loop 统一 kernel：full-load 退化为单 tile** · 453 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/tile_loop_unified_kernel_full_load_degenerate.md`
- **TopK非末维多路线修复失败回退** · 277 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/topk_non_last_dim_multi_route_failure.md`
- **torch_npu AdamW.step 触发 ERR99999** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/torch_npu_optimizer_step_err99999.md`
- **ttk aclnn warmup 依赖 tbe→scipy 双路径** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ttk_aclnn_warmup_tbe_scipy_pythonpath.md`
- **TTK kernel 模式 VEC_ERROR 而 aclnn 正常的判别** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ttk_kernel_vec_error_aclnn_cross_check.md`
- **TTK Opc init failed：PYTHONPATH 被覆盖** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ttk_opc_init_pythonpath_overwrite.md`
- **UB 预算反解切分：R 因子反解、R 驻留扩 A、钳制环** · 201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ub_budget_backsolve_dual_axis_split_clamp_ring.md`
- **tiling UB 预算不留 headroom 输出全 0** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ub_budget_backsolve_headroom_silent_zero.md`
- **561103 非连续输入：storageDims 须传物理 shape** · 222 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/561103_storagedims_physical_not_view_shape.md`
- **ACLNN 561002：占位 1D 张量过不了 rank 校验** · 188 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_561002_rank1_placeholder_tensor_rank_check.md`
- **ACLNN 561002：scatter 复现用例形状撞校验** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_561002_scatter_non_dim_shape_violation.md`
- **aclnn 条件输出用 (0,) 空张量非零填充** · 130 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_conditional_output_zero_shape_entity.md`
- **aclnn 交付前 3 次 bit-wise 确定性回归** · 328 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_determinism_3run_bitwise.md`
- **aclnn生成头Optional后缀只加指针型参数** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_header_optional_suffix_pointer_params_only.md`
- **TTK无负向支持：aclnn两段式负向验证** · 219 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_negative_two_stage_verify_padded_out.md`
- **aclnn 算子 PyTorch ST 接入完整配方** · 354 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_operator_pytorch_st_integration_recipe.md`
- **aclnn 算子 PyTorch ST 接入工程五件套配方** · 1201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_pytorch_st_torch_adapter_project_recipe.md`
- **aclnn ST 自研 TTK stat_rel_err 比较器** · 229 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_st_cxx_stat_rel_err_comparator_port.md`
- **aclnn ST 空张量用例接入：0 字节分配与拷贝守卫** · 779 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_st_empty_tensor_case_zero_byte_guards.md`
- **C++ ST fp16/bf16 全 dtype 数据通路适配配方** · 311 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_st_full_dtype_chain_adaptation.md`
- **aclnn ST 独立 oracle 分模式比对与 stub 退出码** · 230 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_st_independent_oracle_stub_exit_code.md`
- **manifest 驱动 aclnn ST 框架（独立 oracle）** · 586 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_st_manifest_oracle_harness_recipe.md`
- **aclnn ST 掩盖 execute 错误码：直连探针** · 209 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_st_masks_execute_error_code_probe.md`
- **改kernel后bench复评须重跑适配并重生成op_plugin** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/bench_reeval_requires_readapt_plugin_rewrite.md`
- **黑盒全量验收：manifest过滤+3×bit-wise确定性** · 225 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/blackbox_acceptance_manifest_filter_determinism.md`
- **黑盒CSV用例数据驱动aclnn ST框架** · 540 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/blackbox_csv_data_driven_st_framework.md`
- **大shape黑盒用例分块流式生成golden防OOM** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/blackbox_large_shape_chunked_golden_worker.md`

下一页：page-027.md
