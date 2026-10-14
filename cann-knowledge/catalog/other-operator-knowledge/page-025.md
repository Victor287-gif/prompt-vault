# other-operator-knowledge: 1681–1750 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **507035 — Host 按逻辑行宽预算导致 UB 溢出** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ub_budget_uses_aligned_physical_width_507035.md`
- **ubFactorR 对齐兜底可大于 R** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ubfactorr_floor_align_bottom_out_exceeds_r.md`
- **__ubuf__ 指针传普通形参导致编译失败** · 196 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ubuf_pointer_preserves_address_space.md`
- **UpdateMask 引用参数拒绝 const — 改用可修改左值** · 341 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/updatemask_requires_mutable_uint32_lvalue.md`
- **useRawCopy UB 越界：tiling 对齐 innerSize** · 221 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/use_raw_copy_rawsize_ub_overflow.md`
- **op_api UT 默认 device placement 无上下文误拒** · 181 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ut_aclcreatetensor_default_placement_no_context.md`
- **UT InferShape 空占位聚合初始化编译失败** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ut_infershape_output_empty_brace_init.md`
- **UT 参数结构体 initializer_list 悬垂致维度垃圾** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ut_param_struct_initializer_list_dangling.md`
- **UT 子目录单独 cmake 失败：依赖包级 binary 目标** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ut_standalone_cmake_binary_target_missing.md`
- **InferShape/InferDataType 模板：attr 驱动可选输出** · 286 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/v2_infershape_inferdtype_bool_attr_optional_outputs.md`
- **VAG intrinsic 只能在 VF 循环内使用** · 178 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vag_intrinsic_only_in_vf_loop.md`
- **ValueDepend 虚影 dtype 组合与 Follow 输出冲突** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/valuedepend_dtype_combination_follow_conflict.md`
- **向量Cast不支持float转int64，经int32中转** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vector_cast_float_to_int64_use_int32_pivot.md`
- **VECTOR_REG_WIDTH 只能在 Device 作用域求值** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vector_reg_width_only_in_device_context.md`
- **Vector Sub 源 UB 非 32B 对齐触发错误 340** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vector_sub_ub_32b_340.md`
- **vendor 包 aclnn 装载链路验证探针清单** · 180 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vendor_package_aclnn_loading_four_probes.md`
- **vendors 目录带 _custom 后缀，ST 拼路径须对齐** · 86 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vendors_custom_suffix_path.md`
- **嵌套 VF 循环触发 bisheng stack smashing ICE** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/vf_bisheng_stack_smashing_ice.md`
- **TTK kernel 模式不可用时 aclnn 采集 AiCore 实测耗时** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/aclnn_task_prof_fallback_collection.md`
- **带宽受限 kernel 计算链优化无带宽收益** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/bandwidth_bound_compute_optimization_no_gain.md`
- **带宽受限算子验收画像：对标官方 aclnn 基线** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/bandwidth_bound_elementwise_paired_baseline_acceptance.md`
- **中等 shape 带宽利用率超 100% — L2 命中假象** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/bandwidth_util_over_100pct_l2_working_set.md`
- **逐 block 串行启动退化多核并行** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/block.md`
- **block 数应覆盖可用 Vector 核** · 256 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/block_count_uses_all_available_vector_cores.md`
- **blockLength 对齐 slice 致 numBlocks 塌缩单核** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/blocklength_slice_numblocks.md`
- **无 HBM 峰值时用 copy-variant 对照判带宽达标** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/copy_variant_bandwidth_acceptance_without_peak.md`
- **csr2gebsr 双指针法窄矩阵退化** · 191 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/csr2gebsr.md`
- **cube_l1_reuse_setting 32 非单调反退** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/cube_l1_reuse_setting_32.md`
- **cummin 非末维 batch-inner 跨步扫描 banned** · 124 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/cummin_batch_inner_banned.md`
- **Duplicate 掩码向量化未修复 triu 大 shape 失败** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/duplicate_triu_shape.md`
- **UB 预填充 Duplicate 引入 Vector 开销拖慢纯写带宽** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/duplicate_ub_prefill_hurts_pure_write_bandwidth.md`
- **Gather 非 last-dim 转置+索引换算预处理占比高** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/gather_last_dim.md`
- **950PR HBM 峰值判定：GetCoreMemBw=31 非带宽** · 224 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/getcoremembw_hbm_ini_rate_not_bandwidth.md`
- **GetValue标量直卷大shape超时：最大用例不可行** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/getvalue_direct_conv_feasibility_ceiling.md`
- **L2 读写端口独立并行 HBM 总线串行共享** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/l2_hbm.md`
- **launch floor触底目标不可达：pipe body标定偏低** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/launch_floor_bottomed_target_unreachable_on_device.md`
- **MIX_AIC_1_1 仅启用 28/56 AIV 带宽不饱和** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/mix_aic_1_1_underutilizes_aiv_on_1_to_2_topology.md`
- **Mix合图包入 V2 致任务数减半 E2E 劣化** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/mix_v2_e2e.md`
- **MLA Prolog decode 950 tile 调优配方** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/mla_prolog_decode_950_tile_tuning_recipe.md`
- **MulPreferContiguous 稀疏广播拦截失败且慢于 304** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/mulprefercontiguous_304.md`
- **Per-process tagging for shared-temp resources under parallel-batch workloads** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_123_per_process_tagging_for_shared_temp_resources_unde.md`
- **Shared NPU — performance data affected by concurrent users** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_15_shared_npu_performance_data_affected_by_concurrent.md`
- **Cross-arch perf timing: device-event symmetric first, wrapper-inclusive fallback, N/A last resort** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_163_cross_arch_perf_timing_methodology.md`
- **int64 fixes cause ~6% performance regression on AscendC SIMT** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_16_int64_fixes_cause_6_performance_regression_on_asce.md`
- **A pybind-wrapped custom-op's e2e wall is not apples-to-apples vs a native vendor op — compare kernel-device time and decompose the wall before blaming the kernel** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_201_pybind_wall_not_apples_to_apples_vs_native.md`
- **Never permanently label a platform feature as "broken" without re-verification** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_23_never_permanently_label_a_platform_feature_as_brok.md`
- **Layout-transformation overhead — permute+contiguous for NCDHW↔NDHWC can eat >60% of wall-clock; expose a raw API** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_248_layout_transformation_overhead_raw_api.md`
- **After optimizing one kernel, must audit all similar kernels** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_24_after_optimizing_one_kernel_must_audit_all_similar.md`
- **da Vinci VEC throughput tiers — Muls (Tier 0) is 8-16× faster than Div (Tier 4)** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_257_davinci_vec_throughput_tiers.md`
- **Performance claims must be based on same-condition A/B data — never on speculation** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_27_performance_claims_must_be_based_on_same_condition.md`
- **before ANY rig-based A/B, align the rig's BUILT kernel to the current production kernel (md5-verify) — a stage rig's built kernel can lag production by generations, making the A/B baseline wrong / the** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_283_before_any_rig_based_a_b_align_the_rig_s_built_ker.md`
- **Performance benchmarking must use the target framework's standard tools** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_31_performance_benchmarking_must_use_the_target_frame.md`
- **View-returning ops have ~0 reference cost — benchmark vs materialized copy** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_39_view_returning_ops_have_0_reference_cost_benchmark.md`
- **on a 220x→dav-m200 port, the upstream #if __CCE_AICORE__ < 220 branches ARE the vendor's own 200x design — rea** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_405_on_a_220x_dav_m200_port_the_upstream_if_cce_aico.md`
- **index/metadata stable at kernel launch (host-written or prior stream-ordered op) needs NO in-kernel MTE2_S syn** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_409_index_metadata_stable_at_kernel_launch_host_writ.md`
- **a kernel whose output is uint64 (packed quant params / hash / packed-index) is bridged through the pybind veri** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_413_a_kernel_whose_output_is_uint64_packed_quant_par.md`
- **CANN perf gap is a knowledge gap, not a hardware limit** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_42_cann_perf_gap_is_a_knowledge_gap_not_a_hardware_li.md`
- **nblk=1 test is the ultimate diagnostic tool for multi-core contention** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_43_nblk_1_test_is_the_ultimate_diagnostic_tool_for_mu.md`
- **512B-aligned GM address yields 30% higher transfer bandwidth** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_47_512b_aligned_gm_address_yields_30_higher_transfer.md`
- **Multi-pass kernels — HBM bandwidth ceiling vs fused CANN** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_53_multi_pass_kernels_hbm_bandwidth_ceiling_vs_fused.md`
- **Reg-based SIMD — register-level vector compute API (officially confirmed)** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_54_reg_based_simd_register_level_vector_compute_api_o.md`
- **hardware gather (Gather/vgather) is the right primitive for ARBITRARY-index gather on SIMD-only arches (dav-m2** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_552_hardware_gather_gather_vgather_is_the_right_prim.md`
- **An AICPU-only upstream source (no op_kernel/, no arch35/) is a VALID port_a3 input** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_555_an_aicpu_only_upstream_source_no_op_kernel_no_ar.md`
- **算子开发三条铁律——不接受单核 / 不接受标量搬运与标量计算 / CV 融合算子不接受纯 AIV（所有 op 类、所有 target，最高优先级）** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_626_cv_aiv_op_target.md`
- **dav-m200 unaligned-dst VEC ops are LEGAL (arbitrary off + non-AE count)** · 19 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_669_dav_m200_unaligned_dst_vec_ops_are_legal_arbitra.md`
- **dav-m200 per-VEC-API-call fixed cost ≈ 450-500 cycles, length-independent** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_672_dav_m200_per_vec_api_call_fixed_cost_450_500_cyc.md`
- **dav-m200 (Ascend310P3) honors 8 vector cores; nblk=8 scales monotonically over 6/7** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_673_dav_m200_ascend310p3_honors_8_vector_cores_nblk.md`
- **every cross-dtype torch .to() costs ~145-167us wall on this torch_npu build, even for 1K elements** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_677_every_cross_dtype_torch_to_costs_145_167us_wall.md`
- **310P data-movement kernels MUST ship MULTI-CORE (DataCopy-only + K-aligned row groups) — single-core is NOT an acceptable fallback; the multi-core scalar-GM-store drop (PB-78) is a reason to eliminate** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_683_310p_data_movement_kernels_must_ship_multi_core.md`
- **Vector-op 性能为准 routing standard — default vectorized + multi-core; scalar/单核只在"实测更快"才允许 (VECTOR_REGIME gate)** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_690_vector_op_routing_standard_default_vectorized_mu.md`

下一页：page-026.md
