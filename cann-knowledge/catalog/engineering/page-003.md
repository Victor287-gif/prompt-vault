# engineering: 141–210 / 823

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Blaze-SHMEM 对称地址、源目标偏移与容量** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_sample_symmetric_offsets_and_capacity.md`
- **Blaze 本地 SWAT 调度坐标与尾块覆盖** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_swat_scheduler_coordinates_and_tail_coverage.md`
- **Blaze SWAT Tiling 搜索与资源边界** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/shmem/blaze_swat_tiling_search_and_resource_bounds.md`
- **Blaze 局部槽位与跨核通道的分别闭合** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/blaze_local_slots_and_cross_core_channels.md`
- **Blaze 两阶段初始化与空闲路径** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/synchronization/blaze_two_phase_initialization_and_idle_paths.md`
- **TTK Kernel CSV 的 TensorList 编码** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/testing/whitebox_ttk_csv_tensorlist_encoding.md`
- **卷积 fallback HW字段与乒乓标志编码** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/tiling/convolution_fallback_hw_fields_and_pingpong_encoding.md`
- **PyTorch 自定义 ACLNN 算子的四层接口契约** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/aclnn_schema_and_graph_contracts.md`
- **ACLNN 扩展辅助层的符号解析与异步资源生命周期** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/aclnn_symbol_resolution_and_lifetime.md`
- **PyTorch Ascend C 扩展的编译单元与启动 ABI** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/compilation_and_launch_boundary.md`
- **直调 Add 的 Tensor 合同与 Tiling 生命周期** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/direct_add_tensor_and_tiling_lifetime.md`
- **直调对象库构建与 PyTorch 分发合同** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/direct_object_build_and_dispatch_contracts.md`
- **MX 扩展的原始存储与分发验证** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/mx_raw_storage_and_dispatch_validation.md`
- **算子 Python 包装的形状、路径与进程参数** · 26 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/python_shape_paths_and_process_arguments.md`
- **注册 PyTorch 适配器的 Tensor 身份与异步所有权** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/registry_adapter_tensor_identity_and_async_ownership.md`
- **PyTorch Ascend C 扩展的任务队列与 Stream 顺序** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/torch_extensions/task_queue_and_stream_order.md`
- **target/ascendc/build_system — Build-System Pattern (BSP) KB subdir** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/build_system_pattern_bsp_subdir.md`
- **TILING_KEY bit-encoding convention: pack orthogonal flags into one uint64, extend never bit-steal** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_136_tiling_key_bit_encoding_convention.md`
- **Membase vs Regbase selection + Reg vector __simd_vf__ programming entry (CANN 9.0.beta1 doc-distill)** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_196_membase_vs_regbase_simd_vf_selection.md`
- **Restricted launch-lib link line makes a matmul-library cube kernel structurally unbuildable — this is the harness-linkage reason cube-class ports default to manual Mmad** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_235_cube_tcubetiling_unbuildable_restricted_link.md`
- **On A5/arch35, SIMD compute CHAINS default to REGBASE, not Membase/LocalTensor — but regbase wins require large per-VF-call amortization (measure, don't blind-rewrite)** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_245_simd_chains_regbase_vs_membase.md`
- **Vendor adv_api regbase primitive substitution requires per-call batch axis A>1 to amortize internal scalar-broadcast cost** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p87_vendor_adv_api_regbase_primitive_substitution_re.md`
- **ACLNN 561102 单 dtype 原型注册与 opbuild 字节约束** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnn_561102_dtype_opbuild.md`
- **arch35 算子扩展 fp16/bf16 支持全链路配方** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_fp16_bf16_dtype_extension_recipe.md`
- **arch35 RegBase VF 逐元素算子完整实现模式** · 621 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_regbase_vf_elementwise_pattern.md`
- **ASCENDC_TPL 模板路由与 TilingKey 实测编码规则** · 216 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ascendc_tpl_routing_tilingkey_encoding.md`
- **bisheng 不支持 bf16 标量强转时用位布局精确提升到 fp32** · 823 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/bf16_scalar_to_fp32_bit_extension.md`
- **Package not found误报：PKG_PATH检查路径错** · 243 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/build_sh_package_check_path_misreport.md`
- **部分清理重建后 CPack 打包失败须全清重建** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/cpack_partial_clean_rebuild_package_failure.md`
- **直调不可用 GET_TILING_DATA_WITH_STRUCT 宏** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/direct_invoke_unavailable_get_tiling_data_with_struct.md`
- **the raw-launch build chain autogenerates per-entry wrappers at configure time with no prune pass, so REMOVING/** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_204_the_raw_launch_build_chain_autogenerates_per_ent.md`
- **on the CANN 9.1.0 tikcfw (dav-m200 builds) FetchEventID returns TEventID, not event_t** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_213_on_the_cann_9_1_0_tikcfw_dav_m200_builds_fetchev.md`
- **build_ascendc.py default SOC_VERSION causes 507035 on Ascend950PR** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_27_build_ascendc_py_default_soc_version_causes_507035.md`
- **standalone-cmake pybind builds (no torch cpp_extension env) leave `TORCH_EXTENSION_NAME` undefined — `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` registers a literal module name the wheel cannot import** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_293_standalone_cmake_pybind_builds_no_torch_cpp_exte.md`
- **A `__VEC_SCOPE__` regbase VF that hardcodes its element/tile count to a fixed inner-dim constant produces garbage above that constant — derive the count from the RUNTIME inner dim** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_76_a_vec_scope_regbase_vf_that_hardcodes_its_element.md`
- **GE图prebuildPattern=Opaque跳过预编译匹配** · 178 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ge_prebuild_pattern_opaque_skips_binary_matching.md`
- **kernel头改动后增量编译沿用旧二进制** · 346 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/incremental_build_stale_kernel_binary.md`
- **kernel binary .o 定位：build/packages 不存在** · 206 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_o_location_not_build_packages.md`
- **单TU编译NdDma报target feature需split构建** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/nd_dma_dci_single_tu_target_feature_split_build.md`
- **arch22->arch35 Phase C: build+register activation criteria and build.sh exit-code interpretation** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_158_phase_c_build_register_exit_code.md`
- **Empirically attempt the build before declaring an L4 "structural ceiling" [V351/Ascend950PR, port_a3_to_a5, L0-agent-discipline + classifier-override]** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_176_empirically_attempt_the_build_before_declaring_an.md`
- **Build recipe for a cube+vec MIX kernel via ascendc_library — non-empty CMAKE_BUILD_TYPE, post-ascendc_library include scoping, and where MultiCoreMatmulTiling lives** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_220_mix_ascendc_library_build_recipe.md`
- **A build-only A5/V351 host (toolkit-only host CANN) needs a clean own-container device-runtime — ABI-matched torch_npu + a complete CANN, not the host's toolkit CANN** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_234_build_only_a5_host_needs_own_device_runtime.md`
- **build_ascendc.py takes SOC from the --soc-version arg, not an env var — manual builds silently mis-target the arch** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_240_build_ascendc_soc_version_arg_not_env.md`
- **A deployed .so failing 'RuntimeError: Unsupported soc version: Ascend950PR 957b' while npu-smi shows Health=OK is a torch_npu / cpython BUILD mismatch — not a host/CANN/device fault** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_244_unsupported_soc_version_build_mismatch.md`
- **matmul::Matmul<> DOES build and run inside the build_ascendc.py pybind harness when the CANN host tiling libs** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_508_matmul_matmul_does_build_and_run_inside_the_buil.md`
- **opbuild dtype 列表不等长报 dtype size 0** · 360 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opbuild_dtype_lists_must_be_equal_length.md`
- **opbuild 编译 op_host 不注入 op_kernel 头路径** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opbuild_no_op_kernel_include_use_relative_path.md`
- **opbuild 静默失败：infershape 引非内联库符号** · 206 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opbuild_silent_error_opbase_noninline_symbol.md`
- **ascendc 2-stage build destabilizes when a THIRD kernel entry is appended to a `*_kernels.cpp`** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_181_ascendc_2_stage_build_destabilizes_when_a_third.md`
- **Missing `config/ascend950/<op>_binary.json` → 950 build SILENTLY SKIPPED [V351, build-system]** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_31_missing_config_ascend950_op_binary_json_950_build.md`
- **CANN merge_mix_obj.sh Crash with Empty --build-type** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_7_cann_merge_mix_obj_sh_crash_with_empty_build_type.md`
- **重编未重装 vendors 旧 tiling 致 561103** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/rebuild_without_reinstall_561103.md`
- **RegBase VF __simd_vf__ 路线致 bisheng 编译崩溃** · 200 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/regbase_vf_simd_vf_bisheng_exit139_banned.md`
- **tilingKey 编码与 TPL 位打包不一致致路由错误** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tiling_key_tpl_bit_encoding_mismatch.md`
- **TilingData 编入构建四探针：opParaSize=8+sizeof** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_compiled_into_build_four_probes.md`
- **TilingData 改布局须重编 kernel 并清 kernel_meta** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tilingdata_layout_change_rebuild_kernel_meta.md`
- **UT 编译缺 context_builder include 路径** · 129 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ut_context_builder_include_path.md`
- **Establish bug-reproducibility BEFORE building a proxy-investigation — a non-reproducible defect's proxy-refutations are not bug-relevant** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_207_establish_bug_reproducibility_before_building_a_pr.md`
- **a reference model hardcoding fp64 compute produces an fp64-disguised native_capture.pt (bit-identical to cpu_t** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_418_a_reference_model_hardcoding_fp64_compute_produc.md`
- **arch35 RegBase VF：TQue 经 GetPhyAddr 桥接** · 381 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/regbase_vf_tque_getphyaddr_bridge.md`
- **softmax regbase VF计算链参考结构** · 308 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/softmax_regbase_vf_compute_chain_reference.md`
- **host tiling改动先Python仿真对账bd/tk** · 222 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/host_tiling_python_sim_crosscheck_before_build.md`
- **TilingContext 独立探针配方（builder 双坑）** · 253 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/optilingcontextbuilder_probe_deleted_copy_compileinfo.md`
- **gtest TilingUT 编入 build.sh 的 binary 目标** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/tilingut_wire_into_buildsh_binary_target.md`
- **TritonGPU 布局编码详解** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/tritongpu_encodings.md`
- **Attention 算子优化** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_attention.md`
- **Triton Ascend Layout-transform / Permutation 代码生成要点** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_layout_transform.md`
- **Reduce 算子优化** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_reduce.md`
- **Sort/Select 算子优化** · 282 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_sort_select.md`

下一页：page-004.md
