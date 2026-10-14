# other-operator-knowledge: 1541–1610 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **DEF dtype 集须被 ASCENDC_TPL_SEL 全覆盖** · 303 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opdef_dtype_tpl_sel_coverage.md`
- **OpDef FORMAT_NCHW 致 5D 报 161002** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opdef_format_nchw_5d_161002.md`
- **OpDef 顶层无 ExtendCfgInfo—kernel 文件按命名约定映射** · 165 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opdef_no_extendcfginfo_member.md`
- **TensorList OpDef 须 ParamType(DYNAMIC)** · 228 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opdef_paramtype_dynamic_for_tensorlist.md`
- **ValueDepend(REQUIRED) 让 axes 走 host 端** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opdef_value_depend_required_for_axes_host_read.md`
- **算子名数字段致 aclnn 安装头文件名去下划线** · 194 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/opname_digit_snake_case_install_header_mismatch.md`
- **optional input 未连接触发 InferShape 失败** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/optional_input_infershape.md`
- **可选输入空指针合法，必选输出不可漏建 161001** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/optional_input_null_required_output_161001.md`
- **可选输出null时占位tensor致507035** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/optional_output_null_placeholder_507035.md`
- **PadMode/单参 EnQue 在 CANN9.0.0 编译失败** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/padmode_enque_cann9_0_0.md`
- **torch_npu dispatcher ops and a raw ACLRT_LAUNCH_KERNEL are NOT stream-ordered against each other** · 401 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md`
- **fixpipe L0C(fp32)→UB(fp16) on dav-m200 SATURATES at ±65504, never inf** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_107_fixpipe_l0c_fp32_ub_fp16_on_dav_m200_saturates_a.md`
- **Duplicate<uint8_t> is REJECTED on dav-m200 (SDK dtype whitelist: half/int16/uint16/int32/uint32/float only)** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_108_duplicate_uint8_t_is_rejected_on_dav_m200_sdk_dt.md`
- **aclnnGeGluV2 header documents approximate=1 as the erf("none") path, but gen_data.py golden + ComputeGeluBase** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_109_aclnngegluv2_header_documents_approximate_1_as_t.md`
- **Shared NPU Contention** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_10_shared_npu_contention.md`
- **CANN ARPESmall (apply_rotary_pos_emb_small.h) silently ignores rotary_mode — always half-mode rotation.** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_110_cann_arpesmall_apply_rotary_pos_emb_small_h_sile.md`
- **CANN aclnnIncreFlashAttention (torch_npu.npu_incre_flash_attention) aicore-TIME-OUTS after ~8 sequential invoc** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_111_cann_aclnnincreflashattention_torch_npu_npu_incr.md`
- **Select with SELMODE::VSEL_CMPMASK_SPR applies the compare mask with PERIOD 64 elements for count > 64 on dav-m** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_112_select_with_selmode_vsel_cmpmask_spr_applies_the.md`
- **ShiftLeft / ShiftRight are NOT SUPPORTED on dav-m200 and fail SILENTLY (ASCENDC_REPORT_NOT_SUPPORT in ext:impl** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_113_shiftleft_shiftright_are_not_supported_on_dav_m2.md`
- **Ascend310P torch_npu F.gelu / aclnnGelu IGNORES the approximate flag — vendor 'none' is BITWISE the tanh appro** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_114_ascend310p_torch_npu_f_gelu_aclnngelu_ignores_th.md`
- **torch_npu lazy host task queue vs raw ACLRT_LAUNCH_KERNEL: a still-queued vendor op can OVERWRITE the raw-laun** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_115_torch_npu_lazy_host_task_queue_vs_raw_aclrt_laun.md`
- **Ascend310P torch_npu has NO fp64 — double dtype silently replaced with float** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_116_ascend310p_torch_npu_has_no_fp64_double_dtype_si.md`
- **VEC fp32 arithmetic (Muls/Adds) works on dav-m200 (2026-08-13).** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_118_vec_fp32_arithmetic_muls_adds_works_on_dav_m200.md`
- **dav-m200 fp32→fp16 Cast supports ONLY CAST_NONE/CAST_ODD — every rounding mode produces garbage/assert (VERIFI** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_119_dav_m200_fp32_fp16_cast_supports_only_cast_none.md`
- **dav-m200 vconv accepts only a per-direction cast-mode whitelist** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_120_dav_m200_vconv_accepts_only_a_per_direction_cast.md`
- **torch.nonzero on Ascend310P3 NPU silently returns CONTENT-CORRUPTED output for rank≥2** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_121_torch_nonzero_on_ascend310p3_npu_silently_return.md`
- **CANN ascendc cmake incremental device-object merge is flaky** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_125_cann_ascendc_cmake_incremental_device_object_mer.md`
- **`Adds` dtype set is NARROWER than `Duplicate`'s on dav-m200 — no uint32 (compile-time static_assert); per-API dtype whitelists differ WITHIN one SoC, so byte-identity bridges on u32-carried patterns m** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_126_adds_dtype_set_is_narrower_than_duplicate_s_on_d.md`
- **torch_npu FFT is DEAD on Ascend310P3 — `torch.fft.rfft/c2c` hits the complex-dtype gate, and `set_compile_mode(jit_compile=False)` only opens the gate before FFTc2c fails at aclnnMatmul 161002 (determ** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_128_torch_npu_fft_is_dead_on_ascend310p3_torch_fft_r.md`
- **SOC_VERSION Ascend910B2 causes 507035 on Ascend950PR hardware** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_12_soc_version_ascend910b2_causes_507035_on_ascend950.md`
- **Branchless merge requires `CHUNK >= TOPK_CAP` invariant** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_14_branchless_merge_requires_chunk_topk_cap_invariant.md`
- **Parallel NPU launches on same docker container produce cross-kernel state pollution** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_15_parallel_npu_launches_on_same_docker_container_pro.md`
- **dav-m200 VEC `Cos`/`Sin` do NOT IEEE-handle ±inf input — `Cos(+inf)=1.0` / `Sin(+inf)=+inf` instead of NaN (fp64 golden is NaN); scalar-detect the non-finite position and write canonical qNaN** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_177_dav_m200_vec_cos_sin_do_not_ieee_handle_inf_inpu.md`
- **dav-m200 composed-function VEC kernel faults deterministically — aicore 507015 / error 0x100000 / vec error info 0x1038 (stable) in the FIRST tile — while every sub-mechanism passes individually in th** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_185_dav_m200_composed_function_vec_kernel_faults_det.md`
- **dav-m200 out-of-range GM reads do NOT fault — a scalar load at GM offset 2^34 silently returns 0xffff and the kernel completes; "zero observable writes + no error" can therefore be a MIS-ADDRESSED acc** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_186_dav_m200_out_of_range_gm_reads_do_not_fault_a_sc.md`
- **`Cast<float,float>` (same-width VEC cast) is runtime NOT_SUPPORTED on dav-m200 — the vconv dtype-assert list admits `Tuple<float, float>` but the intrinsics impl reports `ASCENDC_REPORT_NOT_SUPPORT`; ** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_187_cast_float_float_same_width_vec_cast_is_runtime.md`
- **CANN `recompile_binary.py` packs host objects with `glob("**/*.o", recursive=True)` which does NOT traverse dot-directories — a build task dir whose path contains a dot-component makes the installed h** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_188_cann_recompile_binary_py_packs_host_objects_with.md`
- **dav-m200 — a CPU-staged tensor's host `data_ptr()` passed to the kernel as a GM operand is read as GM SILENTLY: output all-NaN, kernel completes, zero error codes (scalar GM loads perform no pointer-p** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_189_dav_m200_a_cpu_staged_tensor_s_host_data_ptr_pas.md`
- **CANN install path drift on npu_dev3 — `/usr/local/Ascend/cann-9.0.0` empty, real install at `/data/cann_b103/cann-9.0.0`** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_18_cann_install_path_drift_on_npu_dev3_usr_local_asce.md`
- **AscendC VEC `Sin()` / `Cos()` lack Payne-Hanek argument reduction — ±inf / huge-error on |x| ≥ 1e10** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_19_ascendc_vec_sin_cos_lack_payne_hanek_argument_redu.md`
- **`GlobalTensor<T>::SetValue(idx, val)` is silent no-op on Ascend950PR** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_20_globaltensor_t_setvalue_idx_val_is_silent_no_op_on.md`
- **V220 TBuf `GetValue` returns sequential-position values (not stored data) when interleaved with TQue `CopyTile` operations [V220]** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_24_v220_tbuf_getvalue_returns_sequential_position_val.md`
- **AscendC `Sigmoid<fp32>` primitive — uniform 2-ULP floor (slight correction to OL-103 "1-ULP" claim)** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_27_ascendc_sigmoid_fp32_primitive_uniform_2_ulp_floor.md`
- **`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` macro is arch35-only — `RegisterAscendBinary 107000` on V220 [V220]** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_28_kernel_task_type_default_kernel_type_aiv_only_macr.md`
- **`add_modules_sources` called twice for same op → silent target conflict in `generate_bin_scripts` [V351, build-system]** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_29_add_modules_sources_called_twice_for_same_op_silen.md`
- **TQue<VECIN,2> Data Corruption** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_2_tque_vecin_2_data_corruption.md`
- **op_host/ archive incomplete — patches shipped instead of complete files [V351+all-modes, finalize-pipeline]** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_33_op_host_archive_incomplete_patches_shipped_instead.md`
- **AscendC `Cast<float, uint8_t>` is unsupported — silent garbage, not an error** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_37_ascendc_cast_float_uint8_t_is_unsupported_silent_g.md`
- **Auto-generated host_stub.cpp `FreeAscendMemDevice` races with async kernel execution on V220 [V220]** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_38_auto_generated_host_stub_cpp_freeascendmemdevice_r.md`
- **bisheng `--enable-simt` codegen — `select i1` on `dav-c310-vec` fires `Copy register different width` [V351/A5, --enable-simt-only]** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_39_bisheng_enable_simt_codegen_select_i1_on_dav_c310.md`
- **NPU Device 0 Post-Reboot Failure** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_3_npu_device_0_post_reboot_failure.md`
- **`RegisterAscendBinary mix ret 107000` printed at teardown on `KERNEL_TYPE_MIX_AIC_1_2` V220 — non-fatal, independent of matmul-primitive [V220, mixed-mode-binary-register]** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_40_registerascendbinary_mix_ret_107000_printed_at_tea.md`
- **V220 manual-cube operand-load forms COMPILE + run clean on A5 but compute garbage — build-success is NOT A5-validation [V351/A5, port_a3, cube, silent-wrong-result]** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_43_v220_manual_cube_operand_load_forms_compile_run_cl.md`
- **`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` → `RegisterAscendBinary aiv 107000` register-FAIL on A5/950PR under the CANN 9.0.0 toolkit; CANN 9.1.T500 registers clean — A5-safety is toolkit-versio** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_44_kernel_task_type_default_kernel_type_aiv_only_regi.md`
- **chunk-loop UB write-after-read hazard — a per-tile buffer RELOADED via MTE2 at the top of each loop iteration and consumed by V in the SAME iteration needs a `V→MTE2` fence at the ITERATION BOUNDARY** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_47_chunk_loop_ub_write_after_read_hazard_a_per_tile_b.md`
- **SIMT GM-scratch written then read ACROSS grid-stride iterations is not coherent even with `asc_threadfence` — prefer all-UB recompute-from-boundary over GM staging for cross-iteration carries** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_48_simt_gm_scratch_written_then_read_across_grid_stri.md`
- **chunk-loop CROSS-ROW UB write-after-read hazard — a SHARED per-tile buffer written by V in one grid-stride ROW and reloaded via MTE2 at the top of the NEXT row needs a `V→MTE2` fence at the ROW bounda** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_49_chunk_loop_cross_row_ub_write_after_read_hazard_a.md`
- **Ascend950PR int8 `Cast` silently drops tail elements when count not multiple of 32 (VEC 32B width)** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_50_ascend950pr_int8_cast_silently_drops_tail_elements.md`
- **standalone-pybind MIX (cube+vec) `507014` silent-hang is a DOUBLE-BOOTSTRAP — the `auto_gen_*_kernel` device wrapper ALREADY did the full KFC/workspace bootstrap; a kernel that RE-RUNS `SetSysWorkspac** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_53_standalone_pybind_mix_cube_vec_507014_silent_hang.md`
- **`MIX_AIC_1_2` AIC↔AIV cross-core handshake is DIRECTION-ASYMMETRIC — the reverse (AIV→AIC) flag is per-subblock-COUNTED, not broadcast → a single-setter reverse handshake DEADLOCKS [220x/Ascend910_938** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_55_mix_aic_1_2_aic_aiv_cross_core_handshake_is_direct.md`
- **a3/220x multi-core (blockDim>1) MIX cube↔vec LOOPED bidirectional handshake deadlocks (507014)** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_56_a3_220x_multi_core_blockdim_1_mix_cube_vec_looped.md`
- **-O2 Required for NPU (No -O0)** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_5_o2_required_for_npu_no_o0.md`
- **Zombie Process Accumulation** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_6_zombie_process_accumulation.md`
- **aclrtlaunch Stub Requires extern "C" Declaration** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_8_aclrtlaunch_stub_requires_extern_c_declaration.md`
- **per-token 动态量化 arch35 kernel 完整配方** · 395 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/per_token_dynamic_quant_recipe.md`
- **直调共享库加载未定义平台符号—补齐CANN链接依赖** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/platform_manager_shared_library_link_dependencies.md`
- **PlatformAscendC 默认构造删除，须传平台指针** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/platformascendc_no_default_ctor.md`
- **proto.h ATTR 无默认值须改 REQUIRED_ATTR** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/proto_attr_no_default_use_required_attr.md`
- **proto 输出序以 spec/aten 序为准，勿照抄 vendor 基线** · 195 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/proto_output_order_follows_spec_aten_order.md`
- **纯 SIMT kernel 形参 __gm__ 导致编译失败** · 203 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pure_simt_parameters_do_not_use_gm_qualifier.md`

下一页：page-024.md
