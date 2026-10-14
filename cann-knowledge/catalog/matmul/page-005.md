# matmul: 281–350 / 567

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Hand-rolled Mmad: output-n and contraction-k axes spanning >1 C0 block (>BASE_K=16) MUST be tiled in BASE_K chunks (+accumulate for k)** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p100_hand_rolled_mmad_output_n_and_contraction_k_axes.md`
- **Single-AIC GEMM with constexpr static tiling + on-stack TCubeTiling** · 115 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p68_single_aic_gemm_with_constexpr_static_tiling_on.md`
- **Manual-Mmad B-operand contraction axis = the L1 tile's C0/inner dim; A and B must source the contraction from the SAME axis** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p99_manual_mmad_b_operand_contraction_axis_the_l1_ti.md`
- **REGIST_MATMUL_OBJ arch35 MIX AIC L1 崩溃** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507015_regist_matmul_obj_arch35_mix_l1_crash.md`
- **507035 全载超 UB：大 groupSize 分载三遍扫描** · 196 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/507035_group_stats_fullload_colsplit.md`
- **950 Matmul高阶API选型先查内置源** · 182 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/950_matmul_api.md`
- **aclnnQuantMatmulV4 直调 161002 根因判别** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnnquantmatmulv4_161002.md`
- **aclnnQuantMatmulV5 FP16 x2Scale 须 UINT64** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnnquantmatmulv5_fp16_x2scale_uint64.md`
- **aclnnQuantMatmulV5 L0C2out 三约束判别** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/aclnnquantmatmulv5_l0c2out.md`
- **arch35 MIX Matmul L1 常驻接入模式（swin 参考）** · 191 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/arch35_mix_matmul_l1_resident_swin_pattern.md`
- **errcode 82 — streamk stepK 致 L1 越界** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/conv3d_bp_filter_v2_streamk_stepk_l1_overflow_errcode_82.md`
- **DAV_3510 Matmul API 无先例走纯 AIV 回退** · 275 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/dav3510_matmul_api_pure_aiv_fallback.md`
- **Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_39_cube_matmulimpl_mm_cfg_cfg_norm_rejected_matmulcon.md`
- **`REGIST_MATMUL_OBJ` lands NO-SYNC branch with single `-DASCENDC_MATMUL_AICORE` — runtime MPU 507015 once cube uses the matmul object [V220+V351, ALL_MODES, build-config + KFC-sync-activation]** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_57_regist_matmul_obj_lands_no_sync_branch_with_single.md`
- **`matmul_intf.h` is NOT included transitively from `kernel_operator.h` — must be explicit `#include` OR per-pass interface-library propagation [V220+V351, ALL_MODES, build-include + KFC-sync-activation** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_58_matmul_intf_h_is_not_included_transitively_from_ke.md`
- **FullloadR softmax kernel AR/ARA 实现配方** · 586 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/fullloadr_softmax_ar_ara_kernel_recipe.md`
- **kernel Matmul 头在 ascendc 树且别名在 intf** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/kernel_matmul_intf_include_tree.md`
- **Matmul 高阶 API 头不可见：ini 与 include 补配** · 249 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/matmul_highlevel_api_include_paths_ini.md`
- **Matmul 头路径 lib/matmul 不存在，应为 adv_api** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/matmul_include_path_lib_to_adv_api.md`
- **MIX_AIC_1_2 kernel 必须 include lib/matmul_intf.h——compile_op.py 注入的 clearWorkspace 在 alias-only 的 matmul.h 下无法解析** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/mix_kernel_clear_workspace_include_matmul_intf.md`
- **matmul::Matmul<> DOES build and run inside the build_ascendc.py pybind harness when the CANN host tiling libs** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_508_matmul_matmul_does_build_and_run_inside_the_buil.md`
- **dav-m200 cube is available via raw Mmad — only MatmulImpl<>/Fixpipe/LoadData2D are arch-gated (2026-08-13).** · 54 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_117_dav_m200_cube_is_available_via_raw_mmad_only_mat.md`
- **`MatmulImpl<>` with manual `CrossCoreSetFlag`/`WaitFlag` + `MIX_AIC_1_2` deadlock on V220 [V220, mixed-mode-sync]** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md`
- **Mmad/HAVE_WORKSPACE kernel — host pybind MUST allocate sysWs + userWs (kernel gets `GetUserWorkspace(w)=w+sysWsSize`) [V220, workspace, matmul]** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_41_mmad_have_workspace_kernel_host_pybind_must_alloca.md`
- **V220 cube/AIC GEMM must NOT Fixpipe directly to the fp16/bf16 OUTPUT tensor — route through an fp32 workspace + a vec cast [V220, fixpipe, matmul]** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_42_v220_cube_aic_gemm_must_not_fixpipe_directly_to_th.md`
- **passing `GetSysWorkSpacePtr()` to `REGIST_MATMUL_OBJ` (unset/invalid for a plain `ascendc_library` stub) → the matmul lib's fixpipe L0C→GM write targets an MPU-protected GM address → `507015` AIC fixp** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_54_passing_getsysworkspaceptr_to_regist_matmul_obj_un.md`
- **REGIST_MATMUL_OBJ 须入口调用且首参传 &pipe** · 233 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/regist_matmul_obj_kernel_entry_tpipe_pointer.md`
- **双Matmul类型不一致致REGIST_MATMUL_OBJ重定义冲突** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/regist_matmul_obj_type_unify.md`
- **SetMatmulConfigParams 枚举需命名空间前缀** · 196 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/setmatmulconfigparams_enum_namespace.md`
- **TSCM A/B 大 K GetTiling 失败：L1 预算+K 切分** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tscm_matmul_large_k_l1_budget_k_split.md`
- **纯AIV内积matmul大batch超时507035** · 210 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/aiv_vec_matmul_large_batch_timeout_507035.md`
- **FP16 NZ-format Cube matmul IS usable on Ascend310P3/dav-m200 even though only INT8/FP4-quantized cube attentio** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md`
- **bf16 多 matmul 链累积漂移判别** · 149 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/bf16_matmul.md`
- **V220 hand-written bare-mmad cube paths silently miscalculate on A5 (L0A transpose misplacement at blockM>=2) — route-guard them off and keep MatmulImpl (PB-43 cross-gen instance)** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/cube_manual_mmad_crossgen_001.md`
- **Matmul workspace 预偏移 totalN 回读 banned** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/matmul_workspace_totaln_banned.md`
- **V351 forward FA — cube-MatmulImpl P@V is required to match CANN precision; vec-only scalar-broadcast P@V drifts +5-25pp mare at large n_kv_tile** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_186_v351_fa_cube_matmul_pv_required.md`
- **A Cube fp32 matmul precision residual on a cancellation case is `baseK`-tiling-INVARIANT — it is reference-ub (accumulation order vs the vendor Mmad), not a tunable, so do not sweep tiles to fix it** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_276_a_cube_fp32_matmul_precision_residual_on_a_cancell.md`
- **HFusion 矩阵乘操作** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hfusion/matmul_ops.md`
- **HIVM 宏操作 BatchMMAD-L1（hivm.hir.batchMmadL1）** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/macro_batch_mmad_l1.md`
- **HIVM 宏操作 Matmul（hivm.hir.matmul）** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/macro_matmul.md`
- **HIVM 宏操作 MixGroupMatmul（hivm.hir.mix_group_matmul）** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/macro_mix_group_matmul.md`
- **HIVM 宏操作 MixMatmul（hivm.hir.mix_matmul）** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/macro_mix_matmul.md`
- **HIVM 宏操作 MMAD-L1（hivm.hir.mmadL1）** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/macro_mmad_l1.md`
- **GEMM 基础代码骨架** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/gemm_basic.md`
- **Matmul 教学 Walkthrough** · 193 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/matmul_walkthrough.md`
- **矩阵乘法 Swizzle2D 优化案例** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_matmul_swizzle2d.md`
- **Fused Matmul GPU→NPU 迁移对照** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/fused_matmul_migration.md`
- **MatMul / GEMM 高性能算子** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/matmul.md`
- **GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant） 实现手册** · 348 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__groupedmatmulswigluquant.md`
- **GroupedMatmulSwigluQuantV2 算子（grouped-matmul-swiglu-quant-v2） 实现手册** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__groupedmatmulswigluquantv2.md`
- **QuantMatmul 算子（quantization-matmul） 实现手册** · 218 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__quantmatmul.md`
- **WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication） 实现手册** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/quantization__weightquantbatchmatmul.md`
- **MatMul/GEMM 优化经验聚合** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/matmul_experiences.md`
- **Matmul 链中间 buffer dtype 优化** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/chained_matmul_buffer_dtype.md`
- **MatMul 算子优化** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_matmul.md`
- **Ascend CV 融合算子生成方法论 — Batch 流水线篇** · 458 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/cv_fusion_pingpong.md`
- **Ascend Matmul Transpose 优化** · 340 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/matmul_transpose.md`
- **MatMul 族策略选型指南** · 257 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/families/matmul_guide.md`
- **Matmul 系优化工作流（含 CV 融合 / MoE）** · 53 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/scenario-routing/matmul.md`
- **Ascend C GMM 高阶 API Tiling 策略** · 313 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/matmul/ascendc-api-gmm-tiling.md`
- **Ascend C Matmul 高阶 API Tiling 策略** · 408 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/matmul/ascendc-api-matmul-tiling.md`
- **MatMul 类算子场景路由** · 32 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/matmul/patterns.md`
- **Reduce 类算子 - AR 全载分支** · 151 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/ar-fullload.md`
- **Reduce 类算子 - ARA 全载分支** · 228 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/ara-fullload.md`
- **Ascend C Matmul 高阶 API 最佳实践** · 364 行 · upstream
  `cannbot-skills/ops/ascendc-api-best-practices/references/api-matmul.md`
- **block_mmad_glu** · 238 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/block/block_mmad_glu.h`
- **block_scheduler_group_matmul** · 106 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/block/block_scheduler_group_matmul.h`
- **weight_quant_matmul_block_mmad** · 362 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/block/weight_quant_matmul_block_mmad.h`
- **group_matmul_kernel_cv1_v2** · 150 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/kernel/group_matmul_kernel_cv1_v2.h`
- **group_matmul_kernel_glu_fused** · 345 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/kernel/group_matmul_kernel_glu_fused.h`

下一页：page-006.md
