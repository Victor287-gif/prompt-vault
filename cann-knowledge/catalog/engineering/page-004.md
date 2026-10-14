# engineering: 211–280 / 823

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Element-wise 算子优化** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_elementwise.md`
- **Triton-Ascend Interpolate 算子开发参考文档** · 235 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_interpolate.md`
- **MatMul 算子优化** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/coding_matmul.md`
- **compile_hint 与 Ascend 扩展 API 速查** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/compile_hint_extension.md`
- **FlashAttention 类算子 — 5 扩展点框架** · 173 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/flashattention/foundation/extension_points.md`
- **# Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 414 行 · community
  `cannbot-skills/community/ops/ascendc-ops-evaluation/scripts/build_ops.py`
- **----------------------------------------------------------------------------------------------------------** · 133 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-operator-project-init/templates/ascend-kernel/build.sh`
- **----------------------------------------------------------------------------------------------------------** · 358 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/scripts/build_ascendc.py`
- **block_mmad_glu** · 238 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/block/block_mmad_glu.h`
- **block_scheduler_group_matmul** · 106 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/block/block_scheduler_group_matmul.h`
- **weight_quant_matmul_block_mmad** · 362 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/block/weight_quant_matmul_block_mmad.h`
- **block_epilogue_dequant_swiglu** · 265 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/block_epilogue_dequant_swiglu.h`
- **block_epilogue_pertoken_quant_single_pass** · 252 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/block_epilogue_pertoken_quant_single_pass.h`
- **block_epilogue_pertoken_quant_two_pass_chunked** · 313 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/block_epilogue_pertoken_quant_two_pass_chunked.h`
- **block_epilogue_swiglu** · 171 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/block_epilogue_swiglu.h`
- **cv_sync_constants** · 32 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/cv_sync_constants.h`
- **epilogue_fusion_membase** · 212 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/epilogue_fusion_membase.h`
- **epilogue_fusion_regbase** · 324 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/epilogue_fusion_regbase.h`
- **online_softmax_cross_core_epilogue** · 396 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/online_softmax_cross_core_epilogue.h`
- **online_softmax_per_tile_epilogue** · 421 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/online_softmax_per_tile_epilogue.h`
- **group_matmul_kernel_cv1_v2** · 150 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/kernel/group_matmul_kernel_cv1_v2.h`
- **group_matmul_kernel_glu_fused** · 345 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/kernel/group_matmul_kernel_glu_fused.h`
- **weight_quant_matmul_kernel** · 611 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/kernel/weight_quant_matmul_kernel.h`
- **dispatch_policy** · 63 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/policy/dispatch_policy.h`
- **common_utils** · 101 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/utils/common_utils.h`
- **integral_constant** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/utils/integral_constant.h`
- **blaze_group_matmul_pertoken_quant_selector** · 119 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_group_matmul_pertoken_quant_selector.h`
- **blaze_group_matmul_tiling_data** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_group_matmul_tiling_data.h`
- **blaze_matmul_tiling** · 457 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_matmul_tiling.h`
- **blaze_matmul_tiling_data** · 50 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/matmul/blaze_matmul_tiling_data.h`
- **quant_matmul_mx_tiling** · 549 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/mx/quant_matmul_mx_tiling.h`
- **quant_matmul_mx_tiling_data** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/mx/quant_matmul_mx_tiling_data.h`
- **weight_quant_tiling** · 428 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/weight_quant/weight_quant_tiling.h`
- **weight_quant_tiling_data** · 59 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/op_tiling/weight_quant/weight_quant_tiling_data.h`
- **Blaze 组装方案恢复方法** · 106 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/blaze-investigation/assembly-recovery-method.md`
- **依赖追溯方法** · 140 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/blaze-investigation/dependency-trace-method.md`
- **Investigation 合同审计方法** · 78 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/blaze-investigation/design-contract-method.md`
- **Blaze Investigation 报告模板** · 249 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/blaze-investigation/investigation-report-template.md`
- **Blaze 源码调查方法** · 131 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/blaze-investigation/source-investigation-method.md`
- **Blaze 项目 CMake 编写指导** · 184 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-cmake-build-guide.md`
- **Blaze 框架总览** · 403 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-framework-overview.md`
- **Matmul Layout 与数据格式开发指南** · 322 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-matmul-layout.md`
- **Ascend C 同步指令编码指导** · 814 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-sync-patterns.md`
- **tensor_api 使用指导手册** · 589 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/tensor-api-reference.md`
- **{operator_name} Blaze 算子设计文档** · 593 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/blaze-design-template.md`
- **{operator_name} 算子开发计划** · 391 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/blaze-plan-template.md`
- **GLU / SwiGLU 组件开发指导** · 755 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/glu-development.md`
- **GLU 精度诊断** · 113 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/glu-precision-diagnosis.md`
- **Grouped MatMul 项目侧 Host/Tiling/ABI 与执行差异** · 299 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/group-matmul-delta.md`
- **Per-token Quant 组件开发指导** · 450 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/per-token-quant-development.md`
- **Tiling 选择与 Scheduler Params 合同** · 114 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/tiling-selection.md`
- **Launcher 开发指导** · 346 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/launcher/launcher-development.md`
- **Cube MatMul -> Vector Elementwise -> Vector Per-token Quant** · 196 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-design.md`
- **Cube MatMul -> Vector Elementwise -> Vector Per-token Quant 开发指导** · 104 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant/cube-matmul-then-vector-elementwise-then-vector-pertoken-quant-development.md`
- **Cube MatMul -> Vector GLU -> Vector Per-token Quant** · 189 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-design.md`
- **Cube MatMul -> Vector GLU -> Vector Per-token Quant 开发指导** · 52 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu-then-vector-pertoken-quant/cube-matmul-then-vector-glu-then-vector-pertoken-quant-development.md`
- **Cube MatMul -> Vector GLU** · 150 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-design.md`
- **Cube MatMul -> Vector GLU 开发指导** · 54 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-glu/cube-matmul-then-vector-glu-development.md`
- **Cube MatMul -> Vector Per-token Quant** · 138 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-design.md`
- **Cube MatMul -> Vector Per-token Quant 开发指导** · 45 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-pertoken-quant/cube-matmul-then-vector-pertoken-quant-development.md`
- **Cube MatMul → Vector Softmax/Reduce 场景设计指导** · 205 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-design.md`
- **Cube MatMul → Vector Softmax/Reduce 开发指导** · 99 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-development.md`
- **CV Sync 与两阶段 Kernel 编排专题** · 156 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv-sync-and-two-phase-entry.md`
- **从 group_matmul_kernel_cv1_v2.h 适配为 Softmax/Reduce 两阶段 Kernel** · 331 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv1-v2-adaptation-for-softmax-reduce.md`
- **Online Softmax/Reduce Epilogue 设计专题** · 174 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/online-softmax-reduce-epilogue-design.md`
- **Pure Reduce 适配指导** · 204 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/reduce-adaptation-guide.md`
- **Block层 L0C 输出与扩展** · 75 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/block-l0c2ub-extension.md`
- **Elementwise/Broadcast Epilogue Fusion 设计指导** · 280 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/elementwise-broadcast-epilogue-fusion-design.md`
- **Elementwise/Broadcast Epilogue Fusion 开发指导** · 202 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/elementwise-broadcast-epilogue-fusion-development.md`
- **MemBase Epilogue层设计与实现** · 82 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/epilogue-membase-design.md`

下一页：page-005.md
