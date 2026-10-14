# triton: 281–350 / 465

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **IOU 算子（Intersection over Union） 实现手册** · 237 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/object_detection__iou.md`
- **NMS 算子（Non-Maximum Suppression） 实现手册** · 347 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/object_detection__nms.md`
- **Paged KV Cache Attention 算子（decode / prefill） 实现手册** · 230 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/paged_attention__paged_kv_cache_attention.md`
- **StorePagedKVCache 算子（Scatter-Copy） 实现手册** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/paged_memory_copy__store_paged_kv_cache.md`
- **MobileViT 系局部 transformer（patch_window_attention） 实现手册** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/patch_window_attention__mobilevit_local_transformer.md`
- **DAModule（DANet 双重注意力模块） 实现手册** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/position_channel_attention__damodule.md`
- **DynamicQuant 算子（reduction + element-wise 复合） 实现手册** · 344 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/quantization__dynamicquant.md`
- **SwigluQuant 算子（quantization-activation） 实现手册** · 287 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/quantization__swigluquant.md`
- **SwigluQuant 关键 kernel 实现** · 156 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/quantization__swigluquant_kernels.md`
- **WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication） 实现手册** · 263 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/quantization__weightquantbatchmatmul.md`
- **LSTM 算子（lstm） 实现手册** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/recurrent__lstm.md`
- **Top-P Filter 算子（核采样过滤） 实现手册** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/sampling__top_p_filter.md`
- **HaloAttention 算子（邻域展开） 实现手册** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/sparse_unfold__halo_attention.md`
- **EMSA 算子（PVT 系空间缩减注意力） 实现手册** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/spatial_attention__emsa.md`
- **Interpolate 算子（transformation-compute） 实现手册** · 380 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__interpolate.md`
- **Interpolate bicubic_area 路径关键 kernel** · 402 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__interpolate_bicubic_area.md`
- **Interpolate bilinear 路径关键 kernel** · 382 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__interpolate_bilinear.md`
- **Interpolate nearest 路径关键 kernel** · 175 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__interpolate_nearest.md`
- **Pad 算子（transformation-memory） 实现手册** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__pad.md`
- **Repeat 算子（transformation-memory） 实现手册** · 285 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/tensor_transform__repeat.md`
- **AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce） 实现手册** · 198 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__attentionsoftmaxwithsoftcappinganddropout.md`
- **LightningIndexer 算子（topk-index） 实现手册** · 161 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__lightningindexer.md`
- **MoeComputeExpertTokens 算子（indexing-gather / counting） 实现手册** · 179 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__moecomputeexperttokens.md`
- **MoeGatingTopKSoftmax 算子（sort-topk） 实现手册** · 232 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__moegatingtopksoftmax.md`
- **RotaryMul 算子（rotarymul） 实现手册** · 269 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__rotarymul.md`
- **SparseFlashAttention 算子（sparse-attention） 实现手册** · 256 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__sparseflashattention.md`
- **FusionAttention 算子（attention） 实现手册** · 266 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_training__fusionattention.md`
- **Triton Ascend API 参考** · 327 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/triton_ascend/api_reference.md`
- **Triton Ascend 完整代码示例** · 181 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/triton_ascend/code_examples.md`
- **Grid 启动限制与核心数查询** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/triton_ascend/grid_launch_limits.md`
- **编程基础：核心概念与标准 Kernel 结构** · 209 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/triton_ascend/programming_basics.md`
- **Block Pointer 迁移注意事项** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/block_pointer_migration.md`
- **编译错误排查（MLIRCompileError）** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/compile_error_debug.md`
- **调试环境变量完整参考** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debug_env_variables.md`
- **Triton-Ascend 调试方法总览** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debug_overview.md`
- **调试、稳定性与 API 规范** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debugging.md`
- **常见编译/运行错误速查（报错-原因-修复）** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debugging_common_errors.md`
- **错误码参考（报错信息对照表）** · 110 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/error_codes.md`
- **GPU→NPU 迁移概览：架构差异与迁移工作流** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/gpu_migration_overview.md`
- **解释器模式调试（TRITON_INTERPRET）** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/interpreter_mode.md`
- **Triton API 在 Ascend NPU 上的差异对照** · 162 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/migration_api_differences.md`
- **GPU→NPU 代码迁移模式（before/after）** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/migration_code_patterns.md`
- **GPU→NPU 迁移常见问题清单** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/migration_common_issues.md`
- **端到端性能优化流程** · 70 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/optimization_flow.md`
- **Profiling 方法论** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/profiling.md`
- **Triton-Ascend 性能分析与瓶颈定位指南** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/profiling_guide.md`
- **运行时错误排查** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/runtime_error_debug.md`
- **形状驱动算子决策树** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/shape_router.md`
- **Triton-Ascend 常见问题速查（FAQ）** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/triton_ascend_faq.md`
- **Triton-Ascend 环境搭建与第一个 Kernel** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/triton_ascend_quickstart.md`
- **激活函数优化经验聚合** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/activation_experiences.md`
- **Attention 优化经验聚合** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/attention_experiences.md`
- **Gather/Scatter 优化经验聚合** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/gather_scatter_experiences.md`
- **Ascend 平台专项经验聚合** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/general_ascend_experiences.md`
- **通用性能优化经验聚合** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/general_perf_experiences.md`
- **数值稳定性与调试经验聚合** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/general_stability_experiences.md`
- **MatMul/GEMM 优化经验聚合** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/matmul_experiences.md`
- **杂项优化经验聚合** · 51 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/misc_experiences.md`
- **归一化优化经验聚合** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/normalization_experiences.md`
- **归约优化经验聚合** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/reduction_experiences.md`
- **RoPE 优化经验聚合** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/rope_experiences.md`
- **Softmax 优化经验聚合** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/softmax_experiences.md`
- **SSM/KDA 优化经验聚合** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/ssm_experiences.md`
- **Adaptive Instance Normalization 2D Backward（AdaIN Bwd）优化经验** · 245 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/adain.md`
- **Ascend Interpolate 算子系统性优化** · 1183 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/ascend_interpolate_optimization.md`
- **Ascend Pooling 算子 Triton 系统性优化指南** · 741 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/ascend_pooling_optimization.md`
- **Autotune 自动调优** · 812 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/autotune.md`
- **标量降级** · 208 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/avoid_scalar_lowering.md`
- **Block Size Scaling 优化模式** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/block_size_scaling.md`
- **Matmul 链中间 buffer dtype 优化** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/chained_matmul_buffer_dtype.md`

下一页：page-006.md
