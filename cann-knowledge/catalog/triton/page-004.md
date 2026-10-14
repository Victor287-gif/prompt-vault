# triton: 211–280 / 465

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **normalization 类算子设计约束** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/normalization_constraints.md`
- **object_detection 类算子设计约束** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/object_detection_constraints.md`
- **paged_attention 类算子设计约束（分页 KV Cache Attention）** · 271 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/paged_attention_constraints.md`
- **paged_memory_copy 类算子设计约束（Paged Memory Copy）** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/paged_memory_copy_constraints.md`
- **patch_window_attention 类算子设计约束（卷积-块化窗口 transformer）** · 248 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/patch_window_attention_constraints.md`
- **position_channel_attention 类算子设计约束（无投影位置/通道双分支）** · 298 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/position_channel_attention_constraints.md`
- **quantization 类算子设计约束** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/quantization_constraints.md`
- **recurrent 类算子设计约束** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/recurrent_constraints.md`
- **sampling 类算子设计约束（核采样过滤）** · 93 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/sampling_constraints.md`
- **sparse_mask 类算子设计约束（三·稀疏类·掩码型）** · 425 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/sparse_mask_constraints.md`
- **sparse_unfold 类算子设计约束（邻域展开 Sparse-展开型 attention）** · 208 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/sparse_unfold_constraints.md`
- **spatial_attention 类算子设计约束（CV-Attn-空间自注意力）** · 152 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/spatial_attention_constraints.md`
- **tensor_transform 类算子设计约束** · 175 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/tensor_transform_constraints.md`
- **transformer_inference 类算子设计约束** · 191 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/transformer_inference_constraints.md`
- **transformer_training 类算子设计约束** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/transformer_training_constraints.md`
- **vision_mlp 类算子设计约束（视觉 MLP）** · 214 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/vision_mlp_constraints.md`
- **功能说明** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/ir/ir_analysis_workflow.md`
- **功能说明** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/ir/ir_extraction.md`
- **功能说明** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/ir/migration_agent_workflow.md`
- **Flash Attention GPU→NPU 迁移对照** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/flash_attention_migration.md`
- **Fused Matmul GPU→NPU 迁移对照** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/fused_matmul_migration.md`
- **Fused SwiGLU GPU→NPU 迁移对照** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/fused_swiglu_migration.md`
- **RoPE GPU→NPU 迁移对照** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/rope_migration.md`
- **Softcap GPU→NPU 迁移对照** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/softcap_migration.md`
- **Attention 算子** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/attention.md`
- **Causal Conv1D 卷积算子** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/convolution.md`
- **Elementwise 算子优化** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/elementwise.md`
- **Embedding / 索引拷贝算子** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/embedding.md`
- **融合算子综合** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/fused_ops.md`
- **Gather/Scatter 算子** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/gather_scatter.md`
- **Loss 函数算子** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/loss.md`
- **MatMul / GEMM 高性能算子** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/matmul.md`
- **Normalization 归一化算子** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/normalization.md`
- **归约算子** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/reduction.md`
- **RoPE 位置编码** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/rope.md`
- **Softmax / LogSoftmax 算子** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/softmax.md`
- **solve_tril 三角矩阵求逆** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/solve_tril.md`
- **SSM / 状态空间模型算子** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/ssm.md`
- **MultiHeadAttention 算子（mha） 实现手册** · 136 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/attention__multiheadattention.md`
- **ConvDepthwise2d 算子（2D depthwise） 实现手册** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/convolution__convdepthwise2d.md`
- **ConvStandard1d 算子（1D 标准卷积） 实现手册** · 75 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/convolution__convstandard1d.md`
- **ConvStandard2d 算子（2D 标准/grouped 卷积） 实现手册** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/convolution__convstandard2d.md`
- **ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`） 实现手册** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/convolution__convstandard3d.md`
- **ConvTranspose2d 算子（2D 转置卷积） 实现手册** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/convolution__convtranspose2d.md`
- **CoTAttention 算子（cotattention） 实现手册** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/cv_attn_agg__cotattention.md`
- **DoubleAttention 算子（doubleattention） 实现手册** · 117 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/cv_attn_agg__doubleattention.md`
- **S2Attention 算子（s2attention） 实现手册** · 85 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/cv_attn_agg__s2attention.md`
- **FluxTimestepGuidanceProjectionEmbedding 算子（diffusion-embedding） 实现手册** · 178 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/diffusion_embedding__flux_timestep_guidance_projection_embedding.md`
- **BAM 算子（bam） 实现手册** · 102 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/dual_branch_attention__bam.md`
- **Polarized Self-Attention 算子（polarized_self_attention） 实现手册** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/dual_branch_attention__polarized_self_attention.md`
- **TripletAttention 算子（triplet_attention） 实现手册** · 99 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/dual_branch_attention__triplet_attention.md`
- **Activation 算子（GELU / SiLU / SwiGLU） 实现手册** · 100 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/elementwise__activation.md`
- **Add 算子（elementwise 逐元素加法）实现手册** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/elementwise__add.md`
- **Flash Attention 实现手册** · 577 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/flash_attention__flash_attention.md`
- **FA 精度闸门与测量口径** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/flash_attention__precision_gate_and_measurement.md`
- **EmbeddingDenseBackward 算子（index-accumulate） 实现手册** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/index_computation__embeddingdensebackward.md`
- **Index / Gather 算子（index-read） 实现手册** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/index_computation__index_gather.md`
- **IndexPut / Scatter 算子（index-write） 实现手册** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/index_computation__indexput_scatter.md`
- **Sort / TopK 算子（index-sort） 实现手册** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/index_computation__sort_topk.md`
- **FusedRecurrentGatedDeltaRule 算子（Gated Delta Rule） 实现手册** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/linear_recurrent__fused_recurrent_gated_delta_rule.md`
- **HyenaFftSizePaddingRfft 算子（transformation-compute） 实现手册** · 270 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/math_compute__hyenafftsizepaddingrfft.md`
- **Sort 算子（sort-topk） 实现手册** · 394 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/math_compute__sort.md`
- **Sum 算子（sum） 实现手册** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/math_compute__sum.md`
- **FFN（Feed-Forward Network）两级 GEMM 算子实现手册** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__ffn.md`
- **GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant） 实现手册** · 348 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__groupedmatmulswigluquant.md`
- **GroupedMatmulSwigluQuantV2 算子（grouped-matmul-swiglu-quant-v2） 实现手册** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__groupedmatmulswigluquantv2.md`
- **QuantMatmul 算子（quantization-matmul） 实现手册** · 218 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/matrix_compute__quantmatmul.md`
- **MLA 类算子（MultiHeadLatentAttention） 实现手册** · 160 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/mla__multi_head_latent_attention.md`
- **AdaptiveInstanceNorm2DBackward 算子（norm-backward / backward） 实现手册** · 232 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/normalization__adaptiveinstancenorm2dbackward.md`
- **GroupNormSwish 算子（norm-fused / forward） 实现手册** · 463 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/normalization__groupnormswish.md`

下一页：page-005.md
