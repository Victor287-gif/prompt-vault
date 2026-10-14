# triton: 141–210 / 465

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **MemRefExt 方言** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/other_dialects/memrefext_dialect.md`
- **Scope 方言** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/other_dialects/scope_dialect.md`
- **Symbol 方言** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/other_dialects/symbol_dialect.md`
- **属性枚举速查索引** · 119 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/reference/attribute_enum_index.md`
- **术语表** · 122 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/reference/glossary.md`
- **硬件规格 IR 映射** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/reference/hardware_specs_ir.md`
- **IR 操作速查索引** · 145 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/reference/ir_operation_index.md`
- **Gluon 实验性方言** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/gluon_dialect.md`
- **Triton 方言体系总览** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/overview.md`
- **Triton 方言操作详解（tt.*）** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/triton_ops.md`
- **Triton 类型与属性详解** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/triton_types_attrs.md`
- **TritonGPU 布局编码详解** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/tritongpu_encodings.md`
- **TritonGPU 方言操作详解（ttg.*）** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/tritongpu_ops.md`
- **原子操作 API 及 NPU 限制** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_atomic_ops.md`
- **比较、逻辑、移位与 where 操作 API** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_comparison_logical_ops.md`
- **线性代数 API（tl.dot / tl.dot_scaled）与 Cube 管线** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_linear_algebra_ops.md`
- **数学运算 API 及 NPU 精度提升规则** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_math_ops.md`
- **内存操作 API（tl.load / tl.store / block_ptr）** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_memory_ops.md`
- **归约操作 API** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_reduction_ops.md`
- **扫描与排序操作 API** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_scan_sort_ops.md`
- **张量形状操作 API** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_shape_ops.md`
- **Triton Ascend API 速查** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/triton_api_cheatsheet.md`
- **自定义算子示例（Custom Op on Triton-Ascend）** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/custom_op_example.md`
- **Flash Attention Causal 代码骨架** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/flash_attention_causal.md`
- **Flash Attention 教学 Walkthrough** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/flash_attention_walkthrough.md`
- **融合 Softmax 示例（Fused Softmax on NPU）** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/fused_softmax.md`
- **GEMM 基础代码骨架** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/gemm_basic.md`
- **LayerNorm 示例（前向融合 + 反向自旋锁分组归约）** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/layer_norm.md`
- **Matmul 教学 Walkthrough** · 193 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/matmul_walkthrough.md`
- **MoE Dispatch 代码骨架** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/moe_dispatch.md`
- **归约操作模式集（Reduction Patterns）** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/reduction_pattern.md`
- **RMSNorm 融合代码骨架** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/rmsnorm_fused.md`
- **向量加法示例（Vector Add on NPU）** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/vector_add.md`
- **反模式 — 在 forward 中查询设备属性** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/ap_device_query_in_forward.md`
- **attention 类算子设计约束** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/attention_constraints.md`
- **Attention 家族写法分类（定 category 的路由索引）** · 255 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/attention_writing_classification.md`
- **block_sparse_attention 类算子设计约束（三·稀疏类·块跳过/top-k）** · 488 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/block_sparse_attention_constraints.md`
- **BLOCK_SIZE 配置选择与边界处理策略** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/blocksize_and_boundary_policy.md`
- **跨轴 3D Broadcast 优化案例** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_elemwise_broadcast_3d.md`
- **Int8 到 FP16 类型转换优化案例** · 56 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_elemwise_cast.md`
- **Slice + Concat 融合算子优化案例** · 71 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_elemwise_concat.md`
- **Zeros 创建张量优化案例** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_elemwise_zeros.md`
- **小 bins 直方图 / 小输出表规约草图设计** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_histogram_small_bins.md`
- **Histogram 直方图统计优化案例** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_index_histogram.md`
- **Index Put 索引赋值优化案例** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_index_put.md`
- **矩阵乘法 Swizzle2D 优化案例** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_matmul_swizzle2d.md`
- **大规模 Amax 归约优化（reduce轴映射多核）** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_reduction_amax_large.md`
- **Amin 归约原子操作优化案例** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_reduction_amin_atomic.md`
- **@architecture_decision("ub-aware-row-block", reason="TopK 每行计算量小，需通过 ROWS_PER_BLOCK 提升每 program 计算密度；行数受 UB 容量约束")** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_sort_topk.md`
- **Attention 算子优化** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_attention.md`
- **Triton Ascend Layout-transform / Permutation 代码生成要点** · 155 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_layout_transform.md`
- **Reduce 算子优化** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_reduce.md`
- **Sort/Select 算子优化** · 282 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_sort_select.md`
- **convolution 类算子设计约束** · 306 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/convolution_constraints.md`
- **cv_attn_agg 类算子设计约束（CV-Attn-聚合分发）** · 203 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/cv_attn_agg_constraints.md`
- **diffusion_embedding 类算子设计约束（扩散模型条件嵌入）** · 137 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/diffusion_embedding_constraints.md`
- **dual_branch_attention 类算子设计约束（CV 双分支特征图门控）** · 310 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/dual_branch_attention_constraints.md`
- **elementwise 类算子设计约束（逐元素激活）** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/elementwise_constraints.md`
- **flash_attention 类算子设计约束** · 262 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/flash_attention_constraints.md`
- **frequency_filter 类算子设计约束（频域滤波 / 纯 Triton DFT）** · 123 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/frequency_filter_constraints.md`
- **Triton Ascend 全局设计约束** · 314 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/global_constraints.md`
- **gqa 类算子设计约束（二·压缩类·head 共享）** · 499 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/gqa_constraints.md`
- **index_computation 类算子设计约束** · 192 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/index_computation_constraints.md`
- **Layout-transform / Permutation 算子设计要点** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/layout_transform_design.md`
- **linear 类算子设计约束（二·压缩类·低秩投影）** · 400 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/linear_constraints.md`
- **linear_recurrent 类算子设计约束（状态递推）** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/linear_recurrent_constraints.md`
- **math_compute 类算子设计约束** · 207 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/math_compute_constraints.md`
- **matrix_compute 类算子设计约束** · 236 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/matrix_compute_constraints.md`
- **mha 类算子设计约束（一·标准类·含投影段）** · 687 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/mha_constraints.md`
- **mla 类算子设计约束（MultiHeadLatentAttention 矩阵吸收）** · 216 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/mla_constraints.md`

下一页：page-004.md
