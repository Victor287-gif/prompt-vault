# 代码示例

可参考的 Triton Ascend kernel 代码骨架模板与逐步教学示例。

## 骨架模板

- [gemm_basic](gemm_basic.md)：带 AutoTune 的标准 GEMM kernel（ND layout, trans_b=True）
- [flash_attention_causal](flash_attention_causal.md)：Flash Attention causal 变体骨架
- [moe_dispatch](moe_dispatch.md)：MoE expert dispatch 骨架
- [rmsnorm_fused](rmsnorm_fused.md)：RMSNorm 融合变体骨架（Residual Add + RMSNorm）

## 教学示例（walkthrough）

- [vector_add](vector_add.md)：入门向量加法：grid 计算、mask 边界保护与 NPU 适配要点
- [fused_softmax](fused_softmax.md)：单 kernel 行级 softmax：数值稳定减最大值 + grid-stride 多行循环
- [matmul_walkthrough](matmul_walkthrough.md)：从 GPU 基线到 NPU 对角线分核 + dot_pad_only_k + multibuffer 优化版
- [layer_norm](layer_norm.md)：LayerNorm 前向融合 + 反向 atomic_cas 自旋锁分组归约
- [flash_attention_walkthrough](flash_attention_walkthrough.md)：Flash Attention 前向：STAGE 分块、online softmax（exp2 域）、make_block_ptr
- [reduction_pattern](reduction_pattern.md)：归约模式集：行/列/全局归约完整 kernel 与 NPU atomic 限制
- [custom_op_example](custom_op_example.md)：register_custom_op 注册 NPU 原生 CCE 自定义算子并在 kernel 中调用
