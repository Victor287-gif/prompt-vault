# 算子专项设计

按算子类型组织的专项设计页，每页覆盖该算子类型的经验路由、适用条件、关键技术要点与常见陷阱。
单算子实现手册（playbook）见 operators/（`../operator_playbooks/index.md`）。

- [matmul](matmul.md)：矩阵乘法/GEMM：Tiling、Core Mapping、融合变体、Cube 利用率
- [attention](attention.md)：Attention：Flash Attention、GQA/MQA、分页 KV Cache、Split-K
- [elementwise](elementwise.md)：逐元素计算与激活函数：SiLU/GELU/SwiGLU、向量化
- [normalization](normalization.md)：归一化：RMSNorm/LayerNorm/GroupNorm、融合模式
- [reduction](reduction.md)：归约：sum/max/min、Split 归约、原子归约
- [gather_scatter](gather_scatter.md)：Gather/Scatter/MoE 路由：binned dispatch、原子更新
- [softmax](softmax.md)：Softmax：online softmax、数值稳定性、log-softmax
- [rope](rope.md)：RoPE 位置编码：多模态 RoPE、partial RoPE、前反向共用
- [embedding](embedding.md)：Embedding：Dense/稀疏 backward、scatter add
- [convolution](convolution.md)：卷积：implicit GEMM、winograd
- [fused_ops](fused_ops.md)：融合算子：GEMM+Activation/Norm/RoPE 等组合模式
- [loss](loss.md)：损失函数：cross-entropy、MSE 等
- [solve_tril](solve_tril.md)：三角求解
- [ssm](ssm.md)：SSM/线性注意力/KDA：chunked scan、标量优化
