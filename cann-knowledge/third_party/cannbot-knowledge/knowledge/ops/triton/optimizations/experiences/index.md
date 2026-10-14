# 优化经验

按算子类别聚合的实战优化经验卡，从 303 条 YAML 经验蒸馏而来。每张卡提取共性优化模式，按质量分降序排列，只保留 stable + composite≥80 的高质量经验。

- [matmul_experiences](matmul_experiences.md)：MatMul/GEMM 优化经验（38 条蒸馏）
- [attention_experiences](attention_experiences.md)：Attention 优化经验（47 条蒸馏）
- [activation_experiences](activation_experiences.md)：激活函数优化经验（14 条蒸馏）
- [normalization_experiences](normalization_experiences.md)：归一化优化经验（33 条蒸馏）
- [reduction_experiences](reduction_experiences.md)：归约优化经验（4 条蒸馏）
- [gather_scatter_experiences](gather_scatter_experiences.md)：Gather/Scatter/MoE 优化经验（15 条蒸馏）
- [rope_experiences](rope_experiences.md)：RoPE 优化经验（10 条蒸馏）
- [softmax_experiences](softmax_experiences.md)：Softmax 优化经验（7 条蒸馏）
- [ssm_experiences](ssm_experiences.md)：SSM/KDA 优化经验（12 条蒸馏）
- [general_perf_experiences](general_perf_experiences.md)：通用性能优化经验（16 条蒸馏）
- [general_ascend_experiences](general_ascend_experiences.md)：Ascend 平台专项经验（16 条蒸馏）
- [general_stability_experiences](general_stability_experiences.md)：数值稳定性经验
- [misc_experiences](misc_experiences.md)：杂项融合经验（40 条蒸馏）

---

## 经验速查

从 303 条 YAML 经验蒸馏为 13 张聚合卡。按算子类别或优化维度导航。

## 按算子类别

| 类别 | 聚合卡 | 源经验数 | 典型提升 |
|------|--------|---------|---------|
| MatMul/GEMM | matmul_experiences | 38 | 10-40% |
| Attention | attention_experiences | 47 | 30-50% |
| 激活函数 | activation_experiences | 14 | 25-35% |
| 归一化 | normalization_experiences | 33 | 25-40% |
| 归约 | reduction_experiences | 4 | 20-35% |
| Gather/Scatter/MoE | gather_scatter_experiences | 13 | 25-35% |
| RoPE | rope_experiences | 10 | 25-35% |
| Softmax | softmax_experiences | 7 | 25-50% |
| SSM/KDA | ssm_experiences | 12 | 30%-3.33x |
| 杂项 | misc_experiences | 80 | — |

## 按优化维度

| 维度 | 聚合卡 | 核心主题 |
|------|--------|----------|
| 通用性能 | general_perf_experiences | Scalar 消除、block_ptr 替代、autotune（16 条） |
| Ascend 平台 | general_ascend_experiences | Grid/constexpr/禁用语法（16 条） |
| 数值稳定性 | general_stability_experiences | exp 防护、fp32 累加、调试（14 条） |

## 快速匹配

```
aiv_scalar_ratio > 30%       → general_perf_experiences (标量消除)
M,N ≥ 512 matmul             → matmul_experiences (对角 tiling)
seq ≥ 1024 attention          → attention_experiences (Flash Attention)
RMSNorm + Skip Add            → normalization_experiences (融合)
GQA/MQA                       → attention_experiences (多头复用 KV)
MoE routing                   → gather_scatter_experiences (Binned dispatch)
KDA 反向标量高                → ssm_experiences (i64→i32, block_ptr→手动)
exp 溢出 / NaN                → general_stability_experiences
```

全量经验索引（303 条 YAML）存于源插件仓库 `knowledge/raw/experiences/_index.yaml`，未随本次搬入。
