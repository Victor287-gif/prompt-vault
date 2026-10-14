---
type: Guide
title: 融合算子综合
description: Triton Ascend 多算子融合组合总览：GEMM+Activation、QKV+Norm+RoPE、Flash Attention、MoE
  Scatter+GEMM 等 8 种融合模式。
tags: [task.design, topic.compute.composite, topic.compute.normalization, topic.attention, topic.compute.indexing]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 融合算子综合

融合的核心目标：消除中间 tensor 的 HBM 读写。每个融合组合一次性在 kernel 内完成多步计算。

## 融合组合速查

| 融合 | 场景 | 提升 |
|------|------|------|
| **Fusion-1** Linear+GELU/SiLU | FFN 第一层 | 30% |
| **Fusion-2** Linear+RMSNorm+RoPE | Attention QKV 投影 | 45% |
| **Fusion-3** Skip+LayerNorm | Transformer 残差块 | 30% |
| **Fusion-4** RMSNorm+Quantization | 量化推理归一化 | 50% |
| **Fusion-5** Flash Attention (QK^T+Softmax+AV) | 所有 Attention | 50% |
| **Fusion-6** MoE Scatter+Expert GEMM | MoE FFN | 35%+ |
| **Fusion-7** GEMM+Bias+Activation | 通用线性层 | 35% |
| **Fusion-8** Phase Split+Barrier 外提 | 多阶段反向 kernel | 10-30% |

## 核心模式

### GEMM + Activation (Fusion-1)

```
GEMM (Cube) → [L0C 寄存器] → GELU/SiLU (Vector) → Store
```

- 激活在 L0C 寄存器上操作，无额外 HBM 读写
- `dl.async_task` 分离 Cube/Vector 调度

### QKV + Norm + RoPE (Fusion-2)

```
x + residual → RMSNorm → QKV Split → RoPE
```

- 三合一单 kernel，消除 3 次 HBM 写
- 向量化 cos/sin 旋转

### Flash Attention (Fusion-5)

```
QK^T → Online Softmax → @V
```

- Online softmax 维护 running max/sum，O(S) 内存

### MoE Scatter + GEMM (Fusion-6)

```
Token Scatter (按 expert 分组) → Expert GEMM (独立)
```

- Binned dispatch 减少随机 scatter

