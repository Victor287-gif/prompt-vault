---
type: Guide
title: RoPE 位置编码
description: Triton Ascend RoPE 位置编码的多模态变体（标准/Partial/Multi-RoPE）、向量化旋转与融合策略。
tags: [task.design, topic.compute.elementwise.arithmetic, topic.compute.composite, topic.programming_model.execution_unit.vector]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sin.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# RoPE 位置编码

RoPE 通过复数旋转实现位置编码，将 query/key 的每对相邻维度进行 `cos`/`sin` 旋转变换。包含多种变体：标准 RoPE、Partial RoPE（Llama-3）、Multi-RoPE（Qwen2-VL）、Neox 风格、SISO 布局等。

## 检测标准

- 含 cos/sin 旋转位置编码计算
- 算子描述含 "rotary" / "RoPE" / "position encoding"
- 特征：将输入切分为 pairs，应用 2D 旋转矩阵

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 标准 RoPE (hd≥64) | exp_rope_001 | 15% | cos/sin 复用+persistent grid-stride |
| NPU 优化 (hd≥64) | exp_rope_npu_001 | 35% | cos/sin 复用+software pipeline+前后向共用 |
| RoPE 前向 varlen | exp_rope_forward_001 | 25% | ALIGNED 路径+do_not_specialize |
| RoPE 反向 varlen/KV | exp_rope_backward_001 | 25% | grid-stride+IS_VARLEN+HAS_KV_LENS |
| Partial RoPE (Llama-3) | exp_partial_rope_001 | 25% | 跳过 NOPE_DIM+向量化前/后半 |
| Qwen2-VL Multi-RoPE | exp_qwen2vl_mrope_001 | 25% | t/h/w 三维融合+head blocking |
| Llama-4 NPU RoPE | exp_rope_llama4_npu_001 | 25% | UB tiling+FMA+interleaved |
| SISO 布局 | exp_rope_siso_001 | 25% | 3D tiling+非交错向量化 |
| Neox/非Neox 双路 | exp_triton_rope_001 | 25% | IS_NEOX_STYLE constexpr |

## 关键技术

- **cos/sin 复用**：预计算频率表，循环内复用
- **constexpr 特化**：`IS_NEOX_STYLE`/`IS_VARLEN`/`HAS_KV_LENS` 编译期分支消除
- **Partial RoPE**：ROPE_DIM < head_dim 时只旋转前半部分，跳过 NOPE_DIM
- **Multi-RoPE**：t/h/w 三维独立频率，head blocking 分组复用

