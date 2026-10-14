---
type: Optimization
title: RoPE 优化经验聚合
description: 从 10 条 RoPE YAML 经验中蒸馏的核心优化模式：cos/sin 复用、constexpr 多路特化与向量化旋转。
tags: [task.optimization, topic.compute.elementwise.arithmetic, topic.compute.composite, topic.programming_model.execution_unit.vector, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sin.md
  role: primary
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# RoPE 优化经验聚合

从 10 条 RoPE YAML 经验中蒸馏。RoPE 通过复数旋转实现位置编码，包含多种变体：标准、Partial（Llama-3）、Multi-RoPE（Qwen2-VL）、Neox 风格等。

## 共性模式

1. **cos/sin 复用**：预计算频率表，循环内复用
2. **constexpr 多路特化**：IS_NEOX_STYLE/IS_VARLEN/HAS_KV_LENS 编译期分支消除
3. **向量化旋转**：对相邻维度 pair 施加 2D 旋转矩阵 `[cos -sin; sin cos]`

## 经验汇总

| 经验 ID | 提升 | 核心优化 | 适用条件 |
|---------|------|----------|----------|
| exp_rope_npu_001 | 35% | cos/sin 复用+software pipeline+前后向共用 | Ascend NPU, hd≥64 |
| exp_rope_forward_001 | 25% | ALIGNED 路径+do_not_specialize | varlen 前向 |
| exp_rope_backward_001 | 25% | grid-stride+IS_VARLEN+HAS_KV_LENS | varlen/KV cache 反向 |
| exp_partial_rope_001 | 25% | 跳过 NOPE_DIM+向量化前/后半 | Llama-3, ROPE_DIM≥32 |
| exp_qwen2vl_mrope_001 | 25% | t/h/w 三维融合+head blocking | Qwen2-VL, hd≥128 |
| exp_rope_llama4_npu_001 | 25% | UB tiling+FMA+interleaved | Llama-4 NPU |
| exp_rope_siso_001 | 25% | 3D tiling+非交错向量化 | SISO 布局 |
| exp_triton_rope_001 | 25% | IS_NEOX_STYLE constexpr 双路 | Neox/非Neox |

