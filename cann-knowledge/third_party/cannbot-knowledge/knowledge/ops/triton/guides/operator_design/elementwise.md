---
type: Guide
title: Elementwise 算子优化
description: Triton Ascend 逐元素算子与激活函数的向量化策略、BLOCK_SIZE 选择与融合优化。
tags: [task.design, topic.compute.elementwise.activation, topic.programming_model.execution_unit.vector, topic.compute.composite, topic.tiling, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/sigmoid.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/exp.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Elementwise 算子优化

逐元素算子的输出与输入形状相同，元素间无依赖。天然适合向量化并行，是收益最高的一类场景。优化重点：最大化内存带宽利用率、算子融合消除中间张量读写、ILP 提升计算吞吐。

## 检测标准

- PyTorch 逐元素原语：`torch.exp`/`torch.relu`/`F.gelu`/`F.silu`/`*`/`+` 等，无 `dim=` 归约参数
- 输入输出形状相同（含广播）
- 无跨元素数据依赖

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 大张量向量化 | exp_elem_vec_001 | 30% | BLOCK_SIZE=1024/2048 |
| 非连续内存 | exp_elem_vec_002 | 15% | 先 `.contiguous()` |
| 自适应块大小 | exp_elem_vec_003 | 20% | autotune {256,512,1024,2048,4096} |
| 指令级并行 | exp_elem_vec_005 | 20% | for-loop tiling 多小 tile |
| int64 比较 | exp_elem_vec_006 | 25% | int64→fp32 |
| 激活+量化 | exp_elem_fusion_004 | 30% | 单 kernel 融合 |
| 激活紧跟 matmul | exp_elem_fusion_001 | 35% | 内联进 matmul 内核 |

## 激活函数专项

| 激活 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| SiLU | exp_silu_fwd_001 | 25% | 前向向量化 |
| GELU | exp_gelu_fwd_001 | 25% | 前向向量化 |
| SwiGLU | exp_swiglu_fwd_001 | 30% | 门控激活前向 |
| SwiGLU+量化 | exp_swiglu_quant_001 | 35% | 激活+量化单核 |

## 关键技术

- **BLOCK_SIZE 选择**：2 的幂次，autotune {256,512,1024,2048,4096}，大张量优先 1024/2048
- **向量化**：BLOCK_SIZE 与硬件向量宽度对齐
- **For-loop tiling**：单次 load+compute+store 拆为多个小 tile 循环，提升 ILP

