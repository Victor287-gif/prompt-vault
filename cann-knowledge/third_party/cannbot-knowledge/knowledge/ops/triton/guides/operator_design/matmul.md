---
type: Guide
title: MatMul / GEMM 高性能算子
description: Triton Ascend 矩阵乘法的经验路由、适用条件、Tiling 策略、Core Mapping 选择与融合变体总览。
tags: [task.design, topic.compute.composite, topic.programming_model.execution_unit.cube, topic.tiling, topic.data_movement.memory_access, topic.memory]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---
# MatMul / GEMM 高性能算子

矩阵乘法是深度学习最核心的计算原语。在 910B 上，计算主要由 Cube 单元承担，Vector 单元负责后处理。通过合理的 Tiling、Core Mapping 与内存层次利用，相比 PyTorch 基线可获得 10-40% 性能提升。

## 检测标准

- 计算图含 `torch.matmul`/`torch.mm`/`torch.bmm`/`F.linear`
- 输出 `[..., M, N]`，输入 `[..., M, K]` 和 `[..., K, N]`
- 含可选转置标志 `trans_a`/`trans_b`

## 经验路由

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 大矩阵标准 GEMM | exp_matmul_001 | 40% | L0 128³ Tiling + L0C 寄存器复用 |
| 量化融合 | exp_matmul_002 | 35% | Cube fixpipe 量化 + 4× K 展开 |
| 双转置 | exp_matmul_005 | 32% | M=256,K=256,N=128 专项配置 |
| 对角 Tiling (M,N≥512) | exp_matmul_diag_tiling_001 | 25% | 对角核映射消除 bank conflict |
| 小矩阵 (<128) | micro_tiling | 20% | 16×16×16 micro-tiling |
| Persistent Kernel | exp_matmul_sched_003 | 25-35% | grid-stride 固定核数 |
| Cube/Vector 异构流水 | exp_matmul_sched_004 | 30% | `dl.async_task` 并行 |
| Grouped GEMM (MoE) | exp_matmul_sched_006 | 25% | work-stealing 均衡负载 |

## 融合变体

| 融合模式 | 提升 |
|----------|------|
| GEMM + Activation (GELU/SiLU) | 30% |
| GEMM + Bias | 15% |
| GEMM + RoPE | 25% |
| GEMM + RMSNorm | 30% |

## 关键技术

- **L0 Tiling**：标准配置 BLOCK_M=128, BLOCK_N=128, BLOCK_K=128；大矩阵扩展至 256
- **对角线遍历**：M,N≥512 时使用 diagonal PID mapping 消除 bank conflict
- **NZ Layout + K 展开**：列主序分块减少 Cube 搬运开销，4× K 展开隐藏延迟
- **PID 分组**：GROUP_SIZE∈{8,16,32}，控制 L2 局部性

## 常见陷阱

- BLOCK_M/BLOCK_N 非 128 倍数 → Cube 利用率骤降
- 双转置未走专项路径 → 性能损失显著
- NZ/ND layout 混用 → 数据错位
- 910b2 `tl.dot` 输入格式不匹配 → 静默数值错误
- Persistent Kernel 核数与硬件不匹配 → 浪费或竞争

