---
type: Guide
title: 端到端性能优化流程
description: Triton-Ascend kernel 端到端优化决策流程：Pattern 检测→瓶颈路由→经验匹配→实施检查清单→重采集验证。
tags: [task.optimization, topic.performance.optimization.optimization_overview, topic.performance.analysis, topic.compute.composite]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 端到端性能优化流程

```
Profile → Identify Bottleneck → Route to Experiences → Implement → Re-profile → Validate
```

## Step 1: Pattern 检测

| Pattern | 指标 | 主要参考 |
|---------|------|----------|
| 矩阵乘 | `tl.dot(a, b)` | ops/matmul |
| Flash Attention | `softmax(QK^T) @ V` | ops/attention |
| 逐元素 | 同 shape 输入输出 | ops/elementwise |
| 归一化 | mean/var/rms | ops/normalization |
| 归约 | `tl.sum/tl.max` 沿轴 | ops/reduction |
| Gather/Scatter | 索引驱动 `tl.gather/scatter` | ops/gather_scatter |

## Step 2: 瓶颈路由

| 瓶颈 | 症状 | 方案 |
|------|------|------|
| 内存带宽 | `aiv_mte2_time` > 2× 理论 | tiling (增大块)、bank_conflict (对齐)、persistent_kernels |
| Scalar fallback | `aiv_scalar_ratio > 10%` | data_types: int64→fp32 |
| L2 bank conflict | M,N≥512 性能退化 | core_mapping: 对角映射 |
| Grid overhead | 大量短命 block | persistent_kernels |
| Pipeline stall | MTE2/Vector 间有间隙 | ilp: care_padding=False |
| 数值不稳定 | NaN/Inf 输出 | online_algorithms, data_types |
| Cube 利用率低 | `aiv_cube_time` 低 | tiling: BLOCK_M/N 升至 128+ |
| 融合机会 | 连续 kernel 有中间 HBM 写 | operator_fusion |

## Step 3: 经验速查

| 任务 | 经验 | 预期提升 | 关键技术 |
|------|------|---------|---------|
| 大矩阵 GEMM (M,N,K≥256) | exp_matmul_001 | +40% | L0 128³ tiling |
| Flash Attention (seq≥1024) | exp_fused_attention_002 | +50% | Online softmax, O(S) |
| RMSNorm 大 hidden dim | 列分块 | +35% | Column blocking, two-pass |
| MoE routing | binned dispatch | +35% | Binned expert dispatch |

## Step 4: 实施检查清单

- [ ] BLOCK_M, BLOCK_N ∈ {128, 256} (matmul)
- [ ] BLOCK_K = 256 (matmul)
- [ ] Tile 不超 192KB UB
- [ ] 行宽 512B 对齐
- [ ] M,N≥512 → diagonal core mapping
- [ ] AutoTune ≥3 GROUP_SIZE (8,16,32)
- [ ] 无 int64 CMP/ADD
- [ ] fp32 accumulation for fp16/bf16
- [ ] `care_padding=False` where safe

## Step 5: 重采集验证

对比优化前后：`total_time`, `aiv_scalar_ratio` (<10%), `aiv_mte2_time` (接近理论), `aiv_vec_ratio` (>60%)。

