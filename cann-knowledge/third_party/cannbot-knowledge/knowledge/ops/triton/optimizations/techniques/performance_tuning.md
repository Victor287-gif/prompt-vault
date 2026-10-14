---
type: Optimization
title: 性能调优方法论
description: Triton Ascend 端到端性能调优：autotune 搜索空间设计、num_warps/num_stages 调参与理论基线对比方法。
tags: [task.optimization, topic.performance.optimization.optimization_overview, topic.tiling, topic.performance.analysis]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 性能调优方法论

端到端性能调优方法论：从理论基线出发，通过 autotune 搜索和 profiling 迭代逼近硬件理论上限。

## OPT-triton-perf-001: Autotune 搜索空间

matmul 标准搜索空间：
- BLOCK_M ∈ {128, 256}
- BLOCK_N ∈ {128, 256}
- BLOCK_K = 256（固定）
- GROUP_SIZE ∈ {8, 16, 32}（至少 3 档）
- num_stages=2, num_warps=4

elementwise 标准搜索空间：
- BLOCK_SIZE ∈ {256, 512, 1024, 2048, 4096}

## OPT-triton-perf-002: 理论基线

- Memory-bound: `time = (input_bytes + output_bytes) / 1.8TB/s`
- Compute-bound (matmul): `time = 2*M*N*K / peak_TFLOPS`
- 目标：实测 ≤ 理论 × 1.2

## OPT-triton-perf-003: Profiling 驱动迭代

```
运行 → 算理论 → aiv_scalar_ratio>10%? → 数据类型修复
→ aiv_mte2 高? → core_mapping/block_size
→ aiv_vec_ratio 低? → 增大 BLOCK_SIZE
→ 理论比 <0.8? → 继续迭代
```

