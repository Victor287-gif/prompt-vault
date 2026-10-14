---
type: Concept
title: 硬件约束速查表
description: Triton Ascend kernel 生成前必检的强制约束与性能约束清单。涵盖 UB 容量、Grid 上限、512B 对齐、autotune
  搜索空间。
tags: [task.lookup, topic.implementation.memory_management, topic.alignment, topic.tiling, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.td
  role: constraint
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/cv_autotune/hardware_consts.py
  role: constraint
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# 硬件约束速查表

> 生成任何 Triton-Ascend kernel 之前，用此表逐项检查。违反任一项将导致编译错误、运行时崩溃或严重性能退化。

## 强制约束（违反→错误/崩溃）

| # | 约束 | 规则 |
|---|------|------|
| **C1** | UB 容量 | 每个 tile 数据量 ≤ 192KB。fp16: BLOCK ≤ 98304 elements；fp32: ≤ 49152 elements |
| **C2** | Grid 大小 | grid_size ≤ 65535（默认硬件限制） |
| **C3** | 512B 行宽对齐 | matmul 块行宽必须是 512B 整数倍。fp16: BLOCK_N 须为 256 的倍数 |
| **C4** | 禁用 `tl.swizzle2d` | Ascend 上行为未定义，用 `grouped_launch_diagonal` 替代 |
| **C5** | 2D Grid 拓扑匹配 | 2D grid 维度乘积 ≤ AI Core 数，优先使用 1D grid |

## 性能约束（违反→严重退化）

| # | 约束 | 症状 | 解决 |
|---|------|------|------|
| **P1** | int64 CMP/ADD → scalar fallback | `aiv_scalar_ratio` 异常高 | `.to(tl.float32)` 再做比较/加法 |
| **P2** | int32 CMP → scalar fallback | 同上 | `.to(tl.float32)` |
| **P3** | `care_padding=False` 未设 | load/compute 串行 | `tl.load(..., care_padding=False)` |
| **P4** | Block 数远多于核数 | 核利用率低 | `TRITON_ALL_BLOCKS_PARALLEL=1` 或 persistent kernel |
| **P5** | 非连续内存未预处理 | 向量化失效 | kernel 外先 `.contiguous()` |
| **P6** | 大矩阵行主序 tiling | L2 bank conflict (25-35%) | 对角 tiling |
| **P7** | fp16/bf16 累加未升精度 | 溢出/精度问题 | fp32 accumulator |

## Tile Size 速查

| 数据类型 | 最大 BLOCK（单 tile） | 推荐 matmul BLOCK_M/N | 推荐 BLOCK_K |
|----------|----------------------|----------------------|-------------|
| fp16/bf16 (2B) | 98304 ≈ 313×313 | 128 或 256 | 256 |
| fp32 (4B) | 49152 ≈ 221×221 | 128 | 128 或 256 |
| fp8 (1B) | 196608 | 256 | 256 |

> matmul 的 L0 缓冲区同时持有 A/B/C tile，实际可用 UB 约 1/3。

## Autotune 搜索空间

```python
# matmul 标准搜索空间
configs = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': 256, 'GROUP_SIZE': gs})
    for bm in [128, 256]
    for bn in [128, 256]
    for gs in [8, 16, 32]
]

# elementwise 标准搜索空间
configs = [
    triton.Config({'BLOCK_SIZE': bs})
    for bs in [256, 512, 1024, 2048, 4096]
]
```

## Kernel 前检查清单

- [ ] BLOCK_SIZE × dtype_bytes ≤ 192KB / 3（为 A/B/C 留余量）
- [ ] matmul 的 BLOCK_N × dtype_bytes 为 512 的整数倍
- [ ] grid_size ≤ num_ai_cores（910b2=20/24 核）
- [ ] 未使用 `tl.swizzle2d`
- [ ] 所有 `tl.arange()` 结果做比较/加法前 `.to(tl.float32)`
- [ ] `tl.load()` 中 `care_padding=False`（安全时）
- [ ] fp16/bf16 归约/累加在 fp32 中进行
- [ ] 大矩阵 M,N≥512 时启用对角 tiling
- [ ] BLOCK_M, BLOCK_N, BLOCK_K 均为 2 的幂次

