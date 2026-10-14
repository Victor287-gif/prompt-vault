---
type: Guide
title: Profiling 方法论
description: Triton-Ascend kernel 性能分析方法论：msprof 采集、关键指标解读、理论基线计算与瓶颈判定流程。
tags: [task.optimization, topic.performance.analysis, topic.programming_model.execution_unit.scalar]
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
# Profiling 方法论

## msProf 采集

```bash
msprof op --kernel-name=target_kernel --output=$HOME/output python3 test.py
```

关键输出：`PipeUtilization.csv`（流水线利用率）、`op_summary_*.csv`（kernel 汇总指标）。

## 关键指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `aiv_vec_ratio` | Vector 计算时间占比 | 高（逐元素 >60%） |
| `aiv_scalar_ratio` | Scalar fallback 时间占比 | **<10%** |
| `aiv_mte2_time` | HBM→UB 加载延迟 | 低（接近理论值） |
| `aiv_cube_time` | Cube 单元时间 | 高（matmul kernel） |
| `total_time` | 总 wall-clock 时间 | 最小化 |

## 瓶颈判定流程

1. **Scalar Fallback** (`aiv_scalar_ratio > 10%`) → int64/int32 转 fp32
2. **Memory 瓶颈** (`aiv_mte2_time` 高) → 对角 core mapping / 增大 BLOCK_SIZE / 对齐检查
3. **Vector 利用率低** (`aiv_vec_ratio < 30%`) → 增大 BLOCK_SIZE / 检查数据类型
4. **流水线停顿** → `care_padding=False` / for-loop tiling
5. **Tiling 开销大** (实测 3-5× 理论) → 减少 tile 数 / persistent kernel

## 理论基线

- Memory-bound: `time = (input_bytes + output_bytes) / 1.8TB/s`
- Compute-bound (matmul): `time = 2*M*N*K / peak_TFLOPS`
- Kernel 属 compute-bound 若 `Compute_time > Memory_time`

## Profiling 完整流程

```
运行 kernel → 记录基线 → 计算理论时间 → 计算性能比
→ 若比 <0.5 (超理论 2×): msprof 采集
→ 先查 aiv_scalar_ratio（最便宜的修复）
→ 查 aiv_mte2_time（bank conflict/对齐）
→ 查 aiv_vec_ratio（向量化健康度）
→ 用 simulator trace 做指令级分析
→ 应用修复 → 重采集 → 循环至 ratio > 0.8
```

