---
type: Optimization
title: NPU 性能优化策略总览
description: Triton-Ascend 性能优化全局视图：计算/访存密集型判定、优化策略矩阵、关键性能指标（vec/cube/scalar ratio）与分阶段检查清单。
tags: [task.optimization, topic.performance.optimization.optimization_overview, topic.performance.analysis, topic.implementation.pipeline, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# NPU 性能优化策略总览

## 硬件资源速览

| 组件 | 说明 | 典型值 |
|------|------|--------|
| AI Core | 1 Cube 核 + 2 Vector 核 | 型号决定数量（20~36） |
| UB | Vector 的片上工作区 | A2: 192KB；910_95/950: 256KB |
| L1 Buffer | Cube 的矩阵数据缓存 | 型号相关 |
| L0C | Cube 累加器输出缓冲 | 固定大小 |
| MTE2 / MTE3 | GM→UB / UB→GM 搬运流水 | 带宽约 1.8TB/s，共享 |

## 算子类型判定与优化策略矩阵

| 算子类型 | 特征 | 典型算子 | 瓶颈 | 优化方向 |
|----------|------|----------|------|----------|
| 计算密集型 | 计算量/访存量比高 | matmul、Conv | Cube/Vector 单元 | 增大分块、CV 融合 |
| 访存密集型 | 比值低 | add、copy、mask | MTE2/MTE3 | 存算并行、care_padding、对齐 |
| 混合型 | 两者兼有 | LayerNorm、Softmax、FA | 视比例 | 综合优化、流水线调度 |

| 优化手段 | 计算密集 | 访存密集 | 混合型 |
|----------|:---:|:---:|:---:|
| 增大 Tiling 分块 | 高 | 中 | 高 |
| 存算并行 (multiBuffer) | 高 | 高 | 高 |
| care_padding=False | 低 | 高 | 中 |
| CV 融合 | 高 | 不适用 | 高 |
| 对齐优化 | 中 | 高 | 高 |
| 数据类型优化 | 中 | 低 | 中 |
| Grid 分核优化 | 中 | 中 | 高 |

## 关键性能指标（msprof op_summary）

| 指标 | 含义 | 理想值 |
|------|------|--------|
| aiv_vec_ratio | Vector 流水利用率 | > 80% |
| aiv_cube_ratio | Cube 流水利用率 | > 80% |
| aiv_scalar_ratio | Scalar 流水利用率 | < 20% |

诊断规则：
- `aiv_scalar_ratio > 30%`：标量退化，检查数据类型是否被硬件 Vector 单元支持
- `aiv_vec_ratio` 过低：Vector 流水断流，检查 Tiling 与存算并行
- `aiv_mte2_ratio` 高而 vec 低：访存密集型

## 分阶段优化检查清单

**阶段一：基础适配**
- [ ] Grid 分核数对齐物理核数（Vector 算子 = vectorcore_num；CV 算子 = aicore_num）
- [ ] 核内 `range(pid, NUM_BLOCKS, NUM_CORE)` 循环
- [ ] 避免 int64 的 ADD/CMP，改 int32 或 fp32
- [ ] `device='npu'` 替代 `device='cuda'`

**阶段二：访存优化**
- [ ] 尾轴对齐：Vector 32B，CV 512B
- [ ] padding 无依赖时 `care_padding=False`
- [ ] offsets 连续，避免离散访问
- [ ] 短轴不对齐用借轴转置

**阶段三：Tiling 优化**
- [ ] autotune 搜索最优 BLOCK_SIZE
- [ ] 二级分块 BLOCK_SIZE + BLOCK_SIZE_SUB 避免 UB 溢出
- [ ] for 循环 Tiling 实现存算并行
- [ ] UB 空间检查

**阶段四：高级优化**
- [ ] CV 融合（矩阵乘 + 后处理）
- [ ] multiBuffer 双缓冲
- [ ] L1 缓存 A/B 分块复用
- [ ] CV 融合中 sync_block 同步

## GPU → NPU 优化思路差异

| 维度 | GPU | NPU |
|------|-----|-----|
| 分核 | block 数可远大于 SM 数 | 固定物理核数 + 核内循环 |
| 片上内存 | Shared Memory 灵活 | UB/L1 固定大小 + 严格对齐 |
| 计算单元 | CUDA/Tensor Core 自动调度 | Cube/Vector 需显式 CV 融合 |
| 存算并行 | 手动双缓冲 | 编译器默认 multiBuffer，需 for 循环配合 |
| 对齐 | 128B（warp 访存） | Vector 32B / Cube 512B |
| 标量退化 | 少见 | int64 ADD/CMP 退化，需特别注意 |

## 迭代优化工作流

```
1. 基础适配（device='npu'、固定 Grid、核内循环）
2. 正确性验证（coreDim / UB 溢出处理）
3. 性能基线（msprof op 采集 op_summary）
4. 瓶颈分析（vec/mte2/scalar ratio 对比理论值）
5. 实施优化（按瓶颈类型选策略）
6. 验证迭代（重采集对比，未达标回步骤 4）
```


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
- [performance_guidelines.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/performance_guidelines.md)
- [migrate_from_gpu.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/migrate_from_gpu.md)
