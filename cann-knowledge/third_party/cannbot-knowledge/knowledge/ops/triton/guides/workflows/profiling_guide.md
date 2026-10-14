---
type: Guide
title: Triton-Ascend 性能分析与瓶颈定位指南
description: msProf 上板 Profiling 与仿真流水图的采集命令、op_summary 关键指标判读、理论耗时计算、瓶颈定位四法与六大性能反模式。
tags: [task.optimization, topic.performance.analysis, topic.programming_model.execution_unit.scalar]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '310p']
---
# Triton-Ascend 性能分析与瓶颈定位指南

性能优化是迭代过程：采集数据 → 分析瓶颈 → 实施优化 → 验证效果。

## msProf 采集命令

```bash
# 上板 Profiling（采集指定 kernel；不指定 --kernel-name 则采集所有算子）
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 test_op.py

# 仿真流水图（指令级性能数据）
export LD_LIBRARY_PATH=<CANN>/ascend_toolkit/latest/tools/simulator/{soc-version}/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=target_kernel_name --soc-version=Ascend910B3 python3 test_op.py
```

输出文件：`op_summary_*.csv`（流水利用率）、`PROF_*`（原始数据）；仿真侧 `trace.json`（拖入 Chrome `chrome://tracing` 或 MindStudio Insight 查看指令时间线）、`visualize_data.bin`（MindStudio Insight 指令-源码关联看板）。

支持的 soc-version：Ascend910A/B/B1/B2/B2C/B3/B4、Ascend310/310B1-B4/310P1-P7。用 `npu-smi info` 确认实际硬件。

## op_summary 关键指标

| 指标 | 含义 | 瓶颈信号 |
|------|------|----------|
| aiv_vec_ratio | Vector 流水利用率 | < 30% 说明 Vector 断流（理想 > 80%） |
| aiv_mte2_ratio | MTE2 搬入占比 | > 50% 说明搬运瓶颈 |
| aiv_scalar_ratio | Scalar 流水占比 | > 30% 说明标量退化（理想 < 20%） |
| aic_cube_ratio | Cube 流水利用率 | < 30% 说明 Cube 断流（理想 > 80%） |
| aiv_vec_time / aiv_mte2_time / aiv_mte3_time / aiv_scalar_time / aic_cube_time | 各流水耗时(us) | 耗时最长者即主瓶颈 |

## 理论耗时计算

```python
# 搬运理论耗时 = 搬运数据量(Byte) / 理论带宽（GM 峰值约 1.8 TB/s）
latency = 4 * 4096 * 4096 / 1.8e12   # fp32 4096x4096 ≈ 37.28 us
# 计算理论耗时 = 计算量(Element) / 理论算力（Vector 峰值 11.06 TOPS）
```
实际性能达理论值 70% 以上通常视为接近最优。MTE2+MTE3 同时搬运共享带宽。

## 瓶颈定位四法

1. **流水利用率分析**（op_summary）：`aiv_vec_ratio < 10%` → Vector 未发挥；`aiv_scalar_ratio > 30%` → 标量退化；`aiv_mte2_ratio > 50%` → 搬运瓶颈；`aic_cube_ratio < 30%` → Cube 断流。多个指标都低时，取耗时最长的流水为主瓶颈。
2. **Tiling 分析**：Block Dim > 物理核数 → Host 调度开销大；远小于 → 核利用率不足；推荐 Block Dim = 物理核数。
3. **仿真流水图**：SCALAR/FLOWCTRL 指令饱和 → 标量计算过多；MTE2/VECTOR 规律性断流 → 基本块切分过小；大量空白段 → 流水线未利用。
4. **代码热点分析**（visualize_data.bin）：load 接口对应大量标量指令 → 离散访存或类型退化；某行指令拍数异常高 → 优化该行。

## 瓶颈分类与优化方向

| 瓶颈类型 | 特征 | 优化方向 |
|----------|------|----------|
| 计算密集 | vec/cube ratio 高 | 增大分块、CV 融合 |
| 访存密集 | mte2 ratio 高 | 存算并行、care_padding、对齐优化 |
| 标量退化 | scalar ratio 高 | 数据类型优化（避免 int64 ADD/CMP） |
| 同步开销 | 流水图大量等待 | 减少同步、care_padding=False |
| Tiling 不合理 | 断流、冗余搬运 | 调整 BLOCK_SIZE、autotune |

## 六大性能反模式

1. **int64/i32 Compare 退化为标量**：`cols < N`（int64）无法走 Vector。修复：`cols.to(tl.float32)` 后再比较。
2. **Grid 分核数过多**：GPU 移植 kernel 的 grid 远超物理核数。修复：`grid = (NUM_CORE,)` 固定物理核数 + 核内 for 循环。
3. **Tiling 过小搬运冗余**：BLOCK_SIZE 过小 → 冗余搬运指令、MTE2 断流。修复：增大 BLOCK_SIZE 或 autotune。
4. **无 for 循环无法存算并行**：单次执行完成，multiBuffer 无法使能。修复：BLOCK_SIZE 内再切 BLOCK_SIZE_SUB 做核内 for 循环 Tiling。
5. **尾轴不对齐自动补齐**：尾轴不满 32B/512B 对齐，硬件补齐浪费带宽（如 shape=(2048,3) bf16 尾轴仅 6B）。修复：1D load + reshape/trans 借轴。
6. **care_padding 导致 MTE2-Vector 串行**：care_padding=True（默认）时 MTE2 等 Vector 初始化。确认 padding 区不影响结果后设 `care_padding=False`。

## 辅助环境变量

`TRITON_PRINT_AUTOTUNING=1`（打印调优结果）、`TRITON_ALL_BLOCKS_PARALLEL=1`（grid 超物理核数时自动收敛）、`TRITON_DEBUG=1`（保存 `~/.triton/cache/*.ttadapter`，可用 `bishengir-compile --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true` 离线分析 IR 中是否有纯 scalar 搬运/计算）、`TRITON_BENCH_METHOD=npu`。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [profiling.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/profiling.md)
- [performance_guidelines.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/performance_guidelines.md)
- [migrate_from_gpu.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/migrate_from_gpu.md)
