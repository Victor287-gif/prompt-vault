---
type: Optimization
title: 高级优化技巧合集（Profiling / 编译期优化 / 检查清单）
description: kernel 跑通后的系统性优化：torch_npu.profiler 与 msprof 采集、PipeUtilization 瓶颈判读、仿真流水图诊断、static_range/constexpr
  编译期优化与六阶段检查清单。
tags: [task.optimization, topic.performance.analysis, topic.performance.optimization.optimization_overview, topic.implementation.pipeline, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# 高级优化技巧合集

触发条件：kernel 已在 NPU 上正确运行但性能未达预期，需系统性优化。

## Profiling 方法

### torch_npu.profiler

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False,
    data_simplification=False,
)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=3, active=30, repeat=1, skip_first=10),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(result_path),
    record_shapes=True,
    experimental_config=experimental_config,
) as prof:
    for i in range(10 + (0 + 3 + 30) * 1):
        fn(*args)
        prof.step()
```

### msprof 命令行

```bash
msprof op --kernel-name=target_kernel_name --output=$HOME/projects/output python3 test_op.py

export LD_LIBRARY_PATH=/root/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/{soc-version}/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=_layer_norm_fwd_fused --soc-version=Ascend910B3 python3 test_op.py
```

### AiCMetrics.PipeUtilization 关键指标

| 指标 | 含义 | 理想值 | 瓶颈信号 |
|------|------|--------|----------|
| aiv_vec_ratio | Vector 流水利用率 | > 80% | < 30% Vector 断流 |
| aiv_mte2_ratio | MTE2 搬入占比 | < 50% | > 50% 搬运瓶颈 |
| aiv_scalar_ratio | Scalar 流水利用率 | < 20% | > 30% 标量退化 |
| aic_cube_ratio | Cube 流水利用率 | > 80% | < 30% Cube 断流 |

### 瓶颈分类 → 优化方向

| 瓶颈类型 | 特征 | 优化方向 |
|----------|------|----------|
| 计算密集 | vec_ratio / cube_ratio 高 | 增大分块、CV 融合 |
| 访存密集 | mte2_ratio 高 | 存算并行、care_padding、对齐优化 |
| 标量退化 | scalar_ratio 高 | 数据类型优化（避免 int64 ADD/CMP） |
| 同步开销 | 流水图大量等待 | 减少同步、care_padding=False |

### 仿真流水图诊断

`trace.json` 用 Chrome `chrome://tracing` 或 MindStudio Insight 打开；`visualize_data.bin` 用 MindStudio Insight（含源码映射）。诊断规则：

1. SCALAR/FLOWCTRL 指令饱和 → 标量计算过多
2. MTE2/VECTOR 规律性断流 → 基本块切分过小
3. 大量空白段 → 流水线未充分利用

### 辅助环境变量

| 环境变量 | 说明 |
|----------|------|
| `TRITON_PRINT_AUTOTUNING=1` | 打印 autotune 过程与最优配置 |
| `TRITON_ALL_BLOCKS_PARALLEL=1` | 按物理核数优化逻辑核数（有同步操作时禁用） |
| `TRITON_DEBUG=1` | 保存中间编译产物 |
| `TRITON_BENCH_METHOD=npu` | autotune 用 NPU 专用计时 |
| `TRITON_AUTOTUNE_PARALLEL_COMPILE=1` | 并行编译加速调优（默认开启） |

## 编译期优化

`tl.static_range`：编译期完全展开循环，消除循环控制开销。循环次数须已知且较小（通常 < 10），`start`/`end`/`step` 均为 `tl.constexpr`；次数过大会代码膨胀：

```python
for i in tl.static_range(BLOCK_SIZE):
    x = tl.load(x_ptr + i)
    tl.store(y_ptr + i, x * x)
```

`tl.constexpr`：固定参数（BLOCK_SIZE、stride、shape 维度）声明为 constexpr，启用常量折叠/传播，减少运行时分支。

## 性能优化检查清单

**阶段一 基础适配**：grid 对齐物理核数（Vector 算子 = `vectorcore_num`，CV 算子 = `aicore_num`）；核内 `range(pid, NUM_BLOCKS, NUM_CORE)` 循环；避免 int64 ADD/CMP（改 int32/fp32）；`device='npu'`。

**阶段二 访存优化**：尾轴对齐（Vector 算子 32B，CV 算子 512B）；无依赖时 `care_padding=False`；offsets 连续避免离散访问；先整体搬 UB 再 select。

**阶段三 Tiling**：autotune 搜索分块；大 BLOCK_SIZE + 小 BLOCK_SIZE_SUB 二级分块防 UB 溢出；for 循环 Tiling；单次分块不超 UB 容量。

**阶段四 存算并行**：MultiBuffer 双缓冲（910_95 需显式 `multibuffer=True`）；for 循环为前提；开启后 UB 可用减半。

**阶段五 高级**：Memory-Bound 算子融合为 Compute-Bound；矩阵乘+后处理用 CV 融合；`tl.constexpr`/`tl.static_range`；归约升 FP32、矩阵乘混合精度保精度。

**阶段六 910_95 专项**：显式开 multibuffer；利用 fixpipe L0C→UB 直通；UB 248KB 允许更大分块；Vector 核数翻倍，grid 可用更多核。