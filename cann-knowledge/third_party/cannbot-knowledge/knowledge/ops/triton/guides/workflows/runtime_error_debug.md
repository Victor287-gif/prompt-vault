---
type: Guide
title: 运行时错误排查
description: NPU 运行时错误的排查方法：UB 溢出、coreDim 超限、精度异常、Scalar 退化、离散访存的诊断命令与修复代码，以及全零/NaN、超时、部分元素错误的排查路径。
tags: [task.debugging, topic.implementation.debugging, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# 运行时错误排查

运行时错误发生在 kernel 编译成功后的 NPU 执行阶段。

## 错误 1：UB 溢出（部分场景运行时才暴露）

报错：`ub overflow, requires 3072256 bits while 1572864 bits available!`（提示语：multi-buffer 使能或某些 op 需要额外 local buffer）。

排查：`ENABLE_PRINT_UB_BITS=1` 查占用 → 查 BLOCK_SIZE → 查 multibuffer → 查 int8（约占 2 倍片上空间）。

UB 容量与估算：

| 硬件 | 单 buffer | double buffer |
|------|----------|---------------|
| A2/A3 | 192 KB | 总和 ≤ 96 KB |
| 910_95 | 256 KB | 总和 ≤ 128 KB |

修复（三选一或组合）：减小 BLOCK_SIZE；核内 Tiling（`BLOCK_SIZE_SUB` for 循环）；`triton.Config({'XS': 128, 'multibuffer': False})`。

## 错误 2：coreDim 超限

报错：`coreDim=524288 can't be greater than UINT16_MAX`。`coreDim = ceil(N / BLOCK_SIZE)`，上限 65535。

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1   # 要求 kernel 对执行顺序不敏感
```
```python
min_block = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block)
```

## 错误 3：精度异常

排查：`TRITON_INTERPRET=1` 取 CPU 基准 → `TRITON_DEVICE_PRINT=1` + `tl.device_print` 打印 NPU 中间结果 → 逐步对比定位差异点。

| 原因 | 表现 | 修复 |
|------|------|------|
| 浮点计算顺序差异 | 微小差异（1e-6） | 正常，`torch.allclose` 验证 |
| 数据类型退化 | 较大差异 | 查 int64/int32 scalar 退化 |
| bf16/fp16 精度损失 | 中等差异 | 关键计算升 fp32 |
| mask 处理差异 | 特定位置差异 | 查 boundary_check/mask |

## 错误 4：Scalar 退化（性能）

表现：`aiv_vec_ratio < 10%`，Scalar 成最长流水。int64 Vector ADD、int64/int32 Vector CMP 会退化为 Scalar。修复：比较操作数转 fp32：

```python
cols = tl.arange(0, BLOCK_N)              # int64
xbar = tl.where(cols < N, x - mean, 0.0)          # 退化
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)      # Vector CMP
```

## 错误 5：离散访存（性能）

表现：MTE2 时间远超理论值（`数据量 / 1.8TB/s`）。修复：保证最低维连续；`tl.make_block_ptr` 的 stride 反映真实内存布局；避免非对齐访存。

## 其他运行时症状

| 症状 | 可能原因 | 排查/修复 |
|------|---------|----------|
| NPU 执行超时 | grid 过多分批调度 / 死锁 / 数据规模过大 | grid 对齐物理核数；查跨核依赖；缩小数据测试 |
| 结果全零或全 NaN | mask 错误未加载数据 / 类型转换错 / 除零 | `tl.device_print` 打印加载后数据；解释器验证 |
| 部分元素错误 | 边界处理 / mask / 对齐 | 查 `offsets < N`、`boundary_check`，解释器逐步验证 |
| 性能远低于预期 | scalar 退化 / 离散访存 / grid 过多 / tiling 不合理 | msProf 上板 + 仿真流水图 + 代码热点 |

## 运行时日志分析

```bash
# 上板 Profiling：op_summary_*.csv（各流水利用率）、PipeUtilization.csv
msprof op --kernel-name=target_kernel --output=$HOME/output python3 your_program.py
# 仿真流水图：trace.json（chrome://tracing 或 MindStudio Insight）、visualize_data.bin
export LD_LIBRARY_PATH=<CANN>/ascend_toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=target_kernel --soc-version=Ascend910B3 python3 your_program.py
```

流水图看点：SCALAR/FLOWCTRL 饱和 → 标量优化；规律性断流 → 基本块过小；空白段 → 流水未利用。

msProf 采集失败检查项：CANN 环境变量、`--soc-version` 与硬件匹配（`npu-smi info`）、`--kernel-name` 用 kernel 函数名而非文件名、输出目录写权限、无其他 msProf 进程。

## 要点

1. UB 溢出是最常见运行时问题，严格控制单核数据量。
2. msProf（上板 + 仿真）是运行时调试核心。
3. Scalar 退化是性能问题常见原因，优先查数据类型。
4. 解释器对而 NPU 错不一定是硬件 bug：浮点精度、mask、类型转换、编译器优化差异皆可能。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [profiling.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/profiling.md)
- [performance_guidelines.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/performance_guidelines.md)
