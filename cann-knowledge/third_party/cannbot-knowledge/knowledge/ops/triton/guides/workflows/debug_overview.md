---
type: Guide
title: Triton-Ascend 调试方法总览
description: 调试工具链（解释器模式/IR 转储/device_print/msProf）、编译流程各阶段定位法、精度/编译/性能三类问题的调试流程与场景速查表。
tags: [task.debugging, topic.implementation.debugging, topic.compilation, topic.precision, topic.performance.analysis]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# Triton-Ascend 调试方法总览

调试原则"由简到繁、由上到下"：先验证逻辑正确性（解释器模式 CPU 执行）→ 再验证编译正确性（IR 转换）→ 最后验证运行正确性（NPU 执行）。

## 调试工具链

| 工具 | 用途 | 启用方式 |
|------|------|---------|
| 解释器模式 | CPU 上执行 kernel，验证逻辑 | `TRITON_INTERPRET=1` |
| 编译器日志 | IR 文件转储到 `~/.triton/dump/` | `TRITON_DEBUG=1`（配合 `TRITON_DISABLE_CACHE=1`） |
| MLIR IR Dump | 每个 Pass 前后 IR 输出到 stderr | `MLIR_ENABLE_DUMP=1`（或 `=kernel_name` 只转储特定 kernel） |
| 运行时打印 | 打印 kernel 内张量值 | `TRITON_DEVICE_PRINT=1` + `tl.device_print` |
| 静态打印 | 打印编译时常量 | `tl.static_print` |
| msProf | 上板性能采集 / 仿真流水图 | `msprof op [simulator] ...` |

## 编译流程与失败点定位

```text
[Python Kernel] --triton.compile--> [ttir.mlir] --Ascend 后端--> [ttadapter.mlir] --bishengir--> [.o]
```

检查 `~/.triton/dump/`：无 `ttir.mlir` → Python→TTIR 失败；有 ttir 无 ttadapter → **TTIR→TTAdapter 转换失败（大部分编译问题发生在此）**；有 ttadapter → 后续二进制编译失败，用 `MLIR_ENABLE_DUMP=1` 定位失败 Pass。

## 三类问题调试流程

**精度问题**：`TRITON_INTERPRET=1` CPU 运行 → 与 PyTorch 参考对比。一致 → 问题在 NPU 执行，用 `tl.device_print` 打印中间结果逐步对比；不一致 → kernel 逻辑错误，插 `breakpoint()` 调试。

**编译错误**：记录完整调用栈 → `TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1` → 按 dump 文件定位阶段 → `MLIR_ENABLE_DUMP=1` 看 Pass 详情。

**性能问题**：msProf 采集 → 分析 PipeUtilization → Vector 利用率低查 scalar 退化；MTE2 过长查 Tiling/访存连续性；Scalar 过长用仿真流水图分析。

## 场景速查表

| 场景 | 症状 | 首选方法 | 辅助方法 |
|------|------|---------|---------|
| 精度差异 | NPU 结果 ≠ 参考 | 解释器模式 | `tl.device_print` |
| 编译错误 | `MLIRCompileError` | `TRITON_DEBUG=1` + IR 分析 | `MLIR_ENABLE_DUMP=1` |
| UB 溢出 | `ub overflow` 编译错误 | 减小 BLOCK_SIZE / 增加 Tiling | `ENABLE_PRINT_UB_BITS=1` |
| coreDim 超限 | `coreDim > UINT16_MAX` | 增大 BLOCK_SIZE / `TRITON_ALL_BLOCKS_PARALLEL=1` | - |
| 性能低下 | 执行时间过长 | msProf 上板 Profiling | 仿真流水图 |
| Scalar 退化 | Vector 利用率低 | 检查数据类型 | 流水图 + 代码热点 |
| 离散访存 | MTE2 时间过长 | 检查 stride/order | IR 分析 |
| 运行时崩溃 | kernel 中断 | 检查数据类型和边界 | 解释器模式验证 |

## 环境变量组合

```bash
# 精度调试
export TRITON_INTERPRET=1 TRITON_DISABLE_CACHE=1
# 编译调试（日常）
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 MLIR_ENABLE_DUMP=1
# 运行时打印
export TRITON_DEVICE_PRINT=1 TRITON_DISABLE_CACHE=1
# 深度编译调试（日志量极大，谨慎）
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 MLIR_ENABLE_DUMP=1 TRITON_ENABLE_LLVM_DEBUG=1 LLVM_DEBUG_ONLY="isel"
```

## NPU 适配要点

1. 调试重点在 TTIR→TTAdapter 转换阶段；
2. 解释器模式是精度调试首选（CPU 用 NumPy 执行，浮点顺序与 NPU 不同，微小差异正常，但逻辑错误会暴露）；
3. `ttir.mlir`/`ttadapter.mlir` 是编译调试的关键产物；
4. msProf（上板 + 仿真）是性能调试核心；
5. 调试时务必 `TRITON_DISABLE_CACHE=1` 避免缓存干扰。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [debugging.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/debugging.md)
- [profiling.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/profiling.md)
