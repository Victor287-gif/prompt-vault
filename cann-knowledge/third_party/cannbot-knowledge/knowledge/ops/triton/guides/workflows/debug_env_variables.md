---
type: Guide
title: 调试环境变量完整参考
description: Triton-Ascend 全部调试/编译控制/运行调度环境变量的默认值、取值与作用，含场景组合速查、缓存交互与调试输出层级。
tags: [task.debugging, topic.implementation.debugging, topic.compilation, topic.performance.analysis]
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
# 调试环境变量完整参考

所有变量默认 0 或未设置；`export VAR=1` 启用。调试时务必配合 `TRITON_DISABLE_CACHE=1` 或 `rm -r ~/.triton/cache`，否则缓存命中导致转储不生效。

## 调试与日志类

| 变量 | 取值 | 作用 |
|------|------|------|
| `TRITON_DEBUG` | 0/1 | 转储中间 IR 到 `~/.triton/dump/`（`kernel.ttir.mlir`、`kernel.ttadapter.mlir`） |
| `MLIR_ENABLE_DUMP` | 0/1/kernelName | 每个 MLIR Pass 前后转储 IR；=kernel 名只转储特定 kernel。日志量小、聚焦高层逻辑，日常首选，90% 问题可定位 |
| `LLVM_IR_ENABLE_DUMP` | 0/1 | 每次 LLVM IR 优化前转储 |
| `TRITON_REPRODUCER_PATH` | 路径 | 每个 MLIR 阶段前生成复现文件；阶段失败时保存失败前 MLIR 状态 |
| `TRITON_INTERPRET` | 0/1 | CPU 解释器执行 kernel（NumPy），可插 Python 断点；调完务必 unset |
| `TRITON_ENABLE_LLVM_DEBUG` | 0/1 | LLVM `-debug` 全量日志（单 kernel 数万行，寄存器级细节，仅限 LLVM 专家） |
| `TRITON_LLVM_DEBUG_ONLY` | 逗号分隔通道名 | 限制 LLVM 调试范围。常用：`isel`（指令选择）、`regalloc`（寄存器分配）、`spiller`（寄存器溢出）、`peephole`、`asm-printer` |
| `USE_IR_LOC` | 0/1 | IR 含位置信息（文件名/行号），建立 IR→LLVM IR 映射，配合性能剖析 |
| `TRITON_PRINT_AUTOTUNING` | 0/1 | autotune 完成后输出每个 kernel 最佳配置及总耗时 |
| `MLIR_ENABLE_REMARK` | 0/1 | MLIR 备注/性能警告输出 |
| `TRITON_KERNEL_DUMP` | 0/1 | 各编译阶段 IR 及最终产物保存到目录（配合 `TRITON_DUMP_DIR`） |
| `TRITON_DUMP_DIR` | 路径（默认当前目录） | 内核转储保存目录 |
| `TRITON_DEVICE_PRINT` | 0/1/true | 启用 `tl.device_print`/`tl.static_print`。用 GM 缓冲区，每线程上限 16 KB，超限丢弃；有运行时开销 |
| `TRITON_MEMORY_DISPLAY` | 0/1 | 生成 `memory_info_aic/aiv.json` 内存使用文件到当前目录 |

## 编译控制类

| 变量 | 取值 | 作用 |
|------|------|------|
| `TRITON_ALWAYS_COMPILE` | 0/1 | 忽略缓存强制每次重新编译 |
| `DISABLE_LLVM_OPT` | 0/1/字符串 | 1=禁用全部 LLVM 优化；字符串如 `"disable-lsr"` 禁用特定优化（LSR 在寄存器压力大的 kernel 可致 10% 性能波动） |
| `MLIR_ENABLE_TIMING` | 0/1 | MLIR 编译时间统计 |
| `LLVM_ENABLE_TIMING` | 0/1 | LLVM 编译时间统计 |
| `TRITON_DEFAULT_FP_FUSION` | 0/1（默认 1） | 浮点融合优化（mul+add→fma）开关 |
| `TRITON_KERNEL_OVERRIDE` | 0/1 | 每个编译阶段用外部文件覆盖生成的 kernel 代码（配合 `TRITON_OVERRIDE_DIR`） |
| `TRITON_OVERRIDE_DIR` | 路径（默认当前目录） | 覆盖文件查找目录 |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | 0/1 | 1=某 kernel 编译失败后跳过后续阶段（批量编译省时） |
| `TRITON_COMPILE_ONLY` | 0/1 | remote_launch 时只编译不运行 |
| `TRITON_DISABLE_FFTS` | 0/1 | 1=禁用 FFTS |
| `TRITON_DISABLE_PRECOMPILE` | 0/1 | 1=禁用预编译 |

## 运行与调度类

| 变量 | 取值 | 作用 |
|------|------|------|
| `TRITON_ALL_BLOCKS_PARALLEL` | 0/1 | 逻辑核数自动收敛到物理核数，允许 grid > 65535。**限制**：kernel 逻辑必须对执行顺序不敏感，有原子操作/跨核同步时会死锁 |
| `TRITON_ENABLE_TASKQUEUE` | 0/1 | 开启 task_queue |
| `TRITON_ENABLE_SANITIZER` | 0/1 | 使能 SANITIZER |
| `ENABLE_PRINT_UB_BITS` | 0/1 | 打印当前 UB 占用量（UB 溢出排查首选） |

## 其他

| 变量 | 取值 | 作用 |
|------|------|------|
| `TRITON_BENCH_METHOD` | `npu`/`default` | `npu`：do_bench 切换为 `do_bench_npu`（配合 `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1`）；`default` 强制原 do_bench |
| `TRITON_REMOTE_RUN_CONFIG_PATH` | 路径 | 远程运行配置路径 |

## 场景组合速查

| 场景 | 组合 |
|------|------|
| 编译错误排查 | `TRITON_DEBUG=1` + `MLIR_ENABLE_DUMP=1` + `TRITON_DISABLE_CACHE=1`（或 `TRITON_ALWAYS_COMPILE=1`） |
| 精度调试 | `TRITON_INTERPRET=1` + `TRITON_DISABLE_CACHE=1`（可叠加 `TRITON_DEVICE_PRINT=1`） |
| 运行时打印 | `TRITON_DEVICE_PRINT=1` + `TRITON_DISABLE_CACHE=1` |
| UB 溢出排查 | `ENABLE_PRINT_UB_BITS=1` + `TRITON_DEBUG=1` + `TRITON_DISABLE_CACHE=1`（可叠加 `TRITON_MEMORY_DISPLAY=1`） |
| coreDim 超限 | `TRITON_ALL_BLOCKS_PARALLEL=1` |
| 调试特定 kernel | `MLIR_ENABLE_DUMP=kernel_name` + `TRITON_ALWAYS_COMPILE=1` |
| 深度编译调试 | 日常组合 + `TRITON_ENABLE_LLVM_DEBUG=1` + `TRITON_LLVM_DEBUG_ONLY="isel"`（日志量极大，谨慎） |
| 性能分析 | `MLIR_ENABLE_TIMING=1` + `LLVM_ENABLE_TIMING=1` + `TRITON_PRINT_AUTOTUNING=1`；msprof 采集见 profiling 卡 |
| 编译加速 | `TRITON_ASCEND_COMPILE_SPEED_OPT=1` |
| NPU benchmark | `TRITON_BENCH_METHOD=npu` |

## 优先级与交互

缓存影响：`TRITON_ALWAYS_COMPILE=1` 忽略缓存；`MLIR_ENABLE_DUMP=1` 可能受缓存干扰（不生效先清缓存）；`TRITON_KERNEL_OVERRIDE=1` 覆盖缓存中的 kernel。

调试输出层级（粗→细）：`TRITON_DEBUG=1` → `MLIR_ENABLE_DUMP=1` →（`LLVM_IR_ENABLE_DUMP=1` / `TRITON_ENABLE_LLVM_DEBUG=1` → `TRITON_LLVM_DEBUG_ONLY`）；旁支：`TRITON_DEVICE_PRINT=1`、`TRITON_REPRODUCER_PATH`、`MLIR_ENABLE_REMARK=1`。

## NPU 适配要点

1. `TRITON_ALL_BLOCKS_PARALLEL`、`ENABLE_PRINT_UB_BITS`、`TRITON_ASCEND_COMPILE_SPEED_OPT` 为 NPU 特有。
2. `num_warps`/`num_stages` 在 NPU 上无效，autotune Config 中被忽略。
3. `TRITON_KERNEL_OVERRIDE` 用法：置 1 + 设 `TRITON_OVERRIDE_DIR`，在目录放对应阶段的 IR 文件。

## 常见问题

- `MLIR_ENABLE_DUMP=1` 无输出 → 缓存干扰：清缓存 + `TRITON_ALWAYS_COMPILE=1`。
- LLVM DEBUG 日志太多 → 用 `TRITON_LLVM_DEBUG_ONLY` 限范围；先用 `MLIR_ENABLE_DUMP=1` 定位。
- `TRITON_ALL_BLOCKS_PARALLEL=1` 死锁 → kernel 含原子操作/屏障/顺序依赖时不能开。
- `TRITON_INTERPRET=1` 很慢 → 正常，仅用于精度验证。
- device_print 输出截断 → 每线程 GM 缓冲 16 KB 上限，减少打印量。


## 相关源

- 合并来源：`docs_triton_ascend/09-Reference/04-env-variables.md`（环境变量完整参考，与本卡主源去重合并）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [environment_variable_reference.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/environment_variable_reference.md)
- [debugging.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/debugging.md)
