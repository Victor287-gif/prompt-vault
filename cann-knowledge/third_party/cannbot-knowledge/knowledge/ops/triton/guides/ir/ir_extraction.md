---
type: Guide
title: IR 提取（run_and_extract.sh）
description: 从 Triton-Ascend 编译过程提取 TTIR / TTAdapter / last_pass.mlir 的完整流程：环境变量、bishengir-compile
  参数、dump flag 探测、autotune 去重与验证诊断。
tags: [task.implementation, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---

# 功能说明

`run_and_extract.sh <script>.py` 运行 Triton kernel 脚本触发编译并转储 IR，再经 `bishengir-compile` 提取每个 kernel 最终阶段的 `last_pass.mlir`，供 [IR 分析工作流](ir_analysis_workflow.md)消费。输出目录为 `IR_OUTPUT_DIR`（由调用方 harness 设置；未设置时回落到 `scripts/../ir_output/`），旧文件不清除、同名覆盖。

## 环境变量

| 变量 | 值 / 默认 | 用途 |
|------|-----------|------|
| `TRITON_DEBUG` | `1` | 启用 IR 转储目录 |
| `TRITON_ALWAYS_COMPILE` | `1` | 强制重新编译，忽略缓存 |
| `TRITON_DISABLE_LINE_INFO` | `0` | 保留源代码行信息 |
| `TRITON_DISABLE_FFTS` | `1` | 禁用 FFTS（910_95 不支持） |
| `BISHENGIR_COMPILE` | 必填或 PATH 可寻 | bishengir-compile 绝对路径；缺失即报错退出 |
| `BISHENGIR_PRINT_IR_FLAG` | 可选 | 强制指定 IR dump flag，跳过自动探测 |
| `BISHENGIR_TARGET` | `Ascend910_9382` | 目标芯片；其他 NPU 由 harness 覆盖 |
| `IR_OUTPUT_DIR` | `scripts/../ir_output` | IR 输出目录 |
| `IR_SAVE_ALL` / `IR_SAVE_TTIR` / `IR_SAVE_TTADAPTER` | `true` | 控制中间 IR 是否落盘；`IR_SAVE_ALL=false` 同时关掉前两者 |
| `ASCEND_RT_VISIBLE_DEVICES` | harness 管理 | NPU 设备选择（脚本不硬编码） |

## 提取流程

1. 带上述 `TRITON_*` 环境变量运行 python 脚本，触发编译并转储 IR（stdout  tee 到 `/tmp/triton_compile_run.log`）。
2. 解析 stdout 中 `Dumping intermediate results to <dir>` 行，得到 dump 目录列表。
3. **autotune 去重**：`@triton.autotune` 每个配置变体生成一个 dump 目录；从 `kernel.ttadapter.mlir` 提取 kernel 名（依次尝试 `func.func @` / `tt.func @` / `module @`），按名去重只保留首个变体——每个 kernel 最终只产一套 IR。
4. 对每个唯一 kernel 在其 dump 目录运行 `bishengir-compile`，固定参数：`--target=$BISHENGIR_TARGET --enable-auto-multi-buffer=False --enable-auto-bind-sub-block=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true` + dump flag，全量输出存 `/tmp/kernel_<idx>_full_ir_dump.txt`。
5. 从编译器输出提取最后阶段 IR，复制为 `<IR_OUTPUT_DIR>/<kernel_name>_last_pass.mlir`。

## IR dump flag 探测

优先级：`BISHENGIR_PRINT_IR_FLAG`（env 强制）> `--mlir-print-ir-after-all` > `--bishengir-print-ir-after=hivm-inject-sync` > `--print-after-all`（按 `bishengir-compile --help` 探测）。

- `--mlir-print-ir-after-all` / `--print-after-all`：输出含多个 `IR Dump After` 段，脚本取**最后一个**段（含段名）作为 last_pass。
- `--bishengir-print-ir-after=<pass>`：stdout 直接就是该 pass 后的 IR，整文件采用。
- 三者都不可用时告警并跳过 bishengir 提取，`last_pass.mlir` 不生成。

## 输出文件与验证

| 文件 | 描述 |
|------|------|
| `<kernel>_ttir.mlir` | Triton 前端 IR（可选，受 `IR_SAVE_TTIR` 控制） |
| `<kernel>_ttadapter.mlir` | 适配器 IR（可选，受 `IR_SAVE_TTADAPTER` 控制） |
| `<kernel>_last_pass.mlir` | BishengIR 最终阶段 IR（**必需**，总是生成） |

验证规则：`last_pass.mlir` 必须非空且含 `hivm.hir.` 或 `llvm.` 关键字，否则判定流水线提前停止。诊断决策树：

- 缺 `ttir` → Triton 前端编译失败，检查 kernel 语法。
- 有 `ttir` 缺 `ttadapter` → adapter 转换失败，检查 Triton 方言→memref 适配 pass。
- `last_pass` 无效 → 流水线停在 HIVM→LLVM lowering 之前：多为 `hivmc` 不在 PATH 或编译器内部错误，查 `/tmp/kernel_*_full_ir_dump.txt`。

全部通过时打印快速特征：HIVM intrinsic 行占比、`llvm.load/store/memcpy` 计数，供初步判断。

## 重要约束与陷阱

- `bishengir-compile` 会调 `hivmc-a5` 子进程，**bishengir 的 bin 目录必须在 `$PATH`**；缺失则流程停在 `InjectIR`，不会推进到 LLVM IR。
- 编译阶段段错误通常是 Triton 缓存中不兼容的预编译头：`rm -rf ~/.triton/cache/*` 后再排查。
- 脚本的环境激活（conda / Ascend toolkit / PATH）由调用方 harness 完成，脚本内不写死路径。
