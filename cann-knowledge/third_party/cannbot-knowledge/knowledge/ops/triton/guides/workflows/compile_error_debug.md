---
type: Guide
title: 编译错误排查（MLIRCompileError）
description: 编译错误分阶段定位法（dump 文件判定）、TTIR/TTAdapter IR 识读要点、十大常见编译错误的报错-原因-修复三元组与 bishengir-compile
  手动编译命令。
tags: [task.debugging, topic.implementation.debugging, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# 编译错误排查（MLIRCompileError）

最常见编译错误是 `MLIRCompileError`，多发生在 TTIR → TTAdapter 转换阶段。

## 分阶段定位法

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1   # 必须，否则缓存命中不生成 dump
python your_program.py
```

检查 `~/.triton/dump/`：

| 文件状态 | 失败阶段 | 下一步 |
|---------|---------|--------|
| 无 `kernel.ttir.mlir` | Python Kernel → TTIR | 检查 kernel 语法 |
| 有 ttir 无 ttadapter | TTIR → TTAdapter | 检查操作是否受 NPU 支持 |
| 有 ttadapter | TTAdapter → 二进制 | `MLIR_ENABLE_DUMP=1` 定位失败 Pass |

查看 Pass 详情：`MLIR_ENABLE_DUMP=1`（或 `=kernel_name` 只转储特定 kernel），重定向 `2>mlir_dump.log`。缓存干扰时 `rm -rf ~/.triton/cache/`。

手动编译 TTAdapter IR 复现：

```bash
bishengir-compile kernel.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

## IR 识读要点

- **TTIR**（平台无关）：`tt.func` kernel 定义、`tt.get_program_id`、`tt.make_range`、`tt.load`/`tt.store`、`arith.*`。
- **TTAdapter IR**（面向 Ascend）：`memref.alloc`（UB buffer，过大即 UB 溢出）、`memref.copy`（MTE2/MTE3 搬运）、`linalg.fill`、`scf.if`、`bufferization.*`。检查是否存在纯 scalar 搬运/计算。

## 十大常见编译错误

| # | 报错信息 | 原因 | 修复 |
|---|---------|------|------|
| 1 | `ub overflow, requires 3072256 bits while 1572864 bits available!` | 单核数据量超 UB（A2/A3 192KB=1572864 bits；910_95 256KB） | 减小 BLOCK_SIZE / 核内 Tiling；关 multibuffer；int8 占 2 倍空间；double buffer 时总和 ≤96KB |
| 2 | `coreDim=524288 can't be greater than UINT16_MAX` | grid > 65535 | `TRITON_ALL_BLOCKS_PARALLEL=1`；或 `BLOCK_SIZE >= next_power_of_2(cdiv(N, 65535))` |
| 3 | `unsupported type: uint64` | NPU 不支持 uint8/16/32/64、fp64 | uint8→int8、uint16→int16、fp64→fp32 |
| 4 | `failed to legalize operation 'tt.advance'` | `tl.advance` + 复杂循环/分支 | 循环内重建 `tl.make_block_ptr` 替代 advance |
| 5 | `unsupported stride pattern for transpose` | stride 顺序实现转置 | 用 `order` 参数表达转置，stride 反映真实布局 |
| 6 | `Failed to run BishengHIR pipeline` | TTAdapter→二进制失败 | `MLIR_ENABLE_DUMP=1` 定位 Pass；bishengir-compile 手动编译 |
| 7 | `misaligned memory access` | 对齐不足（Vector 32B；CV 融合 512B） | 起始地址/BLOCK_SIZE×element_size 对齐；`tl.multiple_of` 提示 |
| 8 | `all shapes must have size >= 1` | tensor 某维 size < 1 | 调用前检查，跳过空 tensor |
| 9 | `unsupported type: bool` | 部分操作不支持 bool | Triton 内部转 int8；必要时显式 `.to(tl.int8)` |
| 10 | `device_print requires a prefix string` | 缺 prefix | `tl.device_print("x: ", x)` |

## Python 层错误与 IR 手动分析

调用栈指向 Triton-Ascend Python 层时，在源文件插 `import pdb; pdb.set_trace()` 交互调试（`l`/`p`/`n`/`s`/`c`）。

IR 手动分析流程：dump IR → 查 TTIR（kernel/操作/形状类型）→ 查 TTAdapter（memref.alloc 大小、不支持的操作、搬运合理性）→ bishengir-compile 手动编译 → 对比 HIVM IR 确认操作映射为 SIMD 指令。

## 要点

1. UB 溢出最常见，优先查 BLOCK_SIZE/Tiling；coreDim 超限次之。
2. `MLIR_ENABLE_DUMP` 是定位 Pass 失败首选，90% 编译问题可由此定位。
3. `TRITON_ENABLE_LLVM_DEBUG` 仅怀疑 LLVM 后端 bug 时用，日志量极大。
4. dump 未生成 → 缓存命中，`TRITON_DISABLE_CACHE=1` 或清缓存。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [debugging.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/debugging.md)
