---
type: Guide
title: 错误码参考（报错信息对照表）
description: Triton-Ascend 编译/运行时错误信息到原因与解决方案的对照表，含 MLIRCompilationError、UB 溢出、对齐、死锁、精度错误及
  NPU 执行错误码。
tags: [task.debugging, topic.compilation, topic.implementation.debugging]
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
# 错误码参考（报错信息对照表）

编译错误发生在 `ttir.mlir → ttadapter.mlir` 转换阶段；运行时错误发生在 kernel 执行阶段。

## 编译错误

### MLIRCompilationError

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `failed to legalize operation` | 操作无法转换到 Ascend 后端 | 查 API 支持矩阵确认是否支持 |
| `unsupported data type` | NPU 不支持的数据类型 | uint8→int32、fp64→fp32 等 |
| `failed to lower tt.dot` | dot 降级失败 | 查输入类型；确保 BLOCK_SIZE 对齐 |
| `pattern failed to match` | 优化 pattern 匹配失败 | `MLIR_ENABLE_DUMP=1` 定位失败位置 |

### UB / Shape / 指针 / 数据类型

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `ub overflow` / `UB space exceeded` | 片上存储超限 | 减小 BLOCK_SIZE；减少中间 tensor；int8 占更多空间，考虑 int32 |
| `local memory limit exceeded` | 局部内存超限 | 减少局部变量数量和大小 |
| `shape size must be >= 1` | 某维 size < 1 | 确保所有维度 ≥ 1 |
| `rank mismatch` | 维度不匹配 | 查广播和 reshape |
| `block size must be power of 2` | BLOCK_SIZE 非 2 幂 | `triton.next_power_of_2()` |
| `make_block_ptr result cannot be used in arithmetic` | block_ptr 不允许算术运算 | 用 `tl.advance` 或重建 `make_block_ptr` |
| `unsupported stride permutation for transpose` | stride 调序转置 | NPU 不允许，用 `order` 参数 |
| `unsupported element type: uint8/uint16/uint32/uint64/fp64` | 不支持的类型 | 替换为 int 类型或 fp32 |
| `dot input type mismatch` | dot 输入类型不一致 | 两输入类型一致且受支持 |
| `accumulator type not supported` | 累加器类型不支持 | dot 累加器必须 fp32 |

## 运行时错误

### 内存访问 / 同步 / 精度

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `illegal memory access` / `segfault` | 非法内存访问 | 查 mask；确保指针偏移不越界 |
| `misaligned address` | 地址未对齐 | NPU 要求 512B 对齐，BLOCK_SIZE 满足对齐 |
| `out of memory` | GM 不足 | 减小 tensor 或 batch size |
| `deadlock detected` | 死锁 | 查 `atomic_cas` 自旋锁是否释放；避免条件分支中取锁 |
| `barrier timeout` | 屏障超时 | 查 `debug_barrier`；确保所有 program 到达屏障 |
| `NaN detected` | NaN | 查除零、0 的负次幂；`tl.where` 保护 |
| `Inf detected` | 无穷 | 查 exp 溢出；减最大值技巧 |

### NPU 执行错误码与超时

| 错误码 | 含义 | 解决方案 |
|--------|------|---------|
| `AICORE_ERROR` | AI Core 执行错误 | 查 kernel 逻辑，解释器模式验证 |
| `AIV_ERROR` | AI Vector 执行错误 | 查 Vector 操作数据类型和 shape |
| `DMA_ERROR` | 数据搬运错误 | 查地址对齐和数据连续性 |
| `CUBE_ERROR` | Cube 矩阵计算错误 | 查 dot 输入格式和类型 |
| `kernel execution timeout` | 执行超时 | 查死循环或死锁 |
| `event timeout` | 同步事件超时 | 查 `sync_block_set/wait` 是否配对 |

## 高频错误详解

1. **`failed to legalize operation 'tt.dot'`**：输入须 fp16/fp32/bf16/int8；加 `al.compile_hint(a, "dot_pad_only_k")`；查 BLOCK_M/N 合法性。
2. **`ub overflow`**：kernel 全部 tensor 片上总和超 UB。double buffer 下 A2/A3 ≤96KB×2=192KB、910_95 ≤128KB×2=256KB；int8 改 int32。
3. **`unsupported data type uint8`**：NPU 硬件不支持无符号整数，host 端转 int32，算完转回。
4. **`make_block_ptr` + 复杂循环/分支编译失败**：已知限制，改手动指针算术或简化控制流。
5. **`atomic_add` 多核累加不支持**：用 `atomic_cas` 自旋锁保护共享缓冲区，或两阶段归约。

## 错误关键字 → 排查方向

- IR 转换：`legalize`（后端不支持）/ `lower`（查数据类型）/ `verify`（操作语义）/ `pattern` / `conversion`。
- 内存布局：`layout` / `encoding` / `allocation` / `padding`。
- 维度 Shape：`broadcast` / `reshape` / `transpose` / `stride`。

## 排查流程

1. 定类型：编译错 → 看完整堆栈 + `MLIR_ENABLE_DUMP=1`；运行时错 → 按内存/同步/精度分支。
2. 启调试：`TRITON_DEBUG=1` + `MLIR_ENABLE_DUMP=1`（或 `=kernel_name`）+ 清缓存 `rm -r ~/.triton/cache`。
3. 解释器验证：`TRITON_INTERPRET=1` 取 CPU 基准。
4. 分层定位：Python 层用 `pdb.set_trace()`；C++ 层用 gdb/core dump。
5. 简化 kernel：只留 load/store，逐步加回计算，每步验证。

## 调试工具速查

| 工具 | 用途 | 命令/变量 |
|------|------|----------|
| IR 转储 | 编译中间产物 | `MLIR_ENABLE_DUMP=1` |
| 解释器 | CPU 验证正确性 | `TRITON_INTERPRET=1` |
| 设备打印 | kernel 内打印 | `TRITON_DEVICE_PRINT=1` + `tl.device_print` |
| 内存分析 | UB 占用 | `TRITON_MEMORY_DISPLAY=1` |
| 编译时间 | 编译瓶颈 | `MLIR_ENABLE_TIMING=1` |
| 缓存清理 | 强制重编 | `rm -r ~/.triton/cache` |


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [debugging.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/debug_guide/debugging.md)
