---
type: API
title: TTIR 优化 Pass（make_ttir 的 8 个后端无关 Pass）
description: make_ttir() 驱动的 TTIR 通用优化序列：Inliner/Combine/Canonicalizer/ReorderBroadcast/CSE/LICM/SymbolDCE/LoopUnroll
  的功能、典型 IR 变换与循环属性控制；该序列与 CUDA 后端共享，不含 Ascend 特定优化。
tags: [task.lookup, topic.compilation, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# TTIR 优化 Pass

`make_ttir()` 在 AST→TTIR 之后执行一组**与后端无关**的通用优化（CUDA/Ascend 共享同一序列），Ascend 特定优化在其后的 `ttir_to_linalg()` 阶段（见 [ascend_passes.md](ascend_passes.md)）。

```python
def make_ttir(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod)
    if opt.debug:  # dump kernel.ttir.mlir
        ...
    return mod
```

## Pass 序列与功能

| # | Pass | 来源 | 功能 |
|---|------|------|------|
| 1 | InlinerPass | `passes.common` | 内联 `@triton.jit` 辅助函数到调用点；`noinline` 函数不内联（其参数必须是 scalar 或 constexpr） |
| 2 | CombinePass | `passes.ttir` | 合并 Triton 操作模式：冗余类型转换链（extf→truncf 消除）、冗余 broadcast+reshape、可合并算术链 |
| 3 | CanonicalizerPass | `passes.common` | 规范化：加零/乘一消除、乘零替换、常量折叠 |
| 4 | ReorderBroadcastPass | `passes.ttir` | 重排 broadcast 位置，推迟广播以减少参与计算的数据量 |
| 5 | CSEPass | `passes.common` | 公共子表达式消除（操作级去重） |
| 6 | LICMPass | `passes.common` | 循环不变量外提（常量加载、循环独立地址计算） |
| 7 | SymbolDCEPass | `passes.common` | 死符号消除（符号级清理，与 CSE 互补） |
| 8 | LoopUnrollPass | `passes.ttir` | 循环展开，仅作用于小循环或带展开属性的循环 |

## 典型 IR 变换

Inliner：

```mlir
// Before
%2 = tt.call @helper(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
// After：helper 函数体直接嵌入调用点，tt.call 消失
```

CSE：

```mlir
// Before
%0 = arith.addf %a, %b : tensor<1024xf32>
%1 = arith.addf %a, %b : tensor<1024xf32>
// After：%1 消除，引用处改用 %0
```

LICM：循环内 `arith.constant dense<1.0>` 等不变量外提到 `scf.for` 之前。

Combine/Canonicalizer 综合效果示例：`tt.splat %arg4`（标量广播）被消除，标量直接参与 `arith.mulf %10, %arg4`。

## 循环展开属性（scf.for 操作属性）

| 属性 | 类型 | 说明 |
|------|------|------|
| `tt.loop_unroll_factor` | i32 | 展开因子 |
| `tt.num_stages` | i32 | 流水线阶段数 |
| `tt.disallow_acc_multi_buffer` | unit | 禁止累加器多缓冲 |
| `tt.flatten` | unit | 循环展平 |
| `tt.warp_specialize` | unit | Warp 特化 |
| `tt.disable_licm` | unit | 禁用 LICM |

这些属性可由前端 `extension.parallel`/`tl.range` 参数写入（见 [ttir_generation.md](ttir_generation.md)）。

## 调试

- `opt.debug=True` 或 `TRITON_KERNEL_DUMP=1`：优化后 TTIR 落盘为缓存目录下 `kernel.ttir.mlir`。
- `pm.enable_debug()` 开启 Pass Manager 调试输出。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [compiler.py#L72](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/compiler.py#L72)
