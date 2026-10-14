---
type: API
title: AST → TTIR 生成（CodeGenerator 与 Ascend 扩展注入）
description: Triton-Ascend 前端 ast_to_ttir/CodeGenerator 的工作原理：Python AST 节点到 Triton
  IR 的映射规则、constexpr 编译期求值、builtin 处理，及 ascend_builder/buffer_builder/WITH_DISPATCH/extension.parallel
  的 Ascend 扩展注入机制。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# AST → TTIR 生成

编译流水线第一阶段：`@triton.jit` 函数 → Python AST → TTIR Module，由 `code_generator.py` 的 `CodeGenerator`（ast.NodeVisitor）完成。前后阶段见 [triton_pipeline_overview.md](triton_pipeline_overview.md)。

## CodeGenerator 架构

```
Python Kernel → fn.parse() → AST
  → CodeGenerator
     ├── builder (ir.builder)：标准 Triton IR 构建器（force_simt_only 时 compile_mode="simt"，否则 "simd"）
     ├── ascend_builder (ascendnpu_ir_builder)：Ascend 专用构建器
     ├── buffer_builder (buffer_ir.buffer_builder)：Buffer IR 构建器
     │    setup_unified_builder() 把 ascend_builder 方法注入 builder；
     │    setup_unified_builder_with_buffer_builder() 把 buffer_builder 方法注入 builder
     ├── gscope / lscope / local_defs(SSA) / constants / scf_stack
  → TTIR Module (mlir.Module)
```

入口 `ast_to_ttir(fn, specialization, context, options, codegen_fns, module_map, module=None)`：构建 prototype（含 constexpr 与 specialization），创建 CodeGenerator，`generator.visit(fn.parse())` 返回 Module。

## AST → IR 映射规则

语句映射：

| Python AST | Triton IR | 说明 |
|------------|-----------|------|
| `FunctionDef` | `tt.func` | kernel 定义 |
| `Assign` / `AnnAssign` | `tt.store` / SSA 赋值 | |
| `AugAssign` | 展开为 Assign + BinOp | |
| `Return` | `tt.return` | |
| `If` | `scf.if` / 条件编译 | 条件为 constexpr/Python bool 时只生成一个分支；为 tensor 时生成 `scf.if` 双分支 |
| `For` | `scf.for` / `tt.static_range` | |
| `While` | `scf.while` | |
| `With` | WITH_DISPATCH 分发 | |
| `Assert` | `tt.assert` | |

表达式映射：`Call`→Triton builtin；`BinOp`→`arith.addf/mulf` 等（`+ - * / // % ** << >> & | ^` 对应 dunder 方法）；`Compare`→`arith.cmpi/cmpf`；`Subscript`→`tt.load`/切片；`Constant`→constexpr；`Name`→SSA 查找。

## constexpr 处理

`tl.constexpr` 标记的值在 IR 生成期直接求值为 Python 值，不产生 IR 操作：

```
x: tl.constexpr = 42   →  不生成 IR，x = Python int 42
y = x + 1              →  编译期求值 y = 43
z = tl.load(ptr)       →  生成 tt.load
w = z * x              →  生成 arith.mulf(z, 42)
```

来源：参数注解 `x: tl.constexpr`、全局注解、`tl.constexpr(42)`。

**全局变量访问限制**：仅当是 constexpr、模块导入、JITFunction、Triton builtin 或 `triton.dtype` 时才可在 kernel 中访问；否则需 `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`（不保证长期支持）。

## builtin 与静态求值

`builtin_namespace` 含 `len/list/range/float/int/isinstance/getattr`，并重映射 `print→device_print`、`min→minimum`、`max→maximum`。编译期直接求值、不生成 IR 的函数：`tl.static_assert`、`tl.static_print`、`int`、`len`、`extension.int64`。

调用 `@triton.jit` 子函数时 `call_JitFunction()`：分离 constexpr 参数 → mangle 函数名 → 未定义则递归生成 → 产生 `call` 操作（后续由 Inliner 内联，见 [ttir_optimization.md](ttir_optimization.md)）。

## Ascend 扩展注入

- **扩展模块**：`triton.language.extra.cann.extension`，提供 `extension.parallel`（并行迭代器）、`extension.int64`、`extension.is_builtin(fn)`。
- **builder 选择**：`visit_Call()` 中 `extension.is_builtin(fn)` 为真时用 `ascend_builder`，否则用标准 `builder`——kernel 代码透明使用 Ascend 扩展。
- **WITH_DISPATCH**：`WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)` 注册 Ascend 扩展 `with` 语句处理器（如 `al.sync_block_*` 同步操作）。
- **extension.parallel**：`visit_For()` 中处理，设置 `hivm.parallel_loop` unit 属性。额外参数：

| 参数 | 说明 |
|------|------|
| `num_stages` | 流水线阶段数 |
| `loop_unroll_factor` | 循环展开因子 |
| `disallow_acc_multi_buffer` | 禁止累加器多缓冲 |
| `flatten` | 循环展平 |
| `warp_specialize` | Warp 特化 |
| `disable_licm` | 禁用 LICM |
| `bind_sub_block` | 绑定子块 |

## TTIR 示例（向量加法）

```mlir
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg3: i32 {tt.divisibility = 16 : i32}) {
    %0 = tt.get_program_id x : i32
    %1 = arith.constant dense<1024> : tensor<1024xi32>
    %2 = arith.muli %0, %1 : i32
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %4 = arith.addi %2, %3 : tensor<1024xi32>
    %5 = arith.cmpi slt, %4, %arg3 : tensor<1024xi1>
    %6 = tt.addptr %arg0, %4 : !tt.ptr<f32>, tensor<1024xi32>
    %7 = tt.load %6, %5 : tensor<1024xf32>
    %8 = tt.addptr %arg1, %4 : !tt.ptr<f32>, tensor<1024xi32>
    %9 = tt.load %8, %5 : tensor<1024xf32>
    %10 = arith.addf %7, %9 : tensor<1024xf32>
    %11 = tt.addptr %arg2, %4 : !tt.ptr<f32>, tensor<1024xi32>
    tt.store %11, %10, %5 : !tt.ptr<f32>
  }
}
```

constexpr 参数（如 `BLOCK_SIZE: tl.constexpr = 1024`）在 TTIR 中已被替换为常量；`if flag:`（flag 为 tensor）生成 `scf.if` 双分支。

## 添加新 Ascend IR 操作的步骤

1. 在 `triton.language.extra.cann.extension` 定义 Python 接口
2. C++ 端实现对应 Dialect/Operation
3. 在 `ascend_builder` 注册操作创建方法
4. 经 `setup_unified_builder()` 注入标准 builder

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [code_generator.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py)
- [code_generator.py#L1354](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py#L1354)
- [code_generator.py#L210](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py#L210)
- [code_generator.py#L273](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py#L273)
- [code_generator.py#L1121](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py#L1121)
- [code_generator.py#L11](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py#L11)
- [code_generator.py#L26](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/compiler/code_generator.py#L26)
