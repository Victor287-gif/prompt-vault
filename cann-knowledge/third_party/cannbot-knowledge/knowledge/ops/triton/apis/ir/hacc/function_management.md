---
type: API
title: HACC 函数管理（hacc.function_kind 与 HACCFunctionInterface）
description: HACCFuncType 枚举（HOST/DEVICE）、HACCFunctionInterface 的查询/设置/参数判定方法、HACCToLLVMIRTranslateAttr
  翻译标记与 rename_func、input_idx/output_idx 等辅助属性。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HACC 函数管理

HACC 用 `HACCFuncType` 枚举给每个函数打上 Host/Device 归属标记，并通过 `HACCFunctionInterface`（继承 `FunctionOpInterface`）向 Pass 暴露统一的查询与设置方法。

## HACCFuncType 枚举

| 枚举值 | 整数值 | 说明 |
|--------|--------|------|
| `HOST` | 1 | Host 端函数，运行在 CPU 上 |
| `DEVICE` | 2 | Device 端函数，运行在 NPU 上 |

IR 表示：`hacc.function_kind = #hacc.function_kind<HOST>` / `<DEVICE>`。

## HACCFunctionInterface 方法

### 查询方法

| 方法 | 返回 | 说明 |
|------|------|------|
| `getHACCFuncType()` | `std::optional<HACCFuncType>` | 函数 HACC 类型，未知返回 `nullopt` |
| `isHost()` / `isDevice()` | `bool` | Host / Device 函数判定 |
| `isDeviceEntry()` | `bool` | 是否为 Device 入口函数 |
| `getHostFuncType()` | `std::optional<HostFuncType>` | Host 函数角色（取值见 [host_device_binding.md](host_device_binding.md)） |
| `isKernelArg(argIdx, argType)` | `bool` | 第 argIdx 个参数是否具有指定 `hacc.arg_type`（取值见 [kernel_args.md](kernel_args.md)） |

### 设置方法

| 方法 | 说明 |
|------|------|
| `setDevice()` / `setDeviceEntry()` | 标记为 Device 函数 / Device 入口函数，自动移除不允许的属性 |
| `setHost()` | 标记为 Host 函数，自动移除不允许的属性 |
| `setHostFuncType(funcType)` | 设置 Host 函数的具体角色类型 |

## HACC To LLVM 翻译标记（HACCToLLVMIRTranslateAttr）

标记函数在 LLVM IR 翻译阶段的行为：

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `ENTRY`(0) | `hacc.entry` | Device 入口函数 |
| `MIX_ENTRY`(1) | `hacc.mix_entry` | 混合 Device 入口函数 |
| `ALWAYS_INLINE`(2) | `hacc.always_inline` | 始终内联函数 |

## 辅助属性

| 属性 | 参数 | 说明 |
|------|------|------|
| `hacc.rename_func` | `FlatSymbolRefAttr:$targetName` | 函数应重命名为目标符号，由 `hacc-rename-func` Pass 执行（见 [transforms.md](transforms.md)） |
| `hacc.input_idx` | `unsigned:$argIdx` | 标记函数参数为输入索引 |
| `hacc.output_idx` | `unsigned:$argIdx` | 标记哪个输入参数对应哪个输出值（NPU Kernel 采用 out-param 模式，输出通过参数传入） |
| `hacc.export_as_dag` | — | 将函数导出为 DAG |
| `hacc.dummy_func` | — | 虚拟函数标记 |
| `hacc.external_function_path` | — | 外部函数路径 |
| `hacc.cached_io` | — | 标记值已被缓存 IO |
| `hacc.no_io_alias` | — | 标记函数输入输出严格不别名 |
| `hacc.block_dim` | — | 函数的 Block 维度属性 |
| `hacc.simt_module` | — | 标记 SIMT 模块 |

## 典型使用模式

```mlir
// Host 函数（Tiling 计算）
func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} { ... }

// Device 入口函数（带 tiling 参数与 Host 绑定）
func.func @kernel_entry(%arg0: tensor<?x?xf16>,
                        %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
              hacc.tiling_function = #hacc.tiling_function<@tiling_func>} { ... }

// 函数重命名：hacc-rename-func 后 @bar → @foo
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} { ... }
```

相关：[overview.md](overview.md)、[host_device_binding.md](host_device_binding.md)、[transforms.md](transforms.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HACCInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td)
