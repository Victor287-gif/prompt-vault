---
type: API
title: HACC 方言总览（Host-Device 异构调用模型）
description: HACC（Heterogeneous Async Computing Call）方言的定位、核心概念（HACCFuncType / DeviceSpecEnum
  / KernelArgType / HostFuncType）、HACCFunctionInterface 与典型 IR 形态。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HACC 方言总览

HACC（Heterogeneous Async Computing Call，异构异步计算调用）是 BiShengIR 编译栈中描述 Host-Device 异构计算模型的方言。它本身不提供计算操作，而是以**属性和接口**的形式标注在 `func.func` 上：函数归属（Host/Device）、NPU 设备规格、Kernel 参数语义、Host 辅助函数绑定，为 HFusion/HIVM  lowering 与运行时调度提供异构语义基础。

- **方言名称**：`hacc`，C++ 命名空间 `::mlir::hacc`
- **依赖方言**：`mlir::DLTIDialect`（设备规格通过 DLTI 机制存储为 `#dlti.dl_entry`）
- **TableGen 定义**：`bishengir/include/bishengir/Dialect/HACC/IR/` 下的 HACCBase.td / HACCAttrs.td / HACCInterfaces.td，Pass 定义在 `Dialect/HACC/Transforms/Passes.td`

## 核心概念速查

| 概念 | 作用 | 详解卡 |
|------|------|--------|
| `HACCFuncType` | 函数 Host/Device 归属（HOST=1 / DEVICE=2） | [function_management.md](function_management.md) |
| `HACCFunctionInterface` | 异构函数统一查询/设置接口（isHost / isDevice / setDevice 等） | [function_management.md](function_management.md) |
| `DeviceSpecEnum` | NPU 硬件规格 14 个字段枚举（核数、UB/L1/L0 大小、对齐等） | [device_specification.md](device_specification.md) |
| `TargetDeviceSpecAttr` | `#hacc.target_device_spec<...>`，把型号映射为 DLTI 条目集合 | [device_specification.md](device_specification.md) |
| `KernelArgType` | Kernel 参数语义分类（input/output/workspace/tiling_data 等 12 类） | [kernel_args.md](kernel_args.md) |
| `HostFuncType` | Host 端辅助函数角色（tiling_function / infer_output_shape 等 8 类） | [host_device_binding.md](host_device_binding.md) |
| FuncRefAttr 系列 | 把 Host 辅助函数符号绑定到 Device 函数上 | [host_device_binding.md](host_device_binding.md) |

## 与其他方言的关系

```
HACC 方言
  ├── 被 HFusion 依赖（HFusion 依赖 hacc::HACCDialect，见 ../hfusion/overview.md）
  ├── 与 DLTIDialect 协作（target_device_spec 基于 DataLayoutEntryInterface 存储）
  ├── 为 HIVM 层提供 Host-Device 调用约定基础（hivm 依赖 hacc，见 ../hivm/overview.md）
  └── Scope 方言间接使用（scope.scope 可携带 tcore_type 等属性）
```

## 典型 IR 形态

```mlir
func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
  attributes {hacc.function_kind = #hacc.function_kind<HOST>} { ... }

func.func @device_kernel(%arg0: tensor<?x?xf16>,
                         %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
  attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
              hacc.tiling_function = #hacc.tiling_function<@tiling_func>} { ... }
```

读 IR 时的判读要点：`hacc.function_kind` 决定函数跑在 CPU 还是 NPU；Device 入口函数上的 `hacc.tiling_function` 等符号引用指向 Host 端在运行前先行执行的辅助计算；参数上的 `hacc.arg_type` 决定该参数在 Kernel 启动时如何被运行时 marshaling。

## 硬件背景与相关卡

- 设备规格字段的逐型号精确数值见 [hardware_specs_ir](../reference/hardware_specs_ir.md)
- GM/L1/L0A/L0B/L0C/UB 层级含义见 [memory_hierarchy](../architecture/memory_hierarchy.md)
- HACC 变换 Pass（函数重命名、设备规格附加）见 [transforms.md](transforms.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HACCBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td)
- [HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td)
- [HACCInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td)
