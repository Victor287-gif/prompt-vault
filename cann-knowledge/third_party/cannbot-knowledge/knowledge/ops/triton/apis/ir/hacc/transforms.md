---
type: API
title: HACC 变换 Pass（hacc-rename-func / hacc-append-device-spec）
description: HACC 方言两个变换 Pass：按 hacc.rename_func 属性重命名函数并更新引用，以及按 NPU 型号向 module 附加
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---

# HACC 变换 Pass

HACC 提供两个变换 Pass，为后续代码生成与运行时调度补齐信息。定义于 `bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td`。

| Pass | 作用域 | 构造函数 |
|------|--------|----------|
| `hacc-rename-func` | `FuncOp` | `mlir::hacc::createRenameFuncPass()` |
| `hacc-append-device-spec` | `ModuleOp` | `mlir::hacc::createAppendDeviceSpecPass()` |

## hacc-rename-func

按 `hacc.rename_func` 属性重命名函数，并更新模块内所有对该函数的调用引用。

```mlir
// 输入
func.func @bar() attributes {hacc.rename_func = #hacc.rename_func<@foo>} { return }
func.func @caller() { func.call @bar() : () -> (); return }

// 输出
func.func @foo() { return }
func.func @caller() { func.call @foo() : () -> (); return }
```

约束：目标函数名不得与模块中已有函数名冲突。

## hacc-append-device-spec

按 `--target` 指定的 NPU 型号，向 module 附加 `#hacc.target_device_spec`（DLTI 机制存储），供 Tiling、内存规划等 Pass 查询。依赖 `hacc::HACCDialect` 与 `mlir::DLTIDialect`。

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target` | `::mlir::hacc::TargetDevice` | `Unknown` | 目标设备名称 |

```bash
bishengir-opt --hacc-append-device-spec="target=Ascend910B4" input.mlir
```

变换效果：

```mlir
module attributes {
  #dlti.target_device_spec = #hacc.target_device_spec<
    #dlti.dl_entry<"AI_CORE_COUNT", 30 : i32>,
    #dlti.dl_entry<"UB_SIZE", 196608 : i32>,
    #dlti.dl_entry<"L1_SIZE", 1048576 : i32>, ...>
} { ... }
```

支持的型号清单与各型号精确规格值分别见 [device_specification.md](device_specification.md) 与 [hardware_specs_ir](../reference/hardware_specs_ir.md)。

## 在编译流程中的位置

```
前端 IR
  ├── hacc-append-device-spec   # 早期执行：附加设备规格，供后续 Pass 查询
  ├── HFusion 变换流程           # 融合 / 自动调度 / 向量化（见 ../hfusion/transforms.md）
  ├── hacc-rename-func          # 后期执行：函数名符合运行时要求
  └── 后端 Lowering
```

相关：[overview.md](overview.md)、[device_specification.md](device_specification.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td)
