---
type: API
title: HACC 设备规格（DeviceSpecEnum 与 hacc.target_device_spec）
description: DeviceSpecEnum 14 个硬件规格字段、TargetDeviceSpecAttr 基于 DLTI 的存储形式、TargetAttr、hacc-append-device-spec
  支持的 NPU 型号清单与内存层级背景。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']

---

# HACC 设备规格

HACC 通过 `DeviceSpecEnum` 枚举和 `TargetDeviceSpecAttr` 属性描述 NPU 硬件规格，编译器据此做 Tiling、内存分配等决策。规格基于 DLTI（Data Layout Target Information）机制存储在 module 上。

## DeviceSpecEnum 字段

| 枚举值 | 整数 | 说明 |
|--------|------|------|
| `AI_CORE_COUNT` | 0 | AI Core 数量 |
| `CUBE_CORE_COUNT` | 1 | Cube Core 数量（矩阵乘单元） |
| `VECTOR_CORE_COUNT` | 2 | Vector Core 数量（向量计算单元） |
| `UB_SIZE` | 3 | Unified Buffer 大小 |
| `L1_SIZE` | 4 | L1 缓存大小 |
| `L0A_SIZE` / `L0B_SIZE` | 5 / 6 | L0A/L0B 大小（矩阵乘 A/B 矩阵缓冲） |
| `L0C_SIZE` | 7 | L0C 大小（矩阵乘 C 结果缓冲） |
| `UB_ALIGN_SIZE` | 8 | UB 对齐大小 |
| `L1_ALIGN_SIZE` | 9 | L1 对齐大小 |
| `L0C_ALIGN_SIZE` | 10 | L0C 对齐大小 |
| `MINIMAL_D_CACHE_SIZE` / `MAXIMUM_D_CACHE_SIZE` | 11 / 12 | D Cache 最小/最大大小 |
| `ARCH` | 13 | 架构版本标识字符串 |

各型号这些字段的精确数值（910B/910_93/310B/950 全系列，bits 单位）见 [hardware_specs_ir](../reference/hardware_specs_ir.md)。

## TargetDeviceSpecAttr

`#hacc.target_device_spec` 携带 `DataLayoutEntryInterface` 条目数组，带 `TargetDeviceSpecTrait` 与 `HACCTargetDeviceSpecTrait`：

```mlir
#hacc.target_device_spec<
  #dlti.dl_entry<"AI_CORE_COUNT", 30 : i32>,
  #dlti.dl_entry<"UB_SIZE", 196608 : i32>>
```

注意 DLTI 条目中的容量字段单位是**字节**（如 UB_SIZE 196608 字节 = 192KB）；[hardware_specs_ir](../reference/hardware_specs_ir.md) 的规格表以 bits 计（同值 1572864 bits），两卡口径不同、换算一致。

`HACCTargetDeviceSpecInterface`（继承 `TargetDeviceSpecInterface`）提供 `getSpecForIdentifierEnum(DeviceSpec)`，按枚举值查规格条目——这是 Tiling Pass 读取 UB/L1 余量的标准入口。

另有 `TargetAttr`（`#hacc.target<"...">`）仅指示目标设备名称字符串。

## 支持的 NPU 型号（hacc-append-device-spec --target）

| 系列 | 型号标识 |
|------|----------|
| 910B | `Ascend910B1` ~ `Ascend910B4` |
| 910_93 | `Ascend910_9362` / `9372` / `9381` / `9382` / `9391` / `9392` |
| 310B | `Ascend310B1` ~ `Ascend310B4` |
| 950（910_95） | `Ascend910_950z` / `9579` / `957b` / `957d` / `9581` / `9589` / `958a` / `958b` / `9599` |
| 950PR | `Ascend950PR_950z` / `9579` / `957a`-`957d` / `9589` / `958a`-`958d` / `9599` / `959a` / `959b` |
| 950DT | `Ascend950DT_950x` / `950y` / `9571`-`9578` / `9581`-`9588` / `9591` / `9592` / `9595` / `9596` / `95A1` / `95A2` |

型号到规格值的附加由 `hacc-append-device-spec` Pass 完成，用法见 [transforms.md](transforms.md)。

## 内存层级背景

规格字段对应如下层级（架构解读见 [memory_hierarchy](../architecture/memory_hierarchy.md)）：

```
GM（全局内存，Host/Device 共享）
 └── L1 Cache（AI Core 一级缓存）
      ├── L0A / L0B / L0C（Cube 单元专用缓冲）
      └── UB（Unified Buffer，Vector 计算工作空间）
```

相关：[overview.md](overview.md)、[transforms.md](transforms.md)、[npu_hardware_overview](../architecture/npu_hardware_overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HACCAttrs.td#L241-L289](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L241-L289)
- [Passes.td#L51-L110](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td#L51-L110)
