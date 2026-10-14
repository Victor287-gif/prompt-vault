---
type: API
title: 硬件规格 IR 映射（DeviceSpec 枚举字段与各型号 TargetDeviceSpecAttr 值）
description: HACC 方言 DeviceSpec 枚举 14 个字段定义、四大系列（910B/910_93/310B/950）共享 BaseSpec
  的 bits 级精确值、逐型号核心数差异与
tags: [task.lookup, topic.architecture, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']

---

# 硬件规格 IR 映射

把 NPU 型号映射到 HACC 方言 `TargetDeviceSpecAttr` 的字段值，全部数据从源码（NPUTargetSpec.td / HACCAttrs.td）精确提取。架构层面的解读见 [npu_hardware_overview](../architecture/npu_hardware_overview.md)。

## DeviceSpec 枚举字段（每个字段 = TargetDeviceSpecAttr 中一个 dl_entry）

| 枚举值（整数） | 对应 TargetSpec 字段 | 说明 |
|----------------|----------------------|------|
| AI_CORE_COUNT(0) | AiCoreCount | AI Core 数量 |
| CUBE_CORE_COUNT(1) | CubeCoreCount | Cube Core 数量 |
| VECTOR_CORE_COUNT(2) | VectorCoreCount | Vector Core 数量（= 2×Cube） |
| UB_SIZE(3) | UbSize | UB 大小（bits） |
| L1_SIZE(4) | L1Size | L1 大小（bits） |
| L0A_SIZE(5) / L0B_SIZE(6) | L0aSize / L0bSize | L0A/L0B 大小（bits） |
| L0C_SIZE(7) | L0cSize | L0C 大小（bits） |
| UB_ALIGN_SIZE(8) | UbAlignSize | UB 对齐（bits） |
| L1_ALIGN_SIZE(9) | L1AlignSize | L1 对齐（bits） |
| L0C_ALIGN_SIZE(10) | L0cAlignSize | L0C 对齐（bits） |
| MINIMAL_D_CACHE_SIZE(11) | MinimalDCacheSize | 最小 DCache（bits） |
| MAXIMUM_D_CACHE_SIZE(12) | MaximumDCacheSize | 最大 DCache（bits） |
| ARCH(13) | Arch | 架构标识字符串 |

## 各系列共享 BaseSpec（bits 精确值）

| 字段 | 910B / 910_93 | 310B | 950（910_95/950PR/950DT） |
|------|---------------|------|----------------------------|
| UbSize | 1572864（192KB） | 2097152（256KB） | 2031616（248KB，预留 8KB） |
| L1Size | 4194304（512KB） | 8388608（1024KB） | 4194304（512KB） |
| L0aSize / L0bSize | 524288（64KB） | 524288 | 524288 |
| L0cSize | 1048576（128KB） | 1048576 | 2097152（256KB） |
| UbAlignSize / L1AlignSize | 256（32B） | 256 | 256 |
| L0cAlignSize | 4096（512B） | 4096 | 4096 |
| DCache（Min~Max） | — | — | 262144 ~ 983040（32~120KB） |
| Arch | "dav-c220" | "dav-m300" | "dav-c310" |

## 逐型号核心数（AiCoreCount / CubeCoreCount / VectorCoreCount）

| 系列 | 型号分档 |
|------|----------|
| 910B | B1/B2 = 24/24/48；B3/B4 = 20/20/40 |
| 910_93 | 9362/9372 = 20/20/40；9381/9382/9391/9392 = 24/24/48 |
| 310B | B1-B4 全部 = 1/1/1 |
| 910_95 | 950z = 4/4/8；9579/957b/957d = 28/28/56；9581/9589/958a/958b = 32/32/64；9599 = 36/36/72 |
| 950PR | 950z = 4/4/8；9579/957a-d = 28/28/56；9589/958a-d = 32/32/64；9599/959a/959b = 36/36/72 |
| 950DT | 950x/950y = 8/8/16；9571-9578 = 28/28/56；9581-9588 = 32/32/64；9591-9596/95A1/95A2 = 36/36/72 |

## IR 表示示例（Ascend910B1）

```mlir
#hacc.target_device_spec<
  #dlti.dl_entry<"AI_CORE_COUNT", 24 : i32>,
  #dlti.dl_entry<"CUBE_CORE_COUNT", 24 : i32>,
  #dlti.dl_entry<"VECTOR_CORE_COUNT", 48 : i32>,
  #dlti.dl_entry<"UB_SIZE", 1572864 : i32>,
  #dlti.dl_entry<"L1_SIZE", 4194304 : i32>,
  #dlti.dl_entry<"L0A_SIZE", 524288 : i32>,
  #dlti.dl_entry<"L0B_SIZE", 524288 : i32>,
  #dlti.dl_entry<"L0C_SIZE", 1048576 : i32>,
  #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>,
  #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>,
  #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>,
  #dlti.dl_entry<"ARCH", "dav-c220">>
```

## 架构系列对比

| Arch | UB | L1 | L0C | DCache | 代表型号 |
|------|-----|-----|-----|--------|----------|
| dav-c220 | 192KB | 512KB | 128KB | — | 910B、910_93 |
| dav-m300 | 256KB | 1024KB | 128KB | — | 310B |
| dav-c310 | 248KB | 512KB | 256KB | 32-120KB | 950 全系列 |

排障提示：UB 溢出报错中的 requires/available 数值单位与本表一致（bits），如 910B 报 `1572864 bits available`；内存规划溢出检测见 [overview](../memory_management/overview.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HACCAttrs.td#L241-L262](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L241-L262)
- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- [HACCAttrs.td#L269-L289](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L269-L289)
