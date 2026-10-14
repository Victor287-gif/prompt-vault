---
type: API
title: HIVM 接口清单（HIVMStructuredOpInterface 等五大接口）
description: HIVM OpInterface 清单：HIVMStructuredOpInterface（迭代器/广播/转置/indexing maps/形状查询方法族）、HIVMCoreTypeInterface、HIVMInferCoreTypeInterface、HIVMUnitFlagEnabledInterface、OpLayoutInterface
  的方法与用途。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 接口清单

接口是编译器 Pass（同步注入、Lowering、Bufferization 等）与具体操作之间的契约。

## HIVMStructuredOpInterface（核心接口）

继承自 DestinationStyleOpInterface、OpPipeInterface、HIVMCoreTypeInterface、FlattenInterface、LibraryFunctionOpInterface。设计上借鉴 LinalgOp，增加 Pipe/CoreType/Flatten 等 NPU 特有维度。

### 操作属性查询

`isElemwiseNaryOp()` / `isInlineBroadcastable()` / `isInlineTransposable()` / `existInlineBroadcastLoopDims()` / `existInlineTransposeLoopDims()`（均 bool）。

### 循环类型处理

| 方法 | 说明 |
|------|------|
| `getIteratorTypesArray()` / `setIteratorTypesArray(...)` | 迭代器类型数组（IteratorType 枚举见 [attr_enumerations.md](attr_enumerations.md)） |
| `getNumLoops()` / `getNumParallelLoops()` | 循环总数 / 并行循环数 |
| `getParallelLoopDims` / `getReductionLoopDims` / `getBroadcastLoopDims` / `getTransposeLoopDims` / `getPadLoopDims` / `getConcatLoopDims` / `getGatherLoopDims` | 各类循环维度 |
| `getPermutationArray()` / `getBroadcastArray()` / `getInlinedBroadcastableAxes(op)` | 转置排列 / 广播数组 / 内联广播轴 |

### Indexing Maps

`getIndexingMaps()`（ArrayAttr）/ `getIndexingMapsArray()`（AffineMap 数组）/ `getLoopsToShapesMap()` / `getShapesToLoopsMap()` / `getMatchingIndexingMap(op)` / `getIndexingMapMatchingResult(res)` / `getIndexingMapIndex(op)`。

### 形状与 Rank

`getRank(op)` / `getShape(op)` / `getStaticShape()` / `hasDynamicShape()` / `hasIndexSemantics()`。

### 内存效果与操作数

`getEffects(...)` / `getTargetSpaceOperands(AddressSpace, bool)` / `getHIVMOperands(bool)` / `getHIVMInputOperands(bool)` / `getHIVMOperandTypes(bool)` / `isVectorOnlyOperand(idx)` / `getContiguousAxes()` / `getUnitAxesMask()` / `getPermutedAxesMask()`。

### 形状具体化

`createFlatListOfOperandDims` / `createFlatListOfOperandStaticDims()` / `createLoopRanges` / `computeStaticLoopSizes()` / `reifyResultShapes(...)`。

## HIVMCoreTypeInterface

| 方法 | 说明 |
|------|------|
| `getCoreType()` | `optional<TCoreType>`，操作在哪种 Core 执行 |

Core 类型两种确定方式：静态（`CoreTypeTrait` 附加）或动态（InferCoreTypeInterface 推断）。

## HIVMInferCoreTypeInterface

| 方法 | 默认 | 说明 |
|------|------|------|
| `inferCoreType()` | 返回 `std::nullopt` | 动态推断 Core 类型 |

用于 Core 类型无法静态确定的操作（如同步操作可能在 Cube 或 Vector 上执行）。

## HIVMUnitFlagEnabledInterface

支持 UnitFlag 同步的操作接口（机制见 [sync_unit_flag.md](sync_unit_flag.md)）：

| 方法 | 说明 |
|------|------|
| `getUnitFlagModes()` / `setUnitFlagModes(...)` | UnitFlag 模式数组读写 |
| `getUnitFlagConditions()` / `setUnitFlagConditions(...)` | 条件值读写 |
| `getUnitFlagModeLibValue(PatternRewriter&)` | 库调用编码值（0b00/0b10/0b11） |

自带验证 `verifyUnitFlagEnabledInterface`，检查 `unit_flag_mode`/`unit_flag_cond` 声明。

## OpLayoutInterface

确定操作数目标 Fractal 布局：

| 方法 | 说明 |
|------|------|
| `getOperandsTargetFractalLayout()` | 操作自行实现 |

mmadL1 实现此接口并提供 `getOperandALayout()` / `getOperandBLayout()` / `getOperandCLayout()` / `getOperandBiasLayout()`（均 `FailureOr<DataLayoutAttr>`，见 [macro_mmad_l1.md](macro_mmad_l1.md)）；编译器据此生成正确的数据搬运指令。

相关：[attr_overview.md](attr_overview.md)、[attr_traits.md](attr_traits.md)、[attr_type_system.md](attr_type_system.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMInterfaces.td#L67-L678](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L67-L678)
- [HIVMInterfaces.td#L27-L46](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L27-L46)
- [HIVMInterfaces.td#L48-L65](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L48-L65)
- [HIVMInterfaces.td#L680-L761](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L680-L761)
- [OpLayoutInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.td)
- [HIVMInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td)
