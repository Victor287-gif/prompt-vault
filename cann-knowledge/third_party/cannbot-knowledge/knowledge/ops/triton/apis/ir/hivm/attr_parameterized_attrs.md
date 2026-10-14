---
type: API
title: HIVM 参数化属性（DataLayoutAttr / AddressSpaceAttr / BlockMappingAttr 等）
description: HIVM 参数化属性定义：DataLayoutAttr（transpose/fractalSizes 约束）、AddressSpaceAttr、PipeAttr、Core
  类型三属性、Event/UnitFlag/SyncBlock 属性、BlockMapping/SubBlockMapping/TightlyCoupledBuffer
  的参数、IR 格式与推断规则。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.memory_management]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 参数化属性

参数化属性是携带参数的复合属性（对比纯枚举，枚举取值见 [attr_enumerations.md](attr_enumerations.md)）。

## DataLayoutAttr（`#hivm.data_layout`）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$data_layout` | EnumParameter\<DataLayoutEnum\> | 是 | 布局枚举值 |
| `$transpose` | OptionalParameter\<BoolAttr\> | 否 | 是否转置；**DOTA_ND/DOTB_ND 时必须提供** |
| `$fractalSizes` | OptionalParameter\<DenseI64ArrayAttr\> | 否 | Fractal 块大小 |

```mlir
#hivm.data_layout<ND>
#hivm.data_layout<dotA_ND, transpose = true>
#hivm.data_layout<dotB_ND, transpose = false>
#hivm.data_layout<nZ>
```

辅助方法：`getFractalSizesArray()`、`getTransposeValue()`、`isNDLayout()`（DOTA_ND/DOTB_ND/DOTC_ND/ND 归一类）、`getFractalBlockSizes()`（取 2 个 Fractal 块大小）。mmadL1 经 OpLayoutInterface 为 A/B/C/Bias 推导此属性（见 [macro_mmad_l1.md](macro_mmad_l1.md)）。

## AddressSpaceAttr（`#hivm.address_space`）

单参数 `$address_space`（AddressSpaceEnum），实现 `DeviceMappingAttrInterface`。

```mlir
#hivm.address_space<gm>    // 全局内存
#hivm.address_space<cbuf>  // L1
#hivm.address_space<ca>    // L0A
#hivm.address_space<cb>    // L0B
#hivm.address_space<cc>    // L0C
#hivm.address_space<ub>    // UB
```

## Pipe / Core 类型属性

| 属性 | IR 格式 | 参数 |
|------|---------|------|
| PipeAttr | `#hivm.pipe<PIPE_M>` / `<PIPE_MTE2>` / `<PIPE_FIX>` / `<PIPE_UNASSIGNED>` | `$pipe`（PipeEnum） |
| TCoreTypeAttr | `#hivm.tcore_type<CUBE>` / `<VECTOR>` / `<CUBE_OR_VECTOR>` / `<CUBE_AND_VECTOR>` | `$tcoretype` |
| TFuncCoreTypeAttr | `#hivm.func_core_type<AIC>` / `<AIV>` / `<MIX>` | `$funcCoreType` |
| TModuleCoreTypeAttr | `#hivm.module_core_type<...>` | `$moduleCoreType`；推断：全 AIV→AIV、全 AIC→AIC、否则 MIX |

## 同步属性

| 属性 | 参数 | 说明 |
|------|------|------|
| EventAttr（`#hivm.event<EVENT_ID0>`~`<EVENT_ID7>`） | `$event` | 静态 Event ID |
| UnitFlagAttr | `$unit_flag` | 单输出 UnitFlag 模式 |
| UnitFlagArrayAttr | UnitFlagAttr 数组 | 宏操作多输出的模式数组（用法见 [sync_unit_flag.md](sync_unit_flag.md)） |
| SyncBlockModeAttr | `$sync_mode` | 跨核同步模式 |
| SyncBlockInstrModeAttr | `$sync_instr_mode` | 默认 INTRA_BLOCK_SYNCHRONIZATION |

## 映射与缓冲区属性

### BlockMappingAttr（`#hivm.block`）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$order` | OptionalParameter\<optional\<int32_t\>\> | 线性维度序号 |

```mlir
#hivm.block<linear_dim = 0>
#hivm.block
```

实现 `DeviceMappingAttrInterface`。

### SubBlockMappingAttr（`#hivm.sub_block`）

`$sub_block`（MappingIdEnum，必选）：Mix Kernel 中 Cube/Vector Block 维度比例的映射。

### TightlyCoupledBufferAttr（`#hivm.tightly_coupled_buffer`）

`$id`（可选 int32）：Cube-Vector 紧耦合缓冲区 ID，允许 Cube 与 Vector Core 共享数据。内存规划语义见 [tightly_coupled_buffer](../memory_management/tightly_coupled_buffer.md)。

### MemoryEffectAttr

`$effect`（MemoryEffectEnum）：SIMT VF 模式下的内存效果标记。

相关：[attr_overview.md](attr_overview.md)、[attr_enumerations.md](attr_enumerations.md)、[attr_type_system.md](attr_type_system.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td#L103-L165](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L103-L165)
- [HIVMAttrs.td#L190-L197](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L190-L197)
- [HIVMAttrs.td#L238-L244](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L238-L244)
- [HIVMAttrs.td#L311-L317](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L311-L317)
- [HIVMAttrs.td#L263-L269](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L263-L269)
- [HIVMAttrs.td#L278-L292](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L278-L292)
- [HIVMAttrs.td#L500-L506](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L500-L506)
- [HIVMAttrs.td#L525-L531](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L525-L531)
- [HIVMAttrs.td#L557-L563](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L557-L563)
- [HIVMAttrs.td#L580-L590](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L580-L590)
- [HIVMAttrs.td#L712-L720](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L712-L720)
- [HIVMAttrs.td#L722-L730](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L722-L730)
- [HIVMAttrs.td#L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)
- [HIVMAttrs.td#L1057-L1063](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1057-L1063)
- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [attribute.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/attribute.mlir)
