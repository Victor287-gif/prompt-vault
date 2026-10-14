---
type: API
title: HIVM 属性类型系统总览
description: HIVM 属性类型系统五类组成：枚举属性（30+）、参数化属性（DataLayout/AddressSpace/Pipe/BlockMapping
  等）、标记属性（multi_buffer/part_of_mix 等）、类型约束、接口与 Trait 的全景索引。
tags: [task.lookup, topic.compilation, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 属性类型系统总览

HIVM 把 NPU 硬件约束（存储层次、Pipe、Core 类型、数据布局）编码进 MLIR 属性系统，使编译器在类型检查与 Lowering 阶段正确处理。分五类：

## 1. 枚举属性（30+ 个）

有限离散值集合，IR 格式 `#hivm.<mnemonic><<VALUE>>`，如 `#hivm.pipe<PIPE_M>`。完整取值表见 [attr_enumerations.md](attr_enumerations.md)：IteratorType、DataLayout、AddressSpace、Pipe、TCoreType、Event、UnitFlag、SyncBlockMode、ReduceOperation、AtomicKind、VFMode 等。

## 2. 参数化属性

携带参数的复合属性（详见 [attr_parameterized_attrs.md](attr_parameterized_attrs.md)）：

| 属性 | 助记符 | 参数 | 说明 |
|------|--------|------|------|
| DataLayoutAttr | `data_layout` | data_layout, transpose?, fractalSizes? | 数据布局映射 |
| AddressSpaceAttr | `address_space` | address_space | 地址空间映射 |
| PipeAttr | `pipe` | pipe | Pipeline 标识 |
| TCoreTypeAttr | `tcore_type` | tcoretype | 操作 Core 类型 |
| TFuncCoreTypeAttr | `func_core_type` | funcCoreType | 函数 Core 类型 |
| TModuleCoreTypeAttr | `module_core_type` | moduleCoreType | 模块 Core 类型 |
| EventAttr | `event` | event | Event ID |
| UnitFlagAttr | `unit_flag` | unit_flag | UnitFlag 模式 |
| SyncBlockModeAttr | `sync_block_mode` | sync_mode | 跨核同步模式 |
| SyncBlockInstrModeAttr | `sync_block_instr_mode` | sync_instr_mode | 同步指令模式 |
| ReduceOpAttr | `reduce_op` | reduce_op | 归约操作类型 |
| BlockMappingAttr | `block` | order? | Block 映射 |
| SubBlockMappingAttr | `sub_block` | sub_block | Sub-Block 映射 |

## 3. 标记属性

| 属性 | 助记符 | 说明 |
|------|--------|------|
| MultiBufferAttr | `multi_buffer` | 多缓冲标记（见 [multi_buffer](../memory_management/multi_buffer.md)） |
| TPartOfMixAttr | `part_of_mix` | Mix Kernel 组成部分 |
| VFAttr | `vector_function` | Vector 函数标记 |
| HasAliaScopesAttr | `has_alias_scopes` | 别名作用域标记 |
| TightlyCoupledBufferAttr | `tightly_coupled_buffer` | 紧耦合缓冲区（含 id 参数，见 [tightly_coupled_buffer](../memory_management/tightly_coupled_buffer.md)） |
| MemUniqueAttr | `mem_unique` | 唯一内存规划 |
| ParallelLoopAttr | `parallel_loop` | 可并行化循环标记 |
| UnlikelyConditionAttr | `unlikely_condition` | 不可能条件标记 |
| SharedMemoryAttr | `shared_memory` | SIMT VF 共享内存 |

## 4. 类型约束

MemRef 的 AddressSpace 标注、Tensor 双语义、元素类型限制与矩阵乘类型组合，见 [attr_type_system.md](attr_type_system.md)。

## 5. 接口与 Trait

- 接口（HIVMStructuredOpInterface、HIVMCoreTypeInterface、HIVMInferCoreTypeInterface、HIVMUnitFlagEnabledInterface、OpLayoutInterface）：编译器 Pass 与操作之间的契约，见 [attr_interfaces.md](attr_interfaces.md)。
- Trait（MacroOpTrait、CoreTypeTrait、ElementwiseNaryOpTrait、BroadcastableOTF 等）：操作行为约束与分类标记，见 [attr_traits.md](attr_traits.md)。

相关：[attr_enumerations.md](attr_enumerations.md)、[attr_parameterized_attrs.md](attr_parameterized_attrs.md)、[overview.md](overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [attribute.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/attribute.mlir)
