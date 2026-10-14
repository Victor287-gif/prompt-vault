---
type: API
title: HIVM 枚举属性速查（30+ 枚举完整取值表）
description: HIVM 方言全部枚举属性的取值速查：IteratorType/DataLayout/AddressSpace/Pipe/TCoreType/Event/UnitFlag/SyncBlockMode/ReduceOperation/AtomicKind/VFMode/DescaleMode/MatmulBiasMode
  等 30 个枚举的整数值与语义。
tags: [task.lookup, topic.compilation, topic.architecture, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 枚举属性速查

全部枚举值从 HIVMAttrs.td 精确提取。IR 中格式为 `#hivm.<mnemonic><<ENUM_VALUE>>`（如 `#hivm.pipe<PIPE_M>`）。参数化封装形式见 [attr_parameterized_attrs.md](attr_parameterized_attrs.md)。

## IteratorType（12 值）

parallel(0) 并行 / broadcast(1) 广播 / transpose(2) 转置 / reduction(3) 归约 / interleave(4) 交织 / deinterleave(5) 解交织 / inverse(6) 逆序 / pad(7) 填充 / concat(8) 拼接 / gather(9) 收集 / cumulative(10) 累积 / opaque(99) 不透明。

## DataLayout（7 值）

| 值 | 字符串 | 说明 |
|---|--------|------|
| 1 | `dotA_ND` | 矩阵 A 的 ND 布局（Cube 矩阵乘） |
| 2 | `dotB_ND` | 矩阵 B 的 ND 布局 |
| 3 | `dotC_ND` | 矩阵 C 的 ND 布局 |
| 4 | `nZ` | nZ 格式（列优先分块） |
| 5 | `zN` | zN 格式（行优先分块） |
| 6 | `ND` | 标准行优先 |
| 7 | `Fractal` | Fractal 格式 |

布局与硬件的对应关系另见 [data_layout](../architecture/data_layout.md)。

## AddressSpace（7 值）

zero(0) 默认 / **gm(1)** 全局内存 / **cbuf(2)** L1 / **ca(3)** L0A / **cb(4)** L0B / **cc(5)** L0C 累加器 / **ub(6)** 统一缓冲区。存储层次见 [memory_hierarchy](../architecture/memory_hierarchy.md)。

## Pipe（15 值）

| 值 | 枚举 | 说明 |
|---|------|------|
| 0 | PIPE_S | Scalar Pipe |
| 1 | PIPE_V | Vector Pipe |
| 2 | PIPE_M | Cube 矩阵 Pipe |
| 3 | PIPE_MTE1 | L1 搬运 |
| 4 | PIPE_MTE2 | GM→片内搬运 |
| 5 | PIPE_MTE3 | 片内→GM 搬运 |
| 6 | PIPE_ALL | 所有 Pipe |
| 7/8 | PIPE_MTE4 / PIPE_MTE5 | MTE 4/5 |
| 9 | PIPE_V2 | Vector Pipe 2 |
| 10 | PIPE_FIX | Fixpipe（L0C→UB/GM） |
| 11/12 | VIRTUAL_PIPE_MTE2_L1A / L1B | 虚拟 Pipe：MTE2 到 L1A/L1B |
| 13 | PIPE_NUM | Pipe 数量标记 |
| 99 | PIPE_UNASSIGNED | 未分配 |

Pipe 并行执行模型见 [pipeline_execution_model](../architecture/pipeline_execution_model.md)。

## Core 类型三枚举

- **TCoreType**（操作级）：CUBE(1) / VECTOR(2) / CUBE_OR_VECTOR(3) / CUBE_AND_VECTOR(4)。
- **TFuncCoreType**（函数级）：AIC(1) Cube kernel / AIV(2) Vector kernel / MIX(3) 混合 / AIC_OR_AIV(4)。
- **TModuleCoreType**（模块级）：AIC(1) / AIV(2) / MIX(3)，由函数组成推断（全 AIC→AIC，全 AIV→AIV，否则 MIX）。

## 同步相关枚举

- **Event**（8 值）：EVENT_ID0(0) ~ EVENT_ID7(7)，每 Pipe 8 个 Event ID（用法见 [sync_pipe_sync.md](sync_pipe_sync.md)）。
- **UnitFlag**（4 值）：DISABLED(0) / RESERVED(1) / ENABLED_WITHOUT_UPDATE(2) / ENABLED_WITH_UPDATE(3)（用法见 [sync_unit_flag.md](sync_unit_flag.md)）。
- **SyncBlockMode**（6 值）：ALL_CUBE(0) / ALL_VECTOR(1) / ALL_SUB_VECTOR(2) / BARRIER_CUBE(3) / BARRIER_VECTOR(4) / ALL(5)（用法见 [sync_block_sync.md](sync_block_sync.md)）。
- **SyncBlockInstrMode**（3 值）：INTER_BLOCK(0) / INTER_SUBBLOCK(1) / INTRA_BLOCK(2，默认)。

## 计算语义枚举

- **ReduceOperation**（12 值）：none(0) / sum(1) / prod(2) / max(3) / min(4) / max_with_index(5) / min_with_index(6) / any(7) / all(8) / xori(9) / ori(10) / andi(11)（vreduce 用法见 [vector_reduction_ops.md](vector_reduction_ops.md)）。
- **CompareMode**（6 值）：eq(0) / ne(1) / lt(2) / gt(3) / ge(4) / le(5)。
- **RoundMode**（7 值）：rint(0) ties-to-even / round(1) ties-away / floor(2) / ceil(3) / trunc(4) / odd(5) Von Neumann / truncwithoverflow(6)。
- **UnsignedMode**（4 值）：si2si(0) / si2ui(1) / ui2si(2) / ui2ui(3)。
- **TypeFn/Cast**（3 值）：cast_signed(0) / cast_unsigned(1) / bitcast(2)。
- **AtomicKind**（9 值）：none(0) / add(1) / max(2) / min(3) / and(4) / or(5) / xor(6) / CAS(7) / XCHG(8)（原子 DMA 见 [dma_atomic.md](dma_atomic.md)）。

## Fixpipe 随路枚举（用法见 [dma_fixpipe.md](dma_fixpipe.md)）

- **FixpipePreQuantMode**（5 值）：NO_QUANT(0) / F322F16(1) / S322I8(9) / QF322F32_PRE(15) / F322BF16(16)。
- **FixpipePreReluMode**（4 值）：NO_RELU(0) / NORMAL_RELU(1) / LEAKY_RELU(2) / P_RELU(3)。
- **FixpipeDualDstMode**（3 值）：NO_DUAL(0) / ROW_SPLIT(1，按 M 拆两 UB) / COLUMN_SPLIT(2，按 N 拆)。
- **FixpipeDMAMode**（3 值）：NZ2ND(0) / NZ2DN(1) / NZ2NZ(2，字符串 `normal`)。

## Matmul 相关枚举

- **DescaleMode**（3 值）：DescaleNull(0) / DescalePerChannel(1，形状 `[n]`) / DescalePerTensor(2，形状 `[1]`)（用法见 [macro_matmul.md](macro_matmul.md)）。
- **MatmulBiasMode**（5 值）：NoBias(0) / PerChannelAdd(1) / PostPerChannelAddWithSplitK(2) / ElementwiseAdd(3) / MMInitPerChannelAddWithSplitK(4)。

## 其他枚举

- **PadMode**（3 值）：PadNull(0) / PadFirstElem(1) / PadValue(2)。
- **EvictionPolicy**（2 值）：EvictFirst(0) / EvictLast(1)。
- **AlignKind**（3 值）：align(0) / unaligned(1) / unknown(2)。
- **AxisKind**（3 值）：first(0) / middle(1) / last(2)。
- **VFMode**（3 值）：SIMD(0) 向量化 / SIMT(1) 多线程 / MIX(2) 混合（SIMT 触发标量降级风险见 [vector_scalar_lowering.md](vector_scalar_lowering.md)）。
- **DataCacheKind**（4 值）：all(0) / ub(1) / out(2) / atomic(3)。
- **DCCIMode**（2 值）：single_cache_line(0) / all_cache_lines(1)。
- **DeinterleaveMode**（3 值）：CHANNEL_0(0) / CHANNEL_1(1) / ALL_CHANNELS(999)。
- **MemoryEffect**（3 值）：read(0) / write(1) / read_write(2)（SIMT VF 内存效果标记）。

相关：[attr_overview.md](attr_overview.md)、[attr_parameterized_attrs.md](attr_parameterized_attrs.md)、[attribute_enum_index](../reference/attribute_enum_index.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMAttrs.td#L51-L78](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L51-L78)
- [HIVMAttrs.td#L84-L101](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L84-L101)
- [HIVMAttrs.td#L171-L188](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L188)
- [HIVMAttrs.td#L203-L236](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L236)
- [HIVMAttrs.td#L298-L309](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L309)
- [HIVMAttrs.td#L250-L261](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L250-L261)
- [HIVMAttrs.td#L271-L276](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L271-L276)
- [HIVMAttrs.td#L330-L341](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L330-L341)
- [HIVMAttrs.td#L356-L364](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L356-L364)
- [HIVMAttrs.td#L378-L406](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L378-L406)
- [HIVMAttrs.td#L408-L425](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L408-L425)
- [HIVMAttrs.td#L431-L446](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L431-L446)
- [HIVMAttrs.td#L452-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L452-L473)
- [HIVMAttrs.td#L479-L498](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L479-L498)
- [HIVMAttrs.td#L512-L523](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L523)
- [HIVMAttrs.td#L540-L555](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L540-L555)
- [HIVMAttrs.td#L569-L578](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L569-L578)
- [HIVMAttrs.td#L596-L623](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L596-L623)
- [HIVMAttrs.td#L637-L658](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L637-L658)
- [HIVMAttrs.td#L670-L678](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L670-L678)
- [HIVMAttrs.td#L688-L696](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L688-L696)
- [HIVMAttrs.td#L948-L954](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L948-L954)
- [HIVMAttrs.td#L736-L747](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L736-L747)
- [HIVMAttrs.td#L753-L763](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L753-L763)
- [HIVMAttrs.td#L783-L796](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L796)
- [HIVMAttrs.td#L803-L814](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L803-L814)
- [HIVMAttrs.td#L821-L830](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L821-L830)
- [HIVMAttrs.td#L847-L854](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L847-L854)
- [HIVMAttrs.td#L865-L874](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L865-L874)
- [HIVMAttrs.td#L886-L895](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L886-L895)
- [HIVMAttrs.td#L903-L914](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L903-L914)
- [HIVMAttrs.td#L1046-L1055](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1046-L1055)
- [attribute.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/attribute.mlir)
