---
type: API
title: 属性枚举速查索引（HIVM / HACC / HFusion 枚举名 → 值列表）
description: AscendNPU-IR 各方言枚举属性的速查表：HIVM 的 AddressSpace/DataLayout/PIPE/核心类型/同步/归约/Fixpipe
  等枚举、HACC 的 DeviceSpec/KernelArgType/HostFuncType、HFusion 的 UnaryFn/BinaryFn/CompareFn
  等，供解析 IR 属性时对照。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']

---

# 属性枚举速查索引

按方言分组的枚举名 → 值列表速查，供阅读 IR 属性（如 `#hivm.pipe<PIPE_V>`、`pre_quant = F322BF16`）时对照。各枚举的语义详解链接见 resource 上游文档。

## HIVM 方言枚举

### 存储与布局

| 枚举 | 值列表 | 详卡 |
|------|--------|------|
| AddressSpace | Zero(0), GM(1), L1(2), L0A(3), L0B(4), L0C(5), UB(6) | [memory_hierarchy](../architecture/memory_hierarchy.md) |
| DataLayout | dotA_ND(1), dotB_ND(2), dotC_ND(3), nZ(4), zN(5), ND(6), Fractal(7) | [data_layout](../architecture/data_layout.md) |

### Pipeline 与核心类型

| 枚举 | 值列表 | 详卡 |
|------|--------|------|
| PIPE | PIPE_S(0), PIPE_V(1), PIPE_M(2), PIPE_MTE1(3), PIPE_MTE2(4), PIPE_MTE3(5), PIPE_ALL(6), PIPE_MTE4(7), PIPE_MTE5(8), PIPE_V2(9), PIPE_FIX(10), VIRTUAL_PIPE_MTE2_L1A(11), VIRTUAL_PIPE_MTE2_L1B(12), PIPE_NUM(13), PIPE_UNASSIGNED(99) | [pipeline_execution_model](../architecture/pipeline_execution_model.md) |
| TCoreType | CUBE(1), VECTOR(2), CUBE_OR_VECTOR(3), CUBE_AND_VECTOR(4) | [npu_hardware_overview](../architecture/npu_hardware_overview.md) |
| TFuncCoreType | AIC(1), AIV(2), MIX(3), AIC_OR_AIV(4) | 同上 |
| TModuleCoreType | AIC(1), AIV(2), MIX(3) | 同上 |
| VFMode | SIMD(0), SIMT(1), MIX(2) | 同上 |

### 同步

| 枚举 | 值列表 |
|------|--------|
| EVENT | EVENT_ID0(0) ~ EVENT_ID7(7)，共 8 个事件 |
| UNIT_FLAG | DISABLED(0), RESERVED(1), ENABLED_WITHOUT_UPDATE(2), ENABLED_WITH_UPDATE(3) |
| SyncBlockMode | ALL_CUBE(0), ALL_VECTOR(1), ALL_SUB_VECTOR(2), BARRIER_CUBE(3), BARRIER_VECTOR(4), ALL(5) |
| SyncBlockInstrMode | INTER_BLOCK_SYNCHRONIZATION(0), INTER_SUBBLOCK_SYNCHRONIZATION(1), INTRA_BLOCK_SYNCHRONIZATION(2) |

### 计算语义

| 枚举 | 值列表 |
|------|--------|
| ReduceOperation | none(0), sum(1), prod(2), max(3), min(4), max_with_index(5), min_with_index(6), any(7), all(8), xori(9), ori(10), andi(11) |
| CompareMode | EQ(0), NE(1), LT(2), GT(3), GE(4), LE(5) |
| RoundMode | RINT(0), ROUND(1), FLOOR(2), CEIL(3), TRUNC(4), ODD(5), TRUNCWITHOVERFLOW(6) |
| UnsignedMode | SI2SI(0), SI2UI(1), UI2SI(2), UI2UI(3) |
| TypeFn | cast_signed(0), cast_unsigned(1), bitcast(2) |
| IteratorType | kParallel(0), kBroadcast(1), kTranspose(2), kReduction(3), kInterleave(4), kDeinterleave(5), kInverse(6), kPad(7), kConcat(8), kGather(9), kCumulative(10), kOpaque(99) |
| AtomicKind | NONE(0), ADD(1), MAX(2), MIN(3), AND(4), OR(5), XOR(6), CAS(7), XCHG(8) |
| AlignKind | ALIGN(0), UNALIGNED(1), UNKNOWN(2) |
| AxisKind | FIRST(0), MIDDLE(1), LAST(2) |

### 访存与 Fixpipe 随路操作

| 枚举 | 值列表 |
|------|--------|
| PadMode | PadNull(0), PadFirstElem(1), PadValue(2) |
| PaddingOption | PAD_ZERO(1), PAD_NAN(2) |
| EvictionPolicy | EvictFirst(0), EvictLast(1) |
| CacheModifier | NONE(1), CA(2), CG(3), WB(4), CS(5), WT(6), CV(7) |
| DataCacheKind | ALL(0), UB(1), OUT(2), ATOMIC(3) |
| DCCIMode | SINGLE_CACHE_LINE(0), ALL_CACHE_LINES(1) |
| MemoryEffect | READ(0), WRITE(1), READ_WRITE(2) |
| FixpipePreQuantMode | NO_QUANT(0), F322F16(1), S322I8(9), QF322F32_PRE(15), F322BF16(16) |
| FixpipePreReluMode | NO_RELU(0), NORMAL_RELU(1), LEAKY_RELU(2), P_RELU(3) |
| FixpipeDualDstMode | NO_DUAL(0), ROW_SPLIT(1), COLUMN_SPLIT(2)（仅 950） |
| FixpipeDMAMode | NZ2ND(0), NZ2DN(1), NZ2NZ(2)（NZ2DN 仅 950） |
| DeinterleaveMode | CHANNEL_0(0), CHANNEL_1(1), ALL_CHANNELS(999) |
| DescaleMode | DescaleNull(0), DescalePerChannel(1), DescalePerTensor(2) |
| MatmulBiasMode | NoBias(0), PerChannelAdd(1), PostPerChannelAddWithSplitK(2), ElementwiseAdd(3), MMInitPerChannelAddWithSplitK(4) |

### 内存规划

| 枚举 | 值列表 | 详卡 |
|------|--------|------|
| MemPlanMode | LOCAL_MEM_PLAN(0), GLOBAL_WORKSPACE_PLAN(1) | [memory_planning](../memory_management/memory_planning.md) |

## HACC 方言枚举

| 枚举 | 值列表 |
|------|--------|
| HACCFuncType | HOST(1), DEVICE(2) |
| HostFuncType | kEntry(1), kTilingFunction(2), kInferOutputShapeFunction(3), kInferWorkspaceShapeFunction(4), kInferSyncBlockLockNumFunction(5), kInferSyncBlockLockInitFunction(6), kInferVFModeFunction(7), kGetTilingStructSizeFunction(8) |
| KernelArgType | kFFTSBaseAddr(0), kInput(1), kOutput(2), kInputAndOutput(3), kWorkspace(4), kSyncBlockLock(5), kTilingKey(6), kTilingData(7), kTilingStruct(8), kMeshArg(9), kSanitizerAddr(10), kGMAddr(11) |
| DeviceSpec | AI_CORE_COUNT(0), CUBE_CORE_COUNT(1), VECTOR_CORE_COUNT(2), UB_SIZE(3), L1_SIZE(4), L0A_SIZE(5), L0B_SIZE(6), L0C_SIZE(7), UB_ALIGN_SIZE(8), L1_ALIGN_SIZE(9), L0C_ALIGN_SIZE(10), MINIMAL_D_CACHE_SIZE(11), MAXIMUM_D_CACHE_SIZE(12), ARCH(13)——字段取值见 [hardware_specs_ir.md](hardware_specs_ir.md) |

## HFusion 方言枚举

| 枚举 | 值列表 |
|------|--------|
| UnaryFn | relu(0), sqrt(1), rsqrt(2), rec(3), vnot(4), tanh(5), sin(6), cos(7), atan(8), tan(9), absi(10), erf(11), log2(12), log10(13), log1p(14), exp2(15), expm1(16), ilogb(17) |
| BinaryFn | vor(0), vand(1), vxor(2), minf(3), maxf(4), powf(5), mod(6), shli(7), shrsi(8), shrui(9), ldexp(10), ceildivsi(11), ceildivui(12), floordivsi(13), powi(14), minnumf(15), maxnumf(16), modui(17), divfhp(18) |
| CompareFn | veq(0), vne(1), vle(2), vlt(3), vge(4), vgt(5), vule(6), vuge(7), vugt(8), vult(9) |
| TernaryFn | select(0) |
| TypeFn / RoundMode / UnsignedMode / PaddingOption / CacheModifier | 与 HIVM 同名枚举取值一致 |
| EvictionPolicy | NORMAL(1), EVICT_FIRST(2), EVICT_LAST(3)（注意与 HIVM 版取值不同） |
| FusionKind / ReduceWithIndexKind | 由 HFusion TableGen 定义，见上游 |

相关：op 级速查见 [ir_operation_index.md](ir_operation_index.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td)
- [HFusionEnums.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td)
- [HFusionAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td)
