---
type: API
title: IR 操作速查索引（按方言分组的 op 名 → 功能速查）
description: AscendNPU-IR 各方言操作的按组速查：HIVM（DMA/间接访问/向量/Macro/同步/其他）、HFusion、HACC、Scope/Symbol/MemRefExt/MathExt/HMAP/Annotation、AscendDPX、Triton/TritonGPU/Gluon，用于
  last_pass.mlir 阅读时快速定位 op 语义。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# IR 操作速查索引

按方言分组的 op → 一句话功能速查。每组只保留检索价值最高的条目；完整逐条清单（含各 op 专属文档链接）见 resource 上游文档。

## HIVM 方言

### DMA 操作

| 操作 | 功能 |
|------|------|
| `hivm.hir.load` | GM→UB 加载（MTE2），支持 padding/隐式转置 |
| `hivm.hir.store` | UB→GM 写回（MTE3），支持原子操作 |
| `hivm.hir.copy` | 地址空间间拷贝，Pipe 按源/目标动态推断 |
| `hivm.hir.fixpipe` | L0C→GM/L1/UB 输出（FIX），支持量化/激活/布局随路操作 |
| `hivm.hir.nd2nz` / `nz2nd` | GM↔L1 的 ND↔NZ 布局转换 |
| `hivm.hir.l12ub` | L1→UB（MTE1），NZ→ND |
| `hivm.hir.atomic_cas` / `atomic_xchg` | 原子比较交换 / 原子交换 |

### 间接访问操作

| 操作 | 功能 |
|------|------|
| `hivm.hir.gather_load` / `scatter_store` | 按索引聚集加载 / 散射写回 |
| `hivm.hir.local_load` / `local_store` | UB 与寄存器间数据转移（SIMT 模式） |
| `hivm.hir.indirect_load` / `indirect_store` | 间接寻址加载 / 写回 |
| `hivm.hir.gatherT` / `scatterT` | 张量级 gather/scatter |
| `hivm.hir.index_put` | 索引赋值 |
| `hivm.hir.embedding_gather` | embedding 查表聚集 |

### 向量操作（全部 PIPE_V / VECTOR 核心）

| 分组 | 操作 |
|------|------|
| 一元 | `vexp` `vabs` `vln` `vrelu` `vrsqrt` `vsqrt` `vtanh` `vsin` `vcos` `verf` `vrec` `vnot` |
| 二元 | `vadd` `vsub` `vmul` `vmulext` `vdiv` `vmax` `vmin` `vor` `vand` `vxor` `vmod` `vmodui` `vpow` |
| 移位 | `vshl` `vshr` |
| 比较/三元/转换/广播 | `vcmp` `vsel` `vcast` `vbrc` |
| 归约/数据移动 | `vreduce` `vtranspose` `vinterleave` `vdeinterleave` `vflip` `vmulextended` `vpad` `vconcat` `vgather` |
| 累积/排序 | `vcumsum` `vcumprod` `vsort` |

### Macro 操作

| 操作 | InPipe→OutPipe | 功能 |
|------|----------------|------|
| `hivm.hir.mmadL1` | MTE1→M | L1 驻留矩阵乘（Cube） |
| `hivm.hir.batchMmadL1` | MTE1→M | 批量矩阵乘 |
| `hivm.hir.matmul` | MTE2→MTE3 | 端到端矩阵乘宏操作 |
| `hivm.hir.mix_matmul` / `mix_group_matmul` | MTE2→MTE3 | Mix 场景矩阵乘 / 分组矩阵乘 |

### 同步操作

| 操作 | 功能 |
|------|------|
| `hivm.hir.set_flag` / `wait_flag` | 跨 Pipeline 事件通知 / 等待（配 EVENT_ID0-7） |
| `hivm.hir.pipe_barrier` | 单 Pipeline 内屏障 |
| `hivm.hir.sync_block` / `sync_block_set` / `sync_block_wait` | 跨 AI Core 块同步 |

### 其他 HIVM 操作

| 操作 | 功能 |
|------|------|
| `hivm.hir.get_block_idx/num` `get_sub_block_idx/num` | 核/子核索引与数量查询 |
| `hivm.hir.pointer_cast` | PlanMemory 后 alloc 的物理偏移表示（见 [memory_planning](../memory_management/memory_planning.md)） |
| `hivm.hir.set_atomic` / `set_mask_norm` / `set_ctrl` | 原子模式 / mask / 控制寄存器设置 |
| `hivm.hir.load_scalar` | 标量加载 |
| `hivm.hir.dcci` | DCache 控制（cache 行维护） |
| `hivm.hir.convert_layout` | 布局转换 |
| `hivm.hir.bitcast` | 位级类型转换 |
| `hivm.hir.set_ffts_base_addr` | 设置 FFTS 基址（Mem-based 核间同步） |
| `hivm.hir.debug` / `init_debug` / `finish_debug` | 调试埋点 |
| `hivm.hir.custom` / `custom_macro` | 自定义 op / 自定义宏 op |

## HFusion 方言（高层可融合 IR）

| 分组 | 操作 |
|------|------|
| 特殊/调试 | `hfusion.print` `assert` `barrier` `symbolic_dim` `sort` `arange` `histogram` |
| 逐元素 | `hfusion.mulext` `isinf` `isnan` `isfinite` |
| 数据移动 | `hfusion.interleave` `deinterleave` `flip` `cumsum` `cumprod` `embedding_gather` `gather_load` `scatter_store` `indirect_load/store` `gatherT` `scatterT` `index_put` |
| 内存/原子 | `hfusion.atomic_cas` `atomic_xchg` |
| 矩阵/归约 | `hfusion.matmul_mx`（MX 量化矩阵乘）、`hfusion.reduce_with_index` |

## HACC 方言

不定义独立操作，通过属性标注 `func.func` 与 `ModuleOp`（`hacc.DeviceSpec`、函数类型、kernel 参数类型等），见 [attribute_enum_index.md](attribute_enum_index.md) 与 [hardware_specs_ir.md](hardware_specs_ir.md)。

## 其他小方言

| 方言 | 操作 | 功能 |
|------|------|------|
| Scope | `scope.scope` / `scope.return` | 局部变量作用域管理 |
| Symbol | `symbol.symbolic_int` / `bind_symbolic_shape` | 符号化维度，支持动态形状 |
| MemRefExt | `memref_ext.alloc_workspace` | 全局工作空间分配（见 [workspace_management](../memory_management/workspace_management.md)） |
| MathExt | `math_ext.ilogb` / `ldep` / `divfhp` | 数学扩展（取指数/ldexp/高精度除法） |
| HMAP | `hmap.all_to_all_v` | 多处理器可变长 all-to-all 通信 |
| Annotation | `annotation.mark` | 编译器标注载体（multi_buffer/stride_align 等属性都挂它上面） |

## AscendDPX 方言（SIMT 设备 op，仅 Reg-based 架构）

| 分组 | 操作 |
|------|------|
| 线程/块索引 | `thread_id_x/y/z` `block_idx[_x/y/z]` `block_dim_*` `grid_dim_*` `core_id` |
| 时钟 | `clock32` `clock64` |
| 内存/同步 | `load` `store` `sync_threads` |
| warp 洗牌 | `shfl.up/down/bfly/idx` |
| 原子 | `atomic_add/sub/and/or/xor/max/min/umax/umin/inc/dec/exchange/cas` |
| 数学 | `exp/exp2/log/log2/log1p/sin/cos/tan/atan/sqrt/rsqrt/recip/relu/floor/ceil/rint/round/div/udiv/pow/umulhi/ldexp/ilogb/cast` |
| 判断 | `isfinite` `isinf` `isnan` |

## Triton / TritonGPU / Gluon 方言

| 方言 | 分组与代表操作 |
|------|----------------|
| Triton（前端） | 指针/类型：`tt.int_to_ptr` `ptr_to_int` `bitcast` `fp_to_fp` `addptr`；访存：`tt.load` `store` `atomic_rmw/cas`；张量：`splat` `expand_dims` `reshape` `broadcast` `cat` `join` `split` `trans` `make_range`；计算：`tt.dot` `dot_scaled` `clampf` `precise_sqrt/divf` `mulhiui` `extern_elementwise` `histogram` `gather`；描述符：`make_tensor_ptr/descriptor` `descriptor_load/store/reduce/gather/scatter`；结构：`tt.func` `call` `return` `get_program_id/num_programs` `print` `assert` |
| TritonGPU（后端） | 布局：`tritongpu.convert_layout`；异步：`async_copy_global_to_local` `async_commit_group` `async_wait`；局部内存：`local_alloc/dealloc` `local_load/store` `memdesc_index/subslice/trans/reshape/reinterpret`；其他：`fp4_to_fp` `global_scratch_alloc` `warp_specialize[.partitions]` `warp_yield` `warp_return` |
| Gluon | `gluon.set_auto_layout`（布局自动推导） |

相关：各 op 绑定的 Pipeline 见 [pipeline_execution_model](../architecture/pipeline_execution_model.md)；枚举属性值见 [attribute_enum_index.md](attribute_enum_index.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [IR](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/)
- [IR](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/IR/)
- [TritonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Triton/IR/TritonOps.td)
- [TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)
- [GluonOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/Gluon/IR/GluonOps.td)
- [AscendDPXOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/AscendDPX/IR/AscendDPXOps.td)
