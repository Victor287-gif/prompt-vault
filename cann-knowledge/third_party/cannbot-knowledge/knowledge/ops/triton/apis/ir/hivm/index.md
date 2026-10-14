# HIVM 方言

HIVM 是 AscendNPU-IR 的核心低层方言，直接映射 NPU 硬件操作。读 `last_pass.mlir` 时遇到的 `hivm.hir.*` 操作在此按族查阅。

## 总览

- [overview](overview.md)：方言定位、六大操作分类与 Address Space/PIPE/Core Type 三组核心枚举

## DMA 数据搬运

- [dma_overview](dma_overview.md)：DMA 操作族总览：数据通路、Pipeline 归属表与 copy 动态 Pipeline 规则
- [dma_load](dma_load.md)：`hir.load`：GM→UB 加载，pad_mode/eviction/init_out_buffer 属性
- [dma_store](dma_store.md)：`hir.store`：UB→GM 写回，atomic_kind 软硬件原子分类
- [dma_copy](dma_copy.md)：`hir.copy`：本地内存层级间拷贝，Pipeline 按地址空间动态确定
- [dma_fixpipe](dma_fixpipe.md)：`hir.fixpipe`：L0C→GM/UB/L1，随路布局转换/量化/激活/双目标
- [dma_nd2nz](dma_nd2nz.md)：`hir.nd2nz`：GM→L1 随路 ND→NZ 布局转换，Cube 数据通路入口
- [dma_nz2nd](dma_nz2nd.md)：`hir.nz2nd`：L1→GM 随路 NZ→ND 布局转换
- [dma_atomic](dma_atomic.md)：`hir.atomic_cas` / `hir.atomic_xchg`：GM 上硬件原子读-改-写
- [dma_gather_scatter](dma_gather_scatter.md)：`hir.gather_load` / `hir.scatter_store`：按索引非连续访存
- [dma_indirect_access](dma_indirect_access.md)：`hir.indirect_load` / `hir.indirect_store`：SIMT 模板按偏移间接访存
- [dma_padding_quantization](dma_padding_quantization.md)：DMA 随路功能汇总：PadMode/量化/激活/Eviction 与支持矩阵

## 向量计算

- [vector_overview](vector_overview.md)：向量操作族总览：HIVM_VectorOp 基类、OTF traits 与降级关系
- [vector_unary_ops](vector_unary_ops.md)：一元逐元操作 12 个：类型白名单、rank 上限与降级映射
- [vector_binary_ops](vector_binary_ops.md)：二元逐元操作 12 个：类型白名单、交换律与 Vector-Scalar 支持
- [vector_ternary_ops](vector_ternary_ops.md)：`hir.vsel` 条件选择与 vcmp→vsel 典型链
- [vector_cast_ops](vector_cast_ops.md)：`hir.vcast` 类型转换：round_mode 七值与转换支持矩阵
- [vector_compare_ops](vector_compare_ops.md)：`hir.vcmp` 比较：整数比较标量降级矩阵（仅 i32 EQ/NE 走向量路径）
- [vector_shift_ops](vector_shift_ops.md)：`hir.vshl` / `hir.vshr` 移位：仅整数类型、vshr round 属性
- [vector_reduction_ops](vector_reduction_ops.md)：`hir.vreduce` 归约 11 种类型与各芯片降级差异
- [vector_data_movement](vector_data_movement.md)：UB 内数据搬移：vbrc/vtranspose/vinterleave/vflip/vpad/vconcat/vgather
- [vector_cumulative_sort](vector_cumulative_sort.md)：`hir.vcumsum`/`vcumprod` 前缀和积与 `hir.vsort` 排序
- [vector_special_ops](vector_special_ops.md)：varange 序列生成、vmulextended/vmulext 乘扩展
- [vector_scalar_lowering](vector_scalar_lowering.md)：标量降级机制：五组触发条件、arith 映射与 10x-100x 损失规避（性能分析关键卡）

## 宏操作（矩阵乘）

- [macro_overview](macro_overview.md)：宏操作体系：本地 MMAD 与全局 MMAD 两大类、MacroOpTrait 与 UnitFlag 同步
- [macro_mmad_l1](macro_mmad_l1.md)：`hir.mmadL1`：L1→L0C 本地矩阵乘加，Split-K 与 HF32 加速
- [macro_batch_mmad_l1](macro_batch_mmad_l1.md)：`hir.batchMmadL1`：带批量维的本地矩阵乘加
- [macro_matmul](macro_matmul.md)：`hir.matmul`：GM→GM 全局矩阵乘，descale 反量化与 Swizzle 调度
- [macro_mix_matmul](macro_mix_matmul.md)：`hir.mix_matmul`：矩阵乘 + tile 级 Vector 后处理融合
- [macro_mix_group_matmul](macro_mix_group_matmul.md)：`hir.mix_group_matmul`：MoE 场景分组矩阵乘

## 同步控制

- [sync_overview](sync_overview.md)：三层同步体系：管道同步、跨核同步、UnitFlag 与自动注入
- [sync_pipe_sync](sync_pipe_sync.md)：`set_flag`/`wait_flag`/`pipe_barrier`：Event Flag 三元组配对与典型 Pipe 组合
- [sync_block_sync](sync_block_sync.md)：`sync_block` 系跨核同步：FFTS 机制、六种模式与锁原语
- [sync_unit_flag](sync_unit_flag.md)：UnitFlag 四模式：嵌入宏操作的条件同步
- [sync_sync_injection](sync_sync_injection.md)：自动同步注入：hivm-inject-sync 与 hivm-graph-sync-solver 对比

## Custom 扩展

- [custom_overview](custom_overview.md)：Custom 扩展机制：hir.custom 与 hir.custom_macro 两级粒度
- [custom_custom_op](custom_custom_op.md)：`hir.custom`：单 Pipeline 自定义扩展操作
- [custom_custom_macro_op](custom_custom_macro_op.md)：`hir.custom_macro`：跨 Pipeline 自定义宏操作

## 属性与类型系统

- [attr_overview](attr_overview.md)：属性类型系统五类组成全景索引
- [attr_enumerations](attr_enumerations.md)：30 个枚举属性取值速查（IteratorType/Pipe/AtomicKind/FixpipePreQuantMode 等）
- [attr_parameterized_attrs](attr_parameterized_attrs.md)：参数化属性：DataLayoutAttr/AddressSpaceAttr/PipeAttr/BlockMapping 等
- [attr_type_system](attr_type_system.md)：MemRef AddressSpace 标注、元素类型清单与类型转换路径
- [attr_interfaces](attr_interfaces.md)：OpInterface 清单：HIVMStructuredOpInterface 方法族等
- [attr_traits](attr_traits.md)：Trait 体系：Rank 系、ElementwiseNaryOpTrait、OTF traits 与依赖关系

## 内建指令

- [intrin_overview](intrin_overview.md)：最低层内建指令六类清单与高层操作映射表
