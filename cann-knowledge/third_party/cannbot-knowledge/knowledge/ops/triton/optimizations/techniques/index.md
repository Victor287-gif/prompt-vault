# 优化技术

跨算子通用的优化技术专题。每项技术覆盖核心方法论、适用条件、代码模式与常见陷阱。

## 总览与方法论

- [perf_optimization_overview](perf_optimization_overview.md)：优化全局视图：计算/访存密集判定、策略矩阵、vec/cube/scalar ratio 指标
- [performance_tuning](performance_tuning.md)：性能调优方法论：autotune 搜索空间、num_warps/num_stages
- [advanced_optimization](advanced_optimization.md)：高级优化合集：profiling 瓶颈判读、仿真流水图诊断、编译期优化与检查清单

## Tiling 与并行

- [tiling](tiling.md)：三级 Tiling 层次（ncore/xblock/xblock_sub）、UB 容量约束、块大小选择
- [tiling_strategy](tiling_strategy.md)：Tiling 完整规范：32B/512B 对齐、存算并行条件、一级/二级分块计算与 UB 估算
- [tiling_and_grid_910_95](tiling_and_grid_910_95.md)：910_95/910B Tiling 与 Grid 分核：UB 预算公式、Grid 收缩、单行改多行
- [spmd_on_npu](spmd_on_npu.md)：SPMD 从 GPU Thread Block 到 NPU AI Core 的映射、coreDim 65535 上限规避
- [grid_and_program_id](grid_and_program_id.md)：program_id/num_programs 语义、Grid 合并 1D 机制与核数对齐
- [core_mapping](core_mapping.md)：Core 映射策略（列优先/行优先/对角线）、GROUP_SIZE、L2 局部性
- [persistent_kernels](persistent_kernels.md)：Persistent Kernel 与 grid-stride loop
- [parallelism_and_loop](parallelism_and_loop.md)：并行化与循环优化：grid-stride、multi-round、LICM

## 数据类型与数值

- [data_types](data_types.md)：数据类型优化：Scalar Fallback 诊断、int64→fp32、混合精度
- [data_types_ascend](data_types_ascend.md)：NPU dtype 支持矩阵、tl.dot/tl.load 类型约束、隐式提升规则与 FP8 现状
- [data_type_precision](data_type_precision.md)：910_95 平台 dtype 速查、归约升精度、matmul 混合精度与精度验证参数

## 访存与数据搬运

- [memory_access_optimization](memory_access_optimization.md)：内存访问优化：Block Pointer、转置 Load、寄存器复用
- [data_movement_optimization](data_movement_optimization.md)：数据通路（MTE1/2/3/FIX）特征、MultiBuffer 双缓冲、for 循环 Tiling
- [care_padding](care_padding.md)：care_padding：padding 填充行为、MTE2/Vector 并行度影响与禁忌场景
- [store_merge](store_merge.md)：同基地址连续 tl.store 合并，减少 MTE3 搬运次数
- [bank_conflict](bank_conflict.md)：Bank Conflict 消除：对角线遍历、padding

## 计算与流水线

- [ilp](ilp.md)：指令级并行：care_padding、异步任务、流水线展开
- [compute_and_branch](compute_and_branch.md)：计算与分支优化：constexpr 特化、tl.where、tl.cdiv
- [operator_fusion](operator_fusion.md)：算子融合模式与决策树
- [online_algorithms](online_algorithms.md)：在线算法：online softmax、running mean/variance
- [cv_pipeline_optimization](cv_pipeline_optimization.md)：CV 流水线：指令重排序、编译参数组合、fixpipe 数据通路与精度回退
- [fixpipe_bias_fusion](fixpipe_bias_fusion.md)：FixPipe 随路操作与 bias 融合：Cube 侧融合、随路转置/量化/激活
- [sync_and_barrier](sync_and_barrier.md)：三层同步机制：set_flag/wait_flag/pipe_barrier、sync_block_*、UnitFlag
- [scalar_degradation_avoidance](scalar_degradation_avoidance.md)：标量降级（Scalar Lowering）触发条件与规避模式，910_95 Reg-based vs 910B Mem-based
- [where_optimization](where_optimization.md)：单位置 False 的 tl.where 退化规避：get_element + insert_slice

## 编译参数与扩展

- [compile_params](compile_params.md)：NPUOptions 编译参数全表、按算子类型配置模板与 910_95 差异
- [compile_hint_extension](compile_hint_extension.md)：compile_hint 与 Ascend 扩展 API 速查（slice/PropagateNan/sub_vec/custom）
- [autotune_on_npu](autotune_on_npu.md)：NPU autotune 与 GPU 差异（Config 仅 Tiling 参数）、AutoTilingTuner 与候选空间
