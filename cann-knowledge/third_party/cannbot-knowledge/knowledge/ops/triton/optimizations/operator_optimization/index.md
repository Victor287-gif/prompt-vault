# 优化期改写（stage: optimize）

手上**已有可跑代码**时适用，由 latency-optimizer 按触发条件判定后施加。
与设计期约束（`../../guides/constraints/index.md`）适用前提互斥，不可混用。

- [`OPT-1` Ascend Interpolate 算子系统性优化](ascend_interpolate_optimization.md)：本文档覆盖 Ascend NPU 上 Triton Interpolate 算子（torch.nn.functional.interpolate，4D NCHW）。
- [`OPT-FA1` FA 形态 Phase 4 优化点清单](fa_phase4_optimization_checklist.md)：Flash Attention 在性能优化阶段的候选优化点清单与优先级。
- [`OPT-2` Ascend Pooling 算子 Triton 系统性优化指南](ascend_pooling_optimization.md)：本指南为 Ascend NPU 上 Pooling 算子（AvgPool/MaxPool，2D/3D）的 Triton 实现提供完整的、可复现的优化方法论。
- [`OPT-3` Autotune 自动调优](autotune.md)：Triton autotune 用于自动选择最优的 kernel 配置参数，主要包括影响分核（split）和切块（tiling）大小的参数，主要使用方式如下：。
- [`OPT-4` 标量降级](avoid_scalar_lowering.md)：标量降级是指：编译器把本该一次处理多个数据的向量操作，降级成逐个元素执行的标量循环。
- [`OPT-5` Block Size Scaling 优化模式](block_size_scaling.md)：在所有指令级优化策略（constexpr 静态化、int32 向量加法、load 重排、向量比较）应用完毕后，通过倍增 BLOCK_SIZE 来探索更大分块带来的性能收益。
- [`OPT-6` Matmul 链中间 buffer dtype 优化](chained_matmul_buffer_dtype.md)：在由多个 tl.dot 串联而成的算子中，中间 buffer 的 dtype 会直接影响下一段 matmul 能否以 Cube 峰值吞吐执行。
- [`OPT-7` 入参静态化 优化模式](constexpr_parameters.md)：在 Triton NPU kernel 中，将固定数值的入参声明为 tl.constexpr，可以让编译器在编译时进行更多的常量折叠和常量传播优化，从而提升 kernel 的执行效率。
- [`OPT-8` Cube/MTE3 分阶段批量解耦优化](cube_mte3_decoupling.md)：更糟的是：当某个归约维（如 head 维 N1）被切成多个 program 并行处理时，该维的归约只能靠多 program 对同一输出地址的 atomic 竞争完成——num_g_groups 个 program 各自发 atomic_add 打到同一 token 位置，RMW 被锁串行仲裁，ato
- [`OPT-9` Device-side Gather 连续化](device_side_gather.md)：for blk_start in range(0, N, BLOCK_K):。
- [`OPT-10` 维度合并与大 BLOCK 累加（归一化算子专用）](dimension_merge_large_block.md)：for c in range(c_start, c_end):。
- [`OPT-11` 维度合并优化](dimension_merge.md)：for n in range(N):           # 64 次。
- [`OPT-12` 离散访存优化模式](discrete_memory_access.md)：在 Triton NPU kernel 中，当线程通过非连续或不可预测的索引向量访问全局内存时，会导致访存效率低下，显著降低带宽利用率。
- [`OPT-13` Grid 形状与多路径特化](grid_dispatch_specialization.md)：单一 kernel 实现无法在不同 workload 规模下同时达到最优性能。
- [`OPT-14` Histogram-like / Small-Output-Table Reduction 优化](histogram_like_table_reduction.md)：当算子满足以下全部特征时，归为本类别：。
- [`OPT-15` Host 侧张量维度拼接优化（Host-Side Tensor Concat）](host_tensor_concat.md)：scores  = tl.dot(a_nope, tl.trans(c_nope))     # cube dot #1, 维度 D。
- [`OPT-16` Kernel 分裂优化](kernel_splitting.md)：多 Case 场景下，单个泛用 Kernel 往往无法在所有 Shape/Dtype 特征下都取得最优性能。
- [`OPT-17` Latency-Bound 循环维度 Tile 合并优化](latency_bound_tile_merge.md)：for hc in range(NUM_HC):                 # 8 次外层迭代。
- [`OPT-18` Libdevice 函数使用](libdevice_usage.md)：Triton-Ascend libdevice 提供了一系列已优化的基础函数库，包括数学函数、三角函数、双曲函数、指数对数函数、激活函数、特殊函数等。
- [`OPT-19` Load 指令重排序](load_order.md)：该文件旨在说明，编译器不会修改用户load指令的顺序，在前面的load指令被其他指令阻塞时，可以将没有数据依赖的load语句放在前面以提前发射，提升并行度。
- [`OPT-20` 循环不变量外提优化](loop_invariant_hoisting.md)：2. 内层循环中有 tl.load(param_ptr + index_expr)。
- [`OPT-21` Ascend Matmul Transpose 优化](matmul_transpose.md)：本文档覆盖 Ascend NPU 上 Triton 矩阵乘法转置类算子（MatmulBothTrans / MatmulTransA / MatmulTransB / BMM / Linear）的系统性性能优化方法。
- [`OPT-22` 混合策略自动选择](mixed_strategy.md)：同一算子在不同 shape 或数据类型下可能需要不同的优化策略。
- [`OPT-23` Multibuffer / 双缓冲优化（通信-计算重叠）](multibuffer_and_double_buffering.md)：Triton kernel 的典型执行流是：。
- [`OPT-24` 减少遍历优化](pass_merge.md)：减少数据遍历次数是 Triton 性能优化的核心手段。
- [`OPT-25` 输出预初始化优化（Output Preinitialization）](preinitialized_output_optimization.md)：若输出张量有大量位置需写入同一默认值（常为 0），不要在 kernel 内逐位置判断填充。
- [`OPT-26` 消除冗余的边界运算](redundant_boundary_operation.md)：冗余的边界运算是指：当 tl.load 已经通过 mask + other=d 在边界区域确定了常量值 d，后续任何显式将该区域重置为 d 的运算都是冗余的。
- [`OPT-27` Scalar 转 Vector 优化模式](scalar_to_vector.md)：在昇腾 NPU Triton kernel 开发中，scalar 操作会占用 Scalar 计算单元，无法充分利用 NPU 的 Vector 计算单元（支持 256 字节 SIMD 并行）。
- [`OPT-28` Tiling 优化](tiling_optimization.md)：在 NPU 架构中，内存带宽通常是性能瓶颈。
- [`OPT-29` 分核优化](vector_core_partition.md)：grid = (batch_size,)  # 如果 batch_size=128，远超 48 核。
- [`OPT-30` Workspace 物化解耦优化](workspace_decoupling.md)：for hc in range(HC_LOOP):。
- [`OPT-31` Adaptive Instance Normalization 2D Backward（AdaIN Bwd）优化经验](adain.md)：class ModelNew(nn.Module):。
- [`OPT-32` 连续拷贝聚合优化](continuous_copy_aggregation.md)：对于纯内存拷贝型算子（Split / Chunk / Slice / Pad / Unbind 等），当多个输出块在输入侧构成连续内存区域时，可以通过聚合拷贝操作，减少 kernel 启动次数和调度开销，显著提升性能。
- [`OPT-33` Ascend CV 融合算子生成方法论 — Batch 流水线篇](cv_fusion_pingpong.md)：CV 融合算子的本质是把计算拆成交替执行的 Vector 阶段 和 Cube 阶段，阶段之间通过 On-Chip buffer 握手。
- [`OPT-34` Ascend CV 融合算子生成方法论 — Tiling 篇](cv_fusion_tiling.md)：Tiling 是在 On-Chip 容量、DMA 吞吐、循环次数 之间找最优切分。
- [`OPT-35` Ascend CV 融合算子生成方法论 — CV 融合篇](cv_fusion.md)：CV 融合算子的本质是 Cube-Vector 异构核心间的数据流水线。
- [`OPT-36` Attention / FlashAttention 类算子优化点库（Triton-Ascend）](flash_attention_optimization.md)：FA 类算子在 Triton-Ascend 上的瓶颈判别、迭代数目标函数、有效方向与证伪方向全表。
- [`OPT-37` Triton-Ascend 通用优化洞察](general_insights.md)：在动手优化之前，先判断算子属于哪一类瓶颈：。
- [`OPT-38` Permute / Layout-transform 算子优化](permute_layout_transform.md)：适用于 permute、transpose、reshape（触发 contiguous copy）等仅改变数据布局、不改变元素值的算子。
- [`OPT-39` 激活函数 + 量化融合（Activation + Quantization Fusion）优化经验](swiglu_quant.md)：class ModelNew(nn.Module):。
- [`OPT-40` Triton-Ascend 验证与调试工作流](workflow_and_debugging.md)：当任务描述使用 get_input_groups() 时，.py 会通过 os.path.dirname(__file__) 读取同目录下的 {op_name}.json。
- [`OPT-41` grid 超 65535 的两种分派改写](grid_overflow_dispatch.md)：对于输入 shape 较大的算子，直接按照 BLOCK_SIZE 切分得到的 grid 总数可能超过 65535。
- [`OPT-42` 内存访问优化实践](memory_access_practice.md)：对于 2D 张量，使用 tl.make_block_ptr 可简化索引并提升访存效率：。
- [`OPT-43` 性能优化实践](perf_practice.md)：优先使用一维 grid (n,)，便于与 Ascend 核心映射一致。
- [`OPT-44` 调试检查清单](debug_checklist.md)：调试检查清单 的 Triton Ascend 编码要点。
- [`OPT-45` Element-wise 算子优化](coding_elementwise.md)：张量在内存中连续存储时，可用一维指针遍历，避免多维索引开销。
- [`OPT-46` Triton-Ascend Interpolate 算子开发参考文档](coding_interpolate.md)：inv_scale_h = float(torch.tensor(H_in / H_out, dtype=torch.float32))。
- [`OPT-47` MatMul 算子优化](coding_matmul.md)：以 fp16/bf16 为例（每个元素 2 字节）：。

---

## 与 SKILL.md 原优化点序号的对照

`triton-latency-optimizer/SKILL.md` 的路由表用序号 1–31 标识优化点，
本库使用全局唯一编号（`OPT-*` / `DC-*` / `GC`，见各条目链接文本）标识优化点。对照如下，供切换期回溯：

| SKILL.md 序号 | 原名称 | entry_id | 卡片 |
|---:|---|---|---|
| 1 | 入参静态化优化 | `OPT-7` | 入参静态化 优化模式 |
| 2 | Tiling 优化 | `OPT-28` | Tiling 优化 |
| 3 | 分核优化 | `OPT-29` | 分核优化 |
| 4 | 离散访存优化 | `OPT-12` | 离散访存优化模式 |
| 5 | Scalar 转 Vector 优化 | `OPT-27` | Scalar 转 Vector 优化模式 |
| 6 | 避免向量 API 标量降级 | `OPT-4` | 标量降级 |
| 7 | Pass 消除合并优化 | `OPT-24` | 减少遍历优化 |
| 8 | 维度合并优化 | `OPT-11` | 维度合并优化 |
| 9 | Libdevice 函数使用 | `OPT-18` | Libdevice 函数使用 |
| 10 | 循环不变量外提 | `OPT-20` | 循环不变量外提优化 |
| 11 | Load 指令重排序 | `OPT-19` | Load 指令重排序 |
| 12 | Grid 形状与多路径特化 | `OPT-13` | Grid 形状与多路径特化 |
| 13 | Autotune 自动调优 | `OPT-3` | Autotune 自动调优 |
| 14 | 混合策略自动选择 | `OPT-22` | 混合策略自动选择 |
| 15 | 维度合并与大 BLOCK 累加 | `OPT-10` | 维度合并与大 BLOCK 累加（归一化算子专用） |
| 16 | 连续拷贝聚合优化 | `OPT-32` | 连续拷贝聚合优化 |
| 17 | 消除冗余的边界运算 | `OPT-26` | 消除冗余的边界运算 |
| 18 | Kernel 分裂优化 | `OPT-16` | Kernel 分裂优化 |
| 19 | Cube/MTE3 分阶段批量解耦优化 | `OPT-8` | Cube/MTE3 分阶段批量解耦优化 |
| 20 | Host 侧张量维度拼接优化 | `OPT-15` | Host 侧张量维度拼接优化（Host-Side Tensor Concat） |
| 21 | Workspace 物化解耦优化 | `OPT-30` | Workspace 物化解耦优化 |
| 22 | Latency-Bound 循环维度 Tile 合并 | `OPT-17` | Latency-Bound 循环维度 Tile 合并优化 |
| 23 | Device-side Gather 连续化 | `OPT-9` | Device-side Gather 连续化 |
| 24 | Matmul 链中间 buffer dtype 优化 | `OPT-6` | Matmul 链中间 buffer dtype 优化 |
| 25 | 输出预初始化 | `OPT-25` | 输出预初始化优化（Output Preinitialization） |
| 26 | Ascend Interpolate 专用优化 | `OPT-1` | Ascend Interpolate 算子系统性优化 |
| 27 | Ascend Pooling 专用优化 | `OPT-2` | Ascend Pooling 算子 Triton 系统性优化指南 |
| 28 | Ascend Matmul Transpose 专用优化 | `OPT-21` | Ascend Matmul Transpose 优化 |
| 29 | CV 融合优化 | `OPT-35` | Ascend CV 融合算子生成方法论 — CV 融合篇 |
| 30 | Attention/FA 类算子专用优化 | `OPT-36` | Attention / FlashAttention 类算子优化点库（Triton-Ascend） |

> 未建立对照的原序号：31（对应文档按 SPEC §2.2/§2.3 未迁入）。
