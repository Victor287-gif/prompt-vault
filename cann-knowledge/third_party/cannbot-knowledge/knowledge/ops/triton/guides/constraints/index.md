# 设计期约束（stage: design）

手上**还没有代码**时适用，由 designer 在草图阶段作为硬性边界消费。
与优化期改写（`../../optimizations/operator_optimization/index.md`）适用前提互斥，不可混用。
单算子实现手册见 operators/（`../operator_playbooks/index.md`）。

- [`AP-1` 反模式 — 在 forward 中查询设备属性](ap_device_query_in_forward.md)：import triton.runtime.driver as driver。
- [`DC-1` attention 类算子设计约束](attention_constraints.md)：attention 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-10` recurrent 类算子设计约束](recurrent_constraints.md)：recurrent 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-11` tensor_transform 类算子设计约束](tensor_transform_constraints.md)：tensor_transform 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-12` transformer_inference 类算子设计约束](transformer_inference_constraints.md)：transformer_inference 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-13` transformer_training 类算子设计约束](transformer_training_constraints.md)：transformer_training 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-14` 跨轴 3D Broadcast 优化案例](case_elemwise_broadcast_3d.md)：跨轴3D广播优化（最后一维很小）：采用两阶段kernel策略（先broadcast展开+reshape为2D，再标准多核处理）提升向量化效率，适用于跨轴broadcast且最后一维特别小（<20）导致向量化效果差的场景
- [`DC-15` Int8 到 FP16 类型转换优化案例](case_elemwise_cast.md)：大shape类型转换（int8→fp16）优化：通过二次切分（BLOCK_SIZE+TILE_SIZE）提高UB利用率，核数2048时性能最优，适用于shape较大（百万级元素）的elementwise类型转换场景
- [`DC-16` Slice + Concat 融合算子优化案例](case_elemwise_concat.md)：Slice+Concat融合算子优化：通过精确切片加载（只load需要部分）和索引计算拼接（避免cat指令）避免中间结果存储和多次内存访问，适用于多输入需要切片后拼接的融合算子场景
- [`DC-17` Zeros 创建张量优化案例](case_elemwise_zeros.md)：小shape张量创建（zeros/arange/full）优化：通过减少核数避免多核启动和调度开销，单核处理性能优于多核并行，适用于shape较小（千级元素）的elementwise创建张量场景
- [`DC-18` 小 bins 直方图 / 小输出表规约草图设计](case_histogram_small_bins.md)：小 bins 直方图 / 小输出表规约类算子的草图设计：per-core 局部表 + 二次归约，避免全局原子竞争，输入展平、fp32 索引、动态核数
- [`DC-19` Histogram 直方图统计优化案例](case_index_histogram.md)：直方图统计（histogram）优化：预排序+二分查找降低算法复杂度（O(n×m)→O(n log n + m log n)，性能提升19倍），转换为float32调用Vec Core硬件加速排序，适用于大规模统计类操作（50万+元素）
- [`DC-2` convolution 类算子设计约束](convolution_constraints.md)：convolution 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-20` Index Put 索引赋值优化案例](case_index_put.md)：索引赋值（index_put）优化：批量加载索引数据到UB后循环内通过get_element复用（避免重复访问全局内存），显著降低内存访问延迟，适用于需要在循环中多次访问同一片数据的不规则内存访问场景
- [`DC-21` 矩阵乘法 Swizzle2D 优化案例](case_matmul_swizzle2d.md)：大矩阵乘法Swizzle2D优化：固定核心数启动（grid=20而非所有块）+Swizzle2D块重排（GROUP_SIZE=4）提升缓存局部性，根据M/N比例自适应选择分组方向，适用于大规模矩阵乘法（千万级元素）的Ascend NPU场景
- [`DC-22` 大规模 Amax 归约优化（reduce轴映射多核）](case_reduction_amax_large.md)：非reduce轴很小、reduce轴很大的归约优化：将reduce轴映射到多核（而非常规的非reduce轴），使用原子操作跨线程块归约，通过二次切分避免超UB，适用于极端shape比例（M<<N如16×262144）的归约场景
- [`DC-23` Amin 归约原子操作优化案例](case_reduction_amin_atomic.md)：原子操作归约（amin）优化：非reduce轴很小时将reduce轴映射多核，提供循环内/外两种原子操作方案（减少存储vs减少竞争），通过二次切分+计算重组提升性能，适用于M<<N（如16×262144）的极端shape比例场景
- [`DC-24` @architecture_decision("ub-aware-row-block", reason="TopK 每行计算量小，需通过 ROWS_PER_BLOCK 提升每 program 计算密度；行数受 UB 容量约束")](case_sort_topk.md)：TopK 优化：目标维 permute 到末维后按行处理，每行切分为 <=4096 的 2 幂 tile，tile 内用 ext.sort 排序取 top-K，再用固定 next_pow2(2*K) 缓冲区合并，避免全局排序和选择排序
- [`DC-25` Attention 算子优化](coding_attention.md)：标准的 Scaled Dot-Product Attention:。
- [`DC-26` Triton Ascend Layout-transform / Permutation 代码生成要点](coding_layout_transform.md)：当算子为 permute、transpose、reshape（触发 contiguous copy）等仅改变数据布局的算子时，代码生成应遵循以下原则。
- [`DC-27` Reduce 算子优化](coding_reduce.md)：def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):。
- [`DC-28` Sort/Select 算子优化](coding_sort_select.md)：Triton Ascend 不支持 break/continue/return 和 Python if 分支，必须用 tl.where + mask 实现条件逻辑。
- [`DC-29` 布局变换算子设计约束](layout_transform_design.md)：当算子属于 permute / transpose / contiguous / reshape-as-copy 等改变数据布局但不改变元素值的算子时，草图应优先采用以下模式。
- [`DC-3` flash_attention 类算子设计约束](flash_attention_constraints.md)：flash_attention 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-30` BLOCK_SIZE 配置选择与边界处理策略](blocksize_and_boundary_policy.md)：在草图中必须添加"设计适用范围"注释：。
- [`DC-4` index_computation 类算子设计约束](index_computation_constraints.md)：index_computation 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-5` math_compute 类算子设计约束](math_compute_constraints.md)：math_compute 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-6` matrix_compute 类算子设计约束](matrix_compute_constraints.md)：matrix_compute 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-7` normalization 类算子设计约束](normalization_constraints.md)：normalization 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-8` object_detection 类算子设计约束](object_detection_constraints.md)：object_detection 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`DC-9` quantization 类算子设计约束](quantization_constraints.md)：quantization 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
- [`GC` Triton Ascend 全局设计约束](global_constraints.md)：跨全部算子类别复用的 15 条设计期硬约束，覆盖并行配置、编译期常量化、精度提升、访存与平台限制；各类别约束卡只写本类别特有条款并链接引用本卡。
- [`DC-31` Attention 家族写法分类](attention_writing_classification.md)：attention 家族按「写法」（kernel 骨架）分类的唯一入口：判别表、叠加优先级、无 template 子类的处置与新增子类规矩；命中 attention 家族时先读本卡确定写法类。
- [`DC-32` mha 类算子设计约束](mha_constraints.md)：MHA / SelfAttention / CrossAttention（4 个 nn.Linear 投影 + softmax(QKᵀ)@V）及相邻写法（反传链、单头大 C SDPA）；投影段约束 M1~M4 与多算子实证。
- [`DC-33` gqa 类算子设计约束](gqa_constraints.md)：GQA/MQA head 共享、router 选路、含投影 GQA，及与块跳过/页表寻址叠加形态；任务单元按 kv_head 分组、掩码在 scores 上、tile 禁切片。
- [`DC-34` linear 类算子设计约束](linear_constraints.md)：Linformer 等「K/V 经可学习低秩矩阵投影降维后再 attention」的线性复杂度注意力；contraction tile 上限、copy 消除、Kr/Vr hoist。
- [`DC-35` block_sparse_attention 类算子设计约束](block_sparse_attention_constraints.md)：块稀疏 / 掩码跳块类 attention（block-sparse、streaming、paged topK、pooled topK，含前向与反向）；program 粒度=掩码块、device 侧建表、p 二段拆分。
- [`DC-36` sparse_mask 类算子设计约束](sparse_mask_constraints.md)：Sparse-掩码型注意力（窗口局部 / 块对角 / 元素周期 / 复合 / 轴向分解 / 加性偏置）；以掩码生效层 + 几何形态为核心路由轴，通用经验 M1~M12。
- [`DC-37` vision_mlp 类算子设计约束](vision_mlp_constraints.md)：视觉 MLP（WeightedPermuteMLP 等 ViP/Permutator）；禁 gather 融合、特化转置 kernel、GEMM slim-tile 分带、小 softmax 内联消费。
- [`DC-38` frequency_filter 类算子设计约束](frequency_filter_constraints.md)：频域滤波（FFT → 逐元素复数变换 → 逆 FFT，如 GlobalFilter）；禁 torch.fft、纯 Triton 可分离 2D-DFT、host 侧禁三角函数。
- [`DC-39` elementwise 类算子设计约束](elementwise_constraints.md)：逐元素非线性激活（GELU / SiLU / SwiGLU）的判别与硬约束：1D 展平连续访存、动态 grid 钳制核数、fp32 计算存回原 dtype、compiler hints 逐算子独立判断。
- [`DC-40` paged_attention 类算子设计约束](paged_attention_constraints.md)：分页 KV Cache Attention（decode / prefill，跨页不跳块）：block_ptr 禁跨页 gather、golden 舍入链精度契约、decode 按 kv_head / prefill split-q 任务划分与证伪方向。
- [`DC-41` cv_attn_agg 类算子设计约束](cv_attn_agg_constraints.md)：CV-Attn 聚合分发（S2 / CoT / DoubleAttention）：聚合-分发结构轴、RNE/Kahan/tanh 精度三件套、结合律重排配精度门控。
- [`DC-42` dual_branch_attention 类算子设计约束](dual_branch_attention_constraints.md)：CV 双分支特征图门控（BAM / PSA / TripletAttention）：禁物化 [B,C2,N]、conv 平面化地址恒非负仿射、任务循环 grid=min(NTASK,cores)。
- [`DC-43` linear_recurrent 类算子设计约束](linear_recurrent_constraints.md)：矩阵状态递推：状态驻 UB 单融合 persistent kernel、masked 2D store 静默丢格、BK=64 编译红线、与 recurrent 家族 host-loop 分工。
- [`DC-44` mla 类算子设计约束](mla_constraints.md)：MLA 矩阵吸收（ckv 双作 K/V、q 拆 nope/pe）：BLOCK_QO=1 可证仿射、KV 复用、16-bit PV 判据式、lse 底数契约。
- [`DC-45` paged_memory_copy 类算子设计约束](paged_memory_copy_constraints.md)：Paged Memory Copy：禁 constexpr-unrolling 标量 scatter 循环、禁 compiler hints、2D tile [CHUNK_SIZE, head_dim] CS32。
- [`DC-46` sampling 类算子设计约束](sampling_constraints.md)：核采样过滤（top-p）：索引与 torch.topk 精确一致、数值归约 fp32、移除规则闭式公式、forward 内禁 PyTorch 回退。
- [`DC-47` sparse_unfold 类算子设计约束](sparse_unfold_constraints.md)：邻域展开 attention（F.unfold/im2col）：aclnn 行布局舍入契约、softmax 分 dtype 逐位配方、运行时除数解码 miscompile、dot 操作数 2 的幂。
- [`DC-48` spatial_attention 类算子设计约束](spatial_attention_constraints.md)：CV-Attn 空间自注意力（PVT/SR-Attention）：fp32 全链路精度契约、depthwise conv 权重 host 预转置、tile 参数档位化生死线。
- [`DC-49` diffusion_embedding 类算子设计约束](diffusion_embedding_constraints.md)：扩散模型条件嵌入（timestep 正弦嵌入 + 小 MLP + 并行投影）：skinny-M 禁 tl.dot 走向量外积、段级深度融合、逐舍入点 dtype 契约、共享 K 多路 pass-merge、tile 整域 law。
- [`DC-50` patch_window_attention 类算子设计约束](patch_window_attention_constraints.md)：CV-Attn 卷积-块化窗口 transformer（MobileViT 系）：复合点积双 B 指针禁令、拼接缓冲紧凑布局、元素级 gather 禁令、fp32 ieee dot、attn 按 n 单块特化。
- [`DC-51` position_channel_attention 类算子设计约束](position_channel_attention_constraints.md)：CV-Attn 无投影位置/通道双分支（DAModule/DANet 系）：双分支统一物理布局、PV 二段拆分、conv implicit GEMM+RNG 复刻、tile 收紧、双路径分派、分段精度契约。
