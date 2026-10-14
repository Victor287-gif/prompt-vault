# 算子实现手册（playbook）

按算子类别组织。每张卡是单个算子的实现手册，含设计约束、算法骨架、
关键 kernel 与性能基准；跨类别通用约束见
global_constraints.md（`../constraints/global_constraints.md`）。

## attention（1 张）

- [MultiHeadAttention 算子（mha） 实现手册](attention__multiheadattention.md)：典型特征: queryB,S_Q,D / key,valueB,S_K,D，n_heads 为 attr。

## convolution（5 张）

- [ConvDepthwise2d 算子（2D depthwise） 实现手册](convolution__convdepthwise2d.md)：算子类别：convolutiondepthwise2d　典型特征：(N, C, H, W) × weight C, 1, kH, kW（groups==C），无跨通道规约。
- [ConvStandard1d 算子（1D 标准卷积） 实现手册](convolution__convstandard1d.md)：算子类别：convolution1d　典型特征：(N, C_in, L) × weight C_out, C_in//groups, K → (N, C_out, L_out)。
- [ConvStandard2d 算子（2D 标准/grouped 卷积） 实现手册](convolution__convstandard2d.md)：典型特征: (N, C_in, H, W) × weight [C_out, C_in//groups, K, K] → (N, C_out, H_out, W_out)；stride/padding/dilation/groups/bias 全为 forward 运行期动态参数，含标准 / grouped / 1×1 / depthwise-like（Cin_pg==1 && Cout_pg==1）变体。
- [ConvStandard3d 算子（3D 标准 / grouped，源 benchmark 实际 `kernel_size=(k,k,1)`） 实现手册](convolution__convstandard3d.md)：算子类别：convolution3dstandardgrouped	典型特征：(N, C, D, H, W) × weight C_out, C_in//groups, Kd, Kh, 1。
- [ConvTranspose2d 算子（2D 转置卷积） 实现手册](convolution__convtranspose2d.md)：算子类别：convolutiontranspose2d　典型特征：weight (in_channels, out_channels, KH, KW)，输出放大。

## cv_attn_agg（3 张）

- [S2Attention 算子（s2attention） 实现手册](cv_attn_agg__s2attention.md)：典型特征: 通道扩展-分支分发——通道先扩 3x（Linear C→3C）拆 3 分支做 spatial shift，mean 归约到 [B,C] 后经 MLP3+GeLU+softmax 逐通道加权，加权结果经 MLP2 聚合回空间。
- [CoTAttention 算子（cotattention） 实现手册](cv_attn_agg__cotattention.md)：典型特征: 空间-通道双 attention——Coordinate Attention：grouped conv key embed 产空间注意权重，value_embed conv1x1 产通道权重，经 2 次轻量 attention（att1/att2）+ bias/mean 汇聚。
- [DoubleAttention 算子（doubleattention） 实现手册](cv_attn_agg__doubleattention.md)：典型特征: conv 特征-双 softmax 汇聚——conv1x1 变换（convA/convB/convV）→ 2 次 softmax-over-hw + 2 次 bmm 汇聚（gather GD），分发双路（结合律重排 vs golden-order）后聚合回空间；终态 P10 gather 重关联 2.2794×。

## diffusion_embedding（1 张）

- [FluxTimestepGuidanceProjectionEmbedding 算子（diffusion-embedding） 实现手册](diffusion_embedding__flux_timestep_guidance_projection_embedding.md)：典型特征: timestep[B]×1000 → 外积 freqs → sin/cos concat；Linear+SiLU+Linear 全程 fp32 + 并行投影路，两路相加 cast 回输入 dtype；3-kernel 全融合 + 向量外积替代 tl.dot + BM 分档 law。

## dual_branch_attention（3 张）

- [BAM 算子（bam） 实现手册](dual_branch_attention__bam.md)：典型特征: BAM：通道 GAP→MLP(C→C/r→C)→sigmoid 支与空间 k×k 膨胀 conv 支加性门控 out=x·(1+sigmoid(ch+sp))，k×k conv 走 shifted-1×1 dot（Cube 路径）。
- [Polarized Self-Attention 算子（polarized_self_attention） 实现手册](dual_branch_attention__polarized_self_attention.md)：典型特征: PSA 通道 softmax-over-N 支与空间 GAP/softmax-over-C2 支门控特征图；并行加性 op57 out=x·(S+T) / 串行乘性 op61 out=x·T·S，7-kernel GEMV 链。
- [TripletAttention 算子（triplet_attention） 实现手册](dual_branch_attention__triplet_attention.md)：典型特征: CH/CW/HW 三分支两两交互 zpool(mean/max)→门控 conv→BN→sigmoid，out=(o1+o2+o3)/3，permute 全消解为索引映射。

## elementwise（2 张）

- [Activation 算子（GELU / SiLU / SwiGLU） 实现手册](elementwise__activation.md)：典型特征: 逐元素非线性激活，1D 展平连续访存，fp32 计算存回原 dtype。
- [Add 算子（elementwise 逐元素加法） 实现手册](elementwise__add.md)：典型特征: out = x + y，scalar/row/跨轴广播（1D-4D），fp32/fp16/bf16 混合；单 kernel 统一 persistent no-autotune，融合搜索基因 G017/G021/G018/G019/G014/G016/G020（§1.5）。

## flash_attention（2 张）

- [Flash Attention 实现手册](flash_attention__flash_attention.md)：FA 形态的设计约束、算法骨架与关键技巧；含与朴素 attention 的差异声明。
- [FA 精度闸门与测量口径](flash_attention__precision_gate_and_measurement.md)：Flash Attention 的精度验收闸门与性能测量口径；不过闸门谈性能无意义。

## index_computation（4 张）

- [EmbeddingDenseBackward 算子（index-accumulate） 实现手册](index_computation__embeddingdensebackward.md)：算子类别: indexaccumulate（按索引将梯度累加到 embedding 权重。
- [Index / Gather 算子（index-read） 实现手册](index_computation__index_gather.md)：算子类别: indexread（按索引从输入中收集元素。
- [IndexPut / Scatter 算子（index-write） 实现手册](index_computation__indexput_scatter.md)：算子类别: indexwrite（按索引向输出写入或原子累加。
- [Sort / TopK 算子（index-sort） 实现手册](index_computation__sort_topk.md)：算子类别: indexsort（基于比较重排并返回索引。

## linear_recurrent（1 张）

- [FusedRecurrentGatedDeltaRule 算子（Gated Delta Rule） 实现手册](linear_recurrent__fused_recurrent_gated_delta_rule.md)：典型特征: 每 (b,vh) 一套 [K,V] fp32 状态，T 步串行递推，delta rule rank-1 修正，状态驻留 UB 单融合 persistent kernel。

## math_compute（3 张）

- [HyenaFftSizePaddingRfft 算子（transformation-compute） 实现手册](math_compute__hyenafftsizepaddingrfft.md)：算子类别: transformationcompute（自定义 FFT / RFFT 类）。
- [Sort 算子（sort-topk） 实现手册](math_compute__sort.md)：算子类别: sorttopk。
- [Sum 算子（sum） 实现手册](math_compute__sum.md)：算子类别: sum（多维 reducesum，支持 dim / keepdim / dim=None 全归约）。

## matrix_compute（4 张）

- [FFN 算子（feed-forward-network） 实现手册](matrix_compute__ffn.md)：两级 GEMM + 激活（标准/门控/int8 量化/MoE 四路径，53 case）；融合搜索最优 **1.3509x**（kernel `f419e763`，G026 traffic-tile 基因）。
- [GroupedMatmulSwigluQuant 算子（grouped-matmul-swiglu-quant） 实现手册](matrix_compute__groupedmatmulswigluquant.md)：算子类别: groupedmatmulswigluquant。
- [GroupedMatmulSwigluQuantV2 算子（grouped-matmul-swiglu-quant-v2） 实现手册](matrix_compute__groupedmatmulswigluquantv2.md)：算子类别: grouped matmul + activation + quantization（expert 边界对齐 BLOCK_M + 扁平并行 + partial row_max 免 atomic，0.7738x）。
- [QuantMatmul 算子（quantization-matmul） 实现手册](matrix_compute__quantmatmul.md)：算子类别: quantizationmatmul。

## mla（1 张）

- [MLA 类算子（MultiHeadLatentAttention） 实现手册](mla__multi_head_latent_attention.md)：典型特征: ckv 同时作 K 与 V 的矩阵吸收，q 拆 nope/pe 双 dot + 分页间接寻址。

## normalization（2 张）

- [AdaptiveInstanceNorm2DBackward 算子（norm-backward / backward） 实现手册](normalization__adaptiveinstancenorm2dbackward.md)：算子类别: normbackward（反向多粒度归约）。
- [GroupNormSwish 算子（norm-fused / forward） 实现手册](normalization__groupnormswish.md)：算子类别: normfused（分组归一化 + Swish 激活融合）；融合搜索最优 **1.4031x**（kernel `e9bbf874`，结构基因 G013/G015 + 参数 FP16_BLOCK=16384/NUM_WARPS=1）。

## object_detection（2 张）

- [IOU 算子（Intersection over Union） 实现手册](object_detection__iou.md)：算子类别: elementwise / pairwisecompute。
- [NMS 算子（Non-Maximum Suppression） 实现手册](object_detection__nms.md)：算子类别: sorttopk + indexinggather 复合（顺序算法 + 每轮成对 IoU 计算）。

## paged_attention（1 张）

- [Paged KV Cache Attention 算子（decode / prefill） 实现手册](paged_attention__paged_kv_cache_attention.md)：典型特征: KV 经 block_table 间接寻址的分页注意力，批量小页 decode / 大页 decode / split-q prefill 三子形态，跨页不跳块（整段 KV 都参与 softmax）。

## paged_memory_copy（1 张）

- [StorePagedKVCache 算子（Scatter-Copy） 实现手册](paged_memory_copy__store_paged_kv_cache.md)：典型特征: 按预计算元数据 scatter-copy K/V token 到 paged cache，离散写入 memory-bound。

## patch_window_attention（1 张）

- [MobileViT 系局部 transformer（patch_window_attention） 实现手册](patch_window_attention__mobilevit_local_transformer.md)：典型特征: 卷积-块化窗口 transformer——conv 局部表示（k×k+1×1）→ NCHW→(b,p,n,D) patch 块化 unfold → 窗口内稠密多头 attention → fold → 尾部 conv(cat([x,y])) 双输入融合卷积。

## position_channel_attention（1 张）

- [DAModule（DANet 双重注意力模块） 实现手册](position_channel_attention__damodule.md)：典型特征: 无投影位置/通道双分支——Q=K=V 同一 Y（conv 输出），position 分支 (B,N,C) 与 channel 分支 (B,C,N) 各做一次 softmax(Y·Yᵀ)·Y 三段式，out=out_pos+out_ch 残差相加。

## quantization（4 张）

- [DynamicQuant 算子（reduction + element-wise 复合） 实现手册](quantization__dynamicquant.md)：算子类别: reduction + elementwise 复合（perrow max_abs reduction + 量化）。
- [SwigluQuant 算子（quantization-activation） 实现手册](quantization__swigluquant.md)：算子类别: quantizationactivation。
- [SwigluQuant 关键 kernel 实现](quantization__swigluquant_kernels.md)：SwigluQuant 的 static/dynamic 双 kernel 实现、INT4 打包与 scale 向量除法路径。
- [WeightQuantBatchmatmul 算子（weight-quantized batch matrix multiplication） 实现手册](quantization__weightquantbatchmatmul.md)：算子类别: matmul + quantization。

## recurrent（1 张）

- [LSTM 算子（lstm） 实现手册](recurrent__lstm.md)：算子类别: recurrentneuralnetwork。

## sampling（1 张）

- [Top-P Filter 算子（核采样过滤） 实现手册](sampling__top_p_filter.md)：典型特征: 逐行双 pass softmax over 排序后 top-k 概率，按 batch 行分 program。

## sparse_unfold（1 张）

- [HaloAttention 算子（邻域展开） 实现手册](sparse_unfold__halo_attention.md)：典型特征: F.unfold/im2col 构造 halo 邻域 KV + 块局部 softmax(QKᵀ/√d)V，常带 nn.Linear 投影段。

## spatial_attention（1 张）

- [EMSA 算子（PVT 系空间缩减注意力） 实现手册](spatial_attention__emsa.md)：典型特征: K/V 生产链含空间缩减下采样（ratio>1 depthwise conv 或 AvgPool2d）＋LayerNorm，scores 链含跨 head 1×1 transform conv（apply_transform 门控），softmax 后接 instance_norm（affine=False，逐 (b,h) 平面全平面归约），输入 queries[B,Lq,D]（Lq=H·W）、d_k≠d_v。

## tensor_transform（6 张）

- [Interpolate 算子（transformation-compute） 实现手册](tensor_transform__interpolate.md)：算子类别: transformationcompute（数据重排伴随权重计算。
- [Interpolate bicubic_area 路径关键 kernel](tensor_transform__interpolate_bicubic_area.md)：Interpolate 算子 bicubic_area 路径的 Triton kernel 实现与易错点。
- [Interpolate bilinear 路径关键 kernel](tensor_transform__interpolate_bilinear.md)：Interpolate 算子 bilinear 路径的 Triton kernel 实现与易错点。
- [Interpolate nearest 路径关键 kernel](tensor_transform__interpolate_nearest.md)：Interpolate 算子 nearest 路径的 Triton kernel 实现与易错点。
- [Pad 算子（transformation-memory） 实现手册](tensor_transform__pad.md)：算子类别: transformationmemory。
- [Repeat 算子（transformation-memory） 实现手册](tensor_transform__repeat.md)：算子类别: transformationmemory。

## transformer_inference（6 张）

- [AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce） 实现手册](transformer_inference__attentionsoftmaxwithsoftcappinganddropout.md)：算子类别: reduce（行级归约 + 前置 elementwise 变换 softcapping）。
- [LightningIndexer 算子（topk-index） 实现手册](transformer_inference__lightningindexer.md)：算子类别: topkindex。
- [MoeComputeExpertTokens 算子（indexing-gather / counting） 实现手册](transformer_inference__moecomputeexperttokens.md)：算子类别: indexinggather / counting。
- [MoeGatingTopKSoftmax 算子（sort-topk） 实现手册](transformer_inference__moegatingtopksoftmax.md)：算子类别: sorttopk。
- [RotaryMul 算子（rotarymul） 实现手册](transformer_inference__rotarymul.md)：算子类别: rotarymul（RoPE 旋转位置编码乘法）。
- [SparseFlashAttention 算子（sparse-attention） 实现手册](transformer_inference__sparseflashattention.md)：算子类别: sparseattention。

## transformer_training（1 张）

- [FusionAttention 算子（attention） 实现手册](transformer_training__fusionattention.md)：算子类别: attention（标准稠密 MHA / FlashAttentionScore）。
