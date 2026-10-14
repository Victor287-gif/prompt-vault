# 实测性能案例（perf_card）

按 shape 特征驱动的实测优化案例，含配置梯度与实测延迟。
每张卡的 `measured` 字段记录最优配置与对照档位。

- [2D Broadcast Division 优化案例](elemwise_broadcast_2d.md)：2D广播除法优化：小维度完整处理不切分（循环外加载复用），通过固定NUM_BLOCKS实现核间并行（40核），核内SUB_M控制粒度平衡UB利用率，适用于broadcast轴大但非broadcast轴小的2D场景
- [中等规模 Amax 归约优化](reduction_amax_medium.md)：中等规模归约（amax）优化：计算重组（循环内累加、循环外归约）减少归约次数，grid=40等于核数时性能最优（25.73us），适用于非reduce轴中等、reduce轴较大（千万级元素）的2D归约场景
- [极小规模 Amax 归约优化](reduction_amax_small.md)：极小规模归约（amax）优化：单核处理（grid=1）优于多核并行（2.16us vs 3.51us），避免并行化带来的调度开销，适用于数据规模很小（<1000元素）的归约场景
- [极大规模 1D Amin 归约优化](reduction_amin_large.md)：极大规模1D归约（amin）优化：二次切分避免超UB+计算重组减少归约次数，网格数接近AI Core数量（grid=32）、UB用满、无尾块时性能最优（9.61us），适用于极大规模1D数据（400万级元素）的全量归约场景
- [大规模 2D Amin 归约优化](reduction_amin_medium.md)：大规模2D归约（amin）reduce轴很大优化：在优先占满UB前提下为reduce轴分配较大切分尺寸（BLOCK_SIZE_N=16384最优），减少循环次数但需权衡单次迭代负载，适用于非reduce轴中等、reduce轴很大（50万级元素）的场景
- [中等规模 1D Amin 归约优化](reduction_amin_small.md)：中等规模1D归约（amin）优化：适中并行度（grid=8时最优2.21us），存在最优平衡点（过小导致单块负载过重、过大引入调度开销），适用于中等规模1D数据（6万级元素）的全量归约场景
- [大规模 Mean 归约优化（行二次切分）](reduction_mean_large.md)：大规模reduce最后根轴（mean）行二次切分优化：每个kernel计算多行减少线程块数量、kernel内二次切分避免超UB，grid=40且SUB切分不含尾块时性能最优（16.00us），尾块计算会显著降低性能，适用于非reduce轴中等、reduce轴较大的2D归约场景
- [中等规模 Mean 归约优化（reduce第一根轴）](reduction_mean_medium.md)：中等规模reduce第一根轴（mean）优化：计算重组减少归约次数，网格规模略小于AI Core数量且避免尾块时性能最佳（grid=32最优9.98us），适用于reduce第一根轴、两轴均中等（百万级元素）的2D归约场景
- [小规模 Prod 归约优化](reduction_prod_small.md)：小规模reduce第一根轴（prod）优化：使用自定义mul函数配合tl.reduce实现连乘（triton无prod接口），最优网格数明显小于AI Core数量（grid=16最优2.15us），过高并行度反而因调度开销降低性能，适用于shape较小（10万级元素）的reduce第一根轴场景
- [Reduction + Elementwise 融合算子优化](reduction_sum_fused.md)：Reduction+Elementwise融合算子优化：先逐元素操作再归约，行二次切分+计算重组，grid=40且SUB切分不含尾块时性能最优（47.58us），融合优化逻辑以reduce为主，适用于需要先逐元素计算再reduce的融合场景
- [大规模 Sum 归约优化](reduction_sum_large.md)：大规模归约（sum）非reduce轴很大优化：计算重组减少归约次数，在优先占满UB前提下为reduce轴分配较大切分尺寸（BLOCK_SIZE_N=1024最优685.65us），适用于非reduce轴非常大（6万+）、reduce轴中等（千级）的2D归约场景
- [Weighted SwiGLU Backward 融合算子优化](reduction_weighted_swiglu.md)：3D融合算子（Weighted SwiGLU Backward）优化：Reshape降维将前两维合并简化并行策略，行二次切分避免超UB，在优先占满UB前提下为reduce轴分配较大切分尺寸，grid数较大时可能性能更优，适用于3D张量逐元素+reduce融合的场景
