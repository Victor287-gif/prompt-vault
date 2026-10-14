# MaskPredicate 优化经验

## 知识卡

- [triu：冗余消除优化](triu_optimization_a2_id_0.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [greater_equal：流水与内存优化](greater_equal_pipeline_memory_a2.md)：greater_equal 算子的流水与内存优化经验，关注标量开销。
- [is_inf：计算与内存优化](is_inf_memory_compute_a2.md)：is_inf 算子的计算与内存优化经验，关注UB 内存。
- [equal：并行切核优化](equal_parallel_kernel_a2.md)：equal 算子的并行切核优化经验，关注标量开销。
- [triu：计算优化](triu_compute_a2.md)：triu 算子的计算优化经验，关注流水线停顿。
- [is_inf：流水与并行优化](is_inf_pipeline_parallel_pipeline_a2.md)：is_inf 算子的流水与并行优化经验，关注标量开销。
- [triu：内存与 Tiling优化](triu_memory_tiling_tiling_memory_a2.md)：triu 算子的内存与 Tiling优化经验，关注混合或尚未确定的瓶颈。
- [triu：内存优化](triu_memory_a2.md)：triu 算子的内存优化经验，关注标量开销。
- [triu：冗余消除优化](triu_optimization_a2_id_1.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [triu：冗余消除优化](triu_optimization_a2_id_2.md)：triu 算子的冗余消除优化经验，关注内存管理开销。
- [is_inf：计算优化](is_inf_compute_a2.md)：is_inf 算子的计算优化经验，关注标量开销。
- [triu：冗余消除优化](triu_optimization_a2_id_3.md)：triu 算子的冗余消除优化经验，关注标量开销。
- [tril：流水与内存优化](tril_pipeline_memory_a2.md)：tril 算子的流水与内存优化经验，关注标量开销。
- [less_equal：流水与并行优化](less_equal_pipeline_parallel_pipeline_a2.md)：less_equal 算子的流水与并行优化经验，关注标量开销。
- [triu：冗余消除优化](triu_optimization_a2_id_4.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [is_inf：内存优化](is_inf_memory_a2.md)：is_inf 算子的内存优化经验，关注混合或尚未确定的瓶颈。
- [triu：冗余消除优化](triu_optimization_a2_id_5.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [triu：冗余消除优化](triu_optimization_a2_id_6.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [less_equal：内存与 Tiling优化](less_equal_memory_tiling_tiling_memory_a2_id_0.md)：less_equal 算子的内存与 Tiling优化经验，关注流水线停顿。
- [tril：Tiling优化](tril_tiling_a2.md)：tril 算子的Tiling优化经验，关注混合或尚未确定的瓶颈。
- [triu：流水与并行优化](triu_pipeline_parallel_pipeline_a2.md)：triu 算子的流水与并行优化经验，关注标量开销。
- [triu：冗余消除优化](triu_optimization_a2_id_7.md)：triu 算子的冗余消除优化经验，关注内存管理开销。
- [triu：冗余消除优化](triu_optimization_a2_id_8.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [triu：冗余消除优化](triu_optimization_a2_id_9.md)：triu 算子的冗余消除优化经验，关注混合或尚未确定的瓶颈。
- [less_equal：内存与 Tiling优化](less_equal_memory_tiling_tiling_memory_a2_id_1.md)：less_equal 算子的内存与 Tiling优化经验，关注混合或尚未确定的瓶颈。
