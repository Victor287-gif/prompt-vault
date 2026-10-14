# LayoutTransform 优化经验

## 知识卡

- [strideslice_neg_concat_v2：计算优化](strideslice_neg_concat_v2_compute_a2.md)：strideslice_neg_concat_v2 算子的计算优化经验，关注标量开销。
- [reshape：流水与并行优化](reshape_pipeline_parallel_pipeline_a2.md)：reshape 算子的流水与并行优化经验，关注标量开销。
- [reshape：循环计算优化](reshape_loop_compute_compute_a2_id_0.md)：reshape 算子的循环计算优化经验，关注流水线停顿。
- [reshape：循环计算优化](reshape_loop_compute_compute_a2_id_1.md)：reshape 算子的循环计算优化经验，关注数据搬运。
- [strideslice_neg_concat_v2：循环计算优化](strideslice_neg_concat_v2_loop_compute_compute_a2.md)：strideslice_neg_concat_v2 算子的循环计算优化经验，关注标量开销。
- [reshape：循环计算优化](reshape_loop_compute_compute_a2_id_2.md)：reshape 算子的循环计算优化经验，关注流水线停顿。
- [reshape：循环计算优化](reshape_loop_compute_compute_a2_id_3.md)：reshape 算子的循环计算优化经验，关注数据搬运。
- [reshape：数据搬运优化](reshape_data_movement_a2.md)：reshape 算子的数据搬运优化经验，关注流水线停顿。
- [reshape：Tiling优化](reshape_tiling_a2.md)：reshape 算子的Tiling优化经验，关注内存带宽。
- [reshape：内存与 Tiling优化](reshape_memory_tiling_tiling_memory_a2.md)：reshape 算子的内存与 Tiling优化经验，关注标量开销。
- [reshape：流水优化](reshape_pipeline_a2.md)：reshape 算子的流水优化经验，关注流水线停顿。
