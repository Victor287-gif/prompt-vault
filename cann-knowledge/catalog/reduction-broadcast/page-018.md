# reduction-broadcast: 1191–1260 / 1263

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Vector 算子归约遍数融合（Reduce Pass Fusion）** · 227 行 · upstream
  `cannbot-skills/ops/tilelang-perf-optimization/Ascend910/references/vector-practices/vector_reduce_pass_fusion.md`
- **Broadcast Shape 专用 Tiling** · 27 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/advanced_tiling.md`
- **Broadcast 总体实现** · 27 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/broadcast_design.md`
- **-----------------------------------------------------------------------------------------------------------** · 55 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/code/broadcast_add_kernel.py`
- **-----------------------------------------------------------------------------------------------------------** · 24 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/code/broadcast_add_tiling.py`
- **-----------------------------------------------------------------------------------------------------------** · 30 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/code/broadcast_common.py`
- **-----------------------------------------------------------------------------------------------------------** · 15 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/code/onedim_add_kernel.py`
- **-----------------------------------------------------------------------------------------------------------** · 14 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/code/onedim_add_tiling.py`
- **Broadcast 尾部填充** · 27 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/datacopypad_design.md`
- **多维 Broadcast 与矩形搬运** · 27 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/nddma_design.md`
- **单轴 Broadcast** · 27 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/onedim_design.md`
- **UB 常驻 Broadcast** · 27 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/broadcast/ub_broadcast_design.md`
- **-----------------------------------------------------------------------------------------------------------** · 31 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_base.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_big_dim.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_cut_one_axis.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_cut_two_axis.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_n_last.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_small_shape.py`
- **-----------------------------------------------------------------------------------------------------------** · 15 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_tensor_move.py`
- **-----------------------------------------------------------------------------------------------------------** · 44 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_transdata_5hd.py`
- **-----------------------------------------------------------------------------------------------------------** · 30 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_transdata_5hd_021axis.py`
- **-----------------------------------------------------------------------------------------------------------** · 17 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/dav3510/transpose_with_gather.py`
- **-----------------------------------------------------------------------------------------------------------** · 36 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/templates/transpose_tiling_data.py`
- **大维度分块 Transpose** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_big_dim.md`
- **单轴切分 Transpose** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_cut_one_axis.md`
- **双轴切分 Transpose** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_cut_two_axis.md`
- **Transpose 融合** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_fusion_design.md`
- **SIMD Gather Transpose** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_gather.md`
- **N-last Layout Conversion** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_n_last.md`
- **小 Shape Transpose** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_small_shape.md`
- **规则 Tensor Move** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_tensor_move.md`
- **Transpose 配置选择** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_tiling_data.md`
- **0-2-1 三维变换** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_vconv_021.md`
- **5HD Layout Conversion** · 22 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/conversion/transpose_vconv_5hd.md`
- **批量短归约与逐记录写回** · 26 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/batched_short_reduction.md`
- **固定小状态迭代归一化** · 21 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/fixed_small_state.md`
- **行归约与逐元素写回** · 21 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/rowwise_reduce_epilogue.md`
- **-----------------------------------------------------------------------------------------------------------** · 22 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/euclidean_norm_group_block_split_r.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/euclidean_norm_interleaved_contiguous_tail_a.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/euclidean_norm_interleaved_contiguous_tailr.py`
- **-----------------------------------------------------------------------------------------------------------** · 104 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/kernel_utils.py`
- **-----------------------------------------------------------------------------------------------------------** · 24 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_ar_full_load.py`
- **-----------------------------------------------------------------------------------------------------------** · 19 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_ar_recompute.py`
- **-----------------------------------------------------------------------------------------------------------** · 24 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_ar_small_r.py`
- **-----------------------------------------------------------------------------------------------------------** · 19 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_ara_full_load.py`
- **-----------------------------------------------------------------------------------------------------------** · 26 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_ara_online.py`
- **-----------------------------------------------------------------------------------------------------------** · 19 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_ara_recompute.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/dav310/softmax_v2_base.py`
- **EuclideanNorm 两阶段 Split-R** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/euclidean_norm_group_block_split_r.md`
- **-----------------------------------------------------------------------------------------------------------** · 18 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/euclidean_norm_group_tiling.py`
- **EuclideanNorm 连续尾块** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/euclidean_norm_interleaved_contiguous_tail_a.md`
- **EuclideanNorm 多行交错** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/euclidean_norm_interleaved_contiguous_tailr.md`
- **-----------------------------------------------------------------------------------------------------------** · 19 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/euclidean_norm_tiling.py`
- **-----------------------------------------------------------------------------------------------------------** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/euclidean_norm_tiling_data.py`
- **Attention Online Softmax** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/fused_attention_online_softmax_design.md`
- **Softmax AR Full-load** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_ar_full_load.md`
- **Softmax AR Recompute** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_ar_recompute.md`
- **Softmax 小归约轴** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_ar_small_r.md`
- **ARA Full-load Softmax** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_ara_full_load.md`
- **ARA Online Softmax** · 34 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_ara_online.md`
- **ARA Recompute Softmax** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_ara_recompute.md`
- **-----------------------------------------------------------------------------------------------------------** · 20 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_tiling.py`
- **-----------------------------------------------------------------------------------------------------------** · 32 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/softmax_v2_tiling_data.py`
- **Reduction 状态常驻** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/state_resident_design.md`
- **Reduction Kernel 使用流程** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/templates/usage_guide.md`
- **Reduction 宽输出轴 Tiling** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/reduce/wide_output_tiling.md`
- **PTO Buffer 管理** · 7 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scalar/buffer_management_conversion_guide.md`
- **Layer Norm Profiler 性能基准说明** · 28 行 · upstream
  `cannbot-skills/ops/torch-ops-profiler/examples/layer_norm_profiler_reference/LAYER_NORM_PROFILER_PERF_GUIDE.md`
- **Layer Norm profiler 参考目录** · 12 行 · upstream
  `cannbot-skills/ops/torch-ops-profiler/examples/layer_norm_profiler_reference/README.md`
- **-*- coding: utf-8 -*-** · 126 行 · upstream
  `cannbot-skills/ops/torch-ops-profiler/examples/layer_norm_profiler_reference/benchmark_layer_norm_torch_npu_profiler.py`

下一页：page-019.md
