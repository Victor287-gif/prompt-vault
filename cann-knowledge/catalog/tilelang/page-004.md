# tilelang: 211–280 / 320

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Elementwise / Gather 的 Tiling、任务与 Vector 数据流搜索** · 211 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/elementwise/tiling_task_vector_search.md`
- **Elementwise SIMD 与融合** · 31 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/elementwise/vector_efficiency_design.md`
- **KV 连续批量搬运** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/flash_attention/getvalue_cacheline_batching_design.md`
- **FlashAttention 标量热循环精简** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/flash_attention/scalar_general_techniques_design.md`
- **FlashAttention 编译期预计算** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/flash_attention/scalar_precompute_design.md`
- **FlashAttention 标量访存降压** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/flash_attention/scalarldst_subbound_design.md`
- **Cube-Vector Buffer 转发** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/flash_attention/tilingdata_cv_ssbuf_forwarding_design.md`
- **GEMM 一侧驻留** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/fullload_design.md`
- **GEMM 提前发射与深流水** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/mte2_preload_design.md`
- **GEMM 多级流水** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/pingpong_design.md`
- **Block-scale 合并载入** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/scale_coalescing_design.md`
- **PTO Stream-K 两阶段实现** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/streamk_design.md`
- **Shape-aware 输出 Tile 调度** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/matmul/swat_design.md`
- **Local GEMM 与通信重叠** · 7 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/mc2/local_matmul_design.md`
- **通信与 PTO GEMM 流水配平** · 7 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/mc2/pipeline_balancing_design.md`
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
- **-----------------------------------------------------------------------------------------------------------** · 101 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/rope/code/rope_vf_common.py`
- **RoPE SIMD 融合** · 11 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/rope/vector_fused_design.md`
- **PTO Buffer 管理** · 7 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scalar/buffer_management_conversion_guide.md`
- **TileLang/PTO 热路径编码原则** · 11 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scalar/coding_principles.md`
- **PTO Conv/GEMM 标量精简** · 7 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scalar/conv_scalar.md`
- **超长行三阶段 Split Scan** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/cum_core_partition_scan.md`
- **SIMD Lane-parallel Scan** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/cum_lane_parallel_scan.md`
- **Sklansky Prefix Network** · 24 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/cum_oneway_sklansky.md`
- **Row Kogge-Stone Scan** · 21 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/cum_row_kogge_stone.md`
- **流式 Scan** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/cum_streaming_scan.md`
- **UB Resident Scan** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/cum_tile_resident_scan.md`
- **-----------------------------------------------------------------------------------------------------------** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/cum_core_partition_scan.py`
- **-----------------------------------------------------------------------------------------------------------** · 19 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/cum_lane_parallel_scan.py`
- **-----------------------------------------------------------------------------------------------------------** · 30 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/cum_oneway_sklansky.py`
- **-----------------------------------------------------------------------------------------------------------** · 24 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/cum_row_kogge_stone.py`
- **-----------------------------------------------------------------------------------------------------------** · 19 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/cum_streaming_scan.py`
- **-----------------------------------------------------------------------------------------------------------** · 24 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/cum_tile_resident_scan.py`
- **-----------------------------------------------------------------------------------------------------------** · 69 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/scan_base.py`
- **-----------------------------------------------------------------------------------------------------------** · 16 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/dav310/scan_cummin_tiling.py`
- **-----------------------------------------------------------------------------------------------------------** · 20 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/scan_tiling.py`
- **-----------------------------------------------------------------------------------------------------------** · 32 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/scan_tiling_data.py`
- **Scan Kernel 接入流程** · 23 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/scan/templates/usage_guide.md`
- **TileLang/PTO SIMT 优化** · 11 行 · upstream
  `cannbot-skills/ops/tilelang-performance-best-practices/Ascend950/references/simt/optimization-guide.md`

下一页：page-005.md
