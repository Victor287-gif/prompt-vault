# pipeline-memory: 561–630 / 731

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **TQue InitBuffer 须传深度和 uint32 长度** · 117 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_initbuffer_requires_depth_and_uint32_length.md`
- **TQue MTE2→MTE3 缺 MTE3_MTE2 fence 致回读错位** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/tque_mte2_mte3_mte3_mte2_fence.md`
- **TTK 缓存失败编译结果，修复后不重试** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ttk_cache_failed_dynamic_compile_result.md`
- **ACL 507035 — 完整 a0 统计缓冲超过 UB** · 268 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ub_slices_reduction_axis_scratch_buffers_507035.md`
- **V351 '读钳制 L=mS-S 触发 + PIPE_ALL 屏障 + 标量 SetValue 尾清零块' 执行即楔死设备（507035/errcode 264）；修复 = 加宽源缓冲使钳制永不触发** · 50 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/v351_datacopypad_clamp_lrnd_zeroing_trap_001.md`
- **按序列位缓存 cos/sin 并让 q/k 共享搬运** · 613 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/cache_position_coefficients_across_shared_inputs.md`
- **MTE2占比高≠带宽bound：发射开销判别** · 158 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/mte2_issue_overhead_vs_bandwidth_bound.md`
- **mte2/mte3 wait 双高示读写串行无重叠** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/mte2_mte3_wait_double_high_serial_no_overlap.md`
- **A single DataCopy transfer must be ≥16KB for best bandwidth** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_46_a_single_datacopy_transfer_must_be_16kb_for_best_b.md`
- **dav-m200 (310P) DataCopyPad-replacement decision tree — 4 branches, GM→UB logical/aligned stride split with the padding-not-written gotcha, UB→GM tail-safe writeback (floor + last-32B window), NEVER r** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_692_dav_m200_310p_datacopypad_replacement_decision_t.md`
- **in-kernel ragged tail on DataCopyPad-less SoCs** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md`
- **逐元素 DataCopyPad 搬运放大 25~64 倍判别** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/per_element_datacopypad_amplification_diagnosis.md`
- **逐 rank 通信路径的 MTE2 主导诊断** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/rank_flag_mte2.md`
- **删 smalle slim TU 致 e≤32 decode icache 税** · 102 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/smalle_slim_tu_e_32_decode_icache.md`
- **tile级合并DMA消除MTE2逐行发射bound** · 293 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/tile_merged_dma_eliminate_mte2_issue_bound.md`
- **TQue 深度 1 带宽减半：fp32 双缓冲修复** · 203 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/tque_depth1_bandwidth_double_buffer.md`
- **arch35 UB→GM DataCopyExtParams 拷贝写失效** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/arch35_ub_gm_datacopyextparams_write_fail.md`
- **Ascend950 DataCopy 写回 L2 驱逐丢对角线** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ascend950_datacopy_l2.md`
- **DataCopy 非32字节对齐尾块丢字节** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopy_32.md`
- **DataCopy GM→UB 需 32B 对齐，非对齐 C 统计输出全零** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopy_gm_to_ub_non_32b_aligned_silent_zero.md`
- **DataCopy 不自动按 tile 推进须显式元素 offset** · 144 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopy_setglobalbuffer_no_auto_advance_tileoffset.md`
- **DataCopyPad padding须按输入dtype对齐** · 135 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_input_dtype_align_507035.md`
- **窄尾 tile 行距错位：补齐 UB gap** · 212 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_narrow_tile_needs_explicit_ub_row_gap.md`
- **DataCopyPad 尾块改逐行一维搬入** · 291 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_partial_2d_blocks_use_per_row_copy.md`
- **合并尾块的非 32B 行导致 PathB 数据损坏** · 271 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_rows_must_keep_32_byte_block_length.md`
- **DataCopyPad多行stride单位误用导致非对齐行输出NaN** · 373 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_stride_fields_use_32_byte_block_units.md`
- **DataCopyPad stride 表示 block 间隙** · 241 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_stride_is_inter_block_gap.md`
- **逐元素 DataCopyPad 做 strided gather 输出全零** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_strided_gather.md`
- **DataCopyPad 未填满强制对齐缓冲 — 小维规约读入脏尾部** · 201 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_tail_must_initialize_full_reduction_extent.md`
- **门梯度整体错位 — CopyOut srcStride 误用行距** · 188 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopypad_unified_segment_burst_srcstride_is_segment_gap.md`
- **DataCopy blockLen 以 32B 块计非字节** · 184 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/datacopyparams_block_len_32b_block_units.md`
- **MTE2跨流水未建依赖导致UB读到上次调用残值** · 282 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/direct_mte_slot_reuse_needs_bidirectional_hard_events.md`
- **共享权重buffer被覆写致l≥1误用wc** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fused_kernel_shared_weight_ub_buffer_later_load_overwrites.md`
- **DataCopyPad 后未先同步 Cast 导致低精度 NaN** · 364 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/mte2_barrier_must_precede_vector_cast.md`
- **多核非对齐 DataCopyPad 写回边界互相覆盖** · 194 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/multicore_non_aligned_datacopypad_rmw_overlap.md`
- **V220 TBuf→MTE3 DataCopy reads stale data even after VEC write — use TQue<VECOUT> for output [V220]** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_128_v220_tbuf_mte3_datacopy_reads_stale_data_even_afte.md`
- **V220→V351 port discipline — explicit `V→MTE3` sync required between Cast/V-compute and DataCopy store; `PipeBarrier<PIPE_V>` is NOT sufficient** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_193_v220_v351_port_discipline_explicit_v_mte3_sync_req.md`
- **ReduceSum复用源改写computeBuf致脏读** · 186 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/reducesum_isreusesource_corrupts_source_buffer.md`
- **单标量输出向量链丢值：回读标量侧再写** · 211 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scalar_output_v_mte3_loss_readback.md`
- **标量 S pipe 误用 TQue 导致输出全常数** · 308 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/scalar_s_pipe_requires_explicit_mte_dependencies.md`
- **共享 UB 跨流水线复用须按方向显式同步** · 352 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/shared_ub_cross_pipeline_per_direction_sync.md`
- **AR 输出张量未初始化导致 Softmax 部分用例错** · 237 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/softmax_ar_output_buffer_must_be_initialized_from_input.md`
- **TBuf 跨 MTE2 与 Vector 须队列交接** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/tbuf_mte2_to_vector_requires_queue_handoff.md`
- **V351 DataCopyPad 尾 tile 按 32B 上取整多读，读到的是下一行真实数据而非零区，污染真实输出列** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_datacopypad_32b_roundup_oob_001.md`
- **V351 DataCopyPad（DataCopyExtParams）GM 侧 srcStride/dstStride 单位是 1 字节，不是 32B 块——A3 习惯的 /32 写法在 A5 上静默错位读取** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_datacopypad_extparams_stride_byte_unit_001.md`
- **【阴性结果】V351 fp32 hi/lo 3-MMA GEMM '缺 MTE1_M/M_MTE1 硬事件围栏'假设——管线级实测证伪，勿再作为首选假设** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_fp32_hilo_3mma_mte1_m_fence_001.md`
- **V351 累加循环'Add(ubAcc,ubAcc,ubTmp) 后无屏障、下轮迭代 DataCopyPad 覆写 ubTmp'的 V→MTE2 跨迭代竞争 → 输出 run 间非确定（max_abs 可达 O(1)）；修复 = 循环末尾 PipeBarrier<PIPE_ALL>；诊断法 = 跨 run 确定性二分** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_loop_carried_v_mte2_accum_race_001.md`
- **Vector 结果搬出前须经 VECOUT 队列** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vector_results_need_vecout_queue_before_mte3.md`
- **fp16 aUbFactor 被 cacheBuf 钳制，路由随 B 翻转** · 157 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/aub_factor_cachebuf_clamp_flips_base_group_route.md`
- **catlass Resource 内存模型与 event-ID 规则：一核一 Resource、GetBufferByByte 子分配、intra-core 同步用 AllocEventID** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/catlass_resource_memory_model_event_ids.md`
- **循环复用 UB buffer 跨迭代 MTE2/V 竞态** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/cross_iteration_ub_reuse_mte2_vf_race.md`
- **GEIR aivec 95/263：网格与缓存修复后仍全败** · 102 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/geir_aivec_95_persists_after_grid_cache_fixes.md`
- **LD_PRELOAD libopapi 致 GE TBE 初始化失败** · 124 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/ld_preload_libopapi_e40021_tbe_init.md`
- **507035 MTE DDR 越界：workspace 须含 16MB 保留区** · 264 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/mte_ddr_507035_workspace_reserved_region.md`
- **Buffer 编程模型（bl.alloc / to_buffer / to_tensor / subview）** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_buffer_model.md`
- **内存层次详解** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/memory_hierarchy.md`
- **Pipeline 执行模型** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/pipeline_execution_model.md`
- **Ascend 特有 Pass** · 121 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/ascend_passes.md`
- **前端 → HFusion 转换** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/frontend_to_hfusion.md`
- **HFusion → HIVM 转换** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hfusion_to_hivm.md`
- **HFusion 变换 Pass** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hfusion_transforms.md`
- **HIVM → 后端转换** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hivm_to_backend.md`
- **HIVM 变换 Pass** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hivm_transforms.md`
- **Linalg IR → 设备二进制** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/linalg_to_binary.md`
- **编译流水线总览** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/overview.md`
- **Pass 依赖关系与执行顺序** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/pass_dependencies.md`
- **Pipeline 选项参考** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/pipeline_options.md`
- **Triton-Ascend 编译流程主线** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_compile_flow.md`
- **Triton-Ascend 编译选项与环境变量** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_compile_options.md`
- **Triton IR 编译路径** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_ir_compilation.md`

下一页：page-010.md
