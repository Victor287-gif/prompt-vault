# other-operator-knowledge: 1051–1120 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **port_a3 complexity routing — algorithmic class (not surface LOC/matmul/CrossCore count) decides L4; flat_quant anchors the L2/L3-shippable band** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_185_flat_quant_l2_l3_calibration_anchor.md`
- **A reference feature gated on an OPTIONAL input tensor is inert when that input is absent — the benchmark's input config, not the op's documented capability, defines the truth** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_202_optional_input_gated_feature_inert_when_absent.md`
- **Prefer the managed cross-core abstraction for a MIX cube↔vec result handshake — hand-rolling exposes uncovered sub-cases one at a time** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_206_prefer_managed_cross_core_handshake_abstraction.md`
- **A non-deterministic reference that RETURNS its RNG state is bit-exact verifiable by capture-and-replay — "non-deterministic" ≠ "unverifiable"** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_217_rng_state_capture_replay_verification.md`
- **Partial-vector alignment — use AlignUp for VEC accumulation while restricting GM access to the actual element count for sub-vector-width dimensions** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_221_partial_vector_alignup_restrict_gm_access.md`
- **Optional-operand kernels collapse to ONE full-form template by substituting the operand's arithmetic-identity value, not by specializing per presence-combination** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_237_optional_operand_identity_substitution.md`
- **Carried op_host tiling that #includes op_kernel/arch35/* structs stays regbase-deliverable by inlining the host/device CONTRACT into a self-contained op_host header** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_239_inline_host_device_contract_arch35_wrap_cheat.md`
- **Global pooling (output_size=1) catastrophic performance — O(winVol) per-output-point vs CANN's O(1) integral-image approach** · 60 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_241_global_pooling_integral_image.md`
- **Offloading a scan's outer-product build or contraction to the IDLE cube unit LOSES at small contraction dim (~N=16) — the 16^3 min tile cannot be filled and the per-IterateAll fixed floor dwarfs the v** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_246_cube_offload_loses_small_contraction_dim.md`
- **Loop reordering for pooling — traverse (id,ih) once per (n,od,oh) group, not once per output point** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_249_pooling_loop_reordering_per_group_traversal.md`
- **Launch a SIMT VF_CALL from inside a SIMD KERNEL_TYPE_AIV_ONLY entry point via GetPhyAddr pointer bridge** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_251_simt_vf_call_from_aiv_only_simd_entry.md`
- **Cap scan/recurrence chunk length CH = min(L, LN_CAP/N) so the per-chunk UB working set stays bounded as state width N grows** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_253_n_adaptive_scan_chunk_length_cap.md`
- **Default to GetBlockIdx() multi-core outer_blocks partitioning for partitionable elementwise/fused ops, not NBLK=1** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_254_multi_core_outer_blocks_partitioning.md`
- **Zero-copy strided split-input access — offset into the contiguous input in-kernel instead of pybind narrow+contiguous** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_255_zero_copy_strided_split_input_access.md`
- **TQue QBUF_DEPTH=2 is the default for SIMD kernels with ≥2 GM reads per row/tile — hides MTE latency** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_258_tque_qbuf_depth_2_default_simd.md`
- **Single-argmax output is invariant to order-preserving pre-filters — collapse the sampling pipeline to one ReduceMax** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_260_single_argmax_invariant_prefilters.md`
- **Source CANN set_env.sh outside shell strict mode, then validate the environment** · 42 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_284_cann_set_env_strict_mode_wrapper.md`
- **Recompute contiguous strides after leading-one rank padding** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_285_recompute_strides_after_rank_padding.md`
- **Precompute permuted source strides on host** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_286_host_precompute_permuted_source_strides.md`
- **Form a safe GM address before any potentially out-of-bounds load** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_287_safe_address_before_oob_load.md`
- **Deterministic variable-size output uses count, exclusive offsets, then ordered emit** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_288_deterministic_variable_output_compaction.md`
- **Verify nested outputs structurally and require every leaf to pass** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_289_verify_nested_outputs_by_leaf.md`
- **Short-circuit exact non-finite equality before tolerance arithmetic** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_290_nonfinite_aware_comparator.md`
- **Duplicate-index CPU scatter may be schedule-dependent and is not unquestioned truth** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_291_duplicate_index_cpu_scatter_reference.md`
- **Probe option-sensitive CPU and NPU reference semantics before deriving the formula** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_292_probe_option_sensitive_reference_semantics.md`
- **Cumsum/Scan is inherently sequential — SIMD for I/O only** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_32_cumsum_scan_sequential_simd_io_only.md`
- **Insertion sort is O(N²) — too slow for sort_len > 4K** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_34_insertion_sort_on2_too_slow_large.md`
- **Data-movement ops — use the VECIN→VECOUT bridge pattern** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_37_data_movement_vecin_vecout_bridge.md`
- **N-dim concat decomposes to 3D (outer × cat_dim × inner)** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_38_ndim_concat_decompose_3d.md`
- **Radix sort fails on AscendC; use counting sort** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_3_radix_sort_fails_use_counting_sort.md`
- **SIMT gather achieves ~1.0x vs CANN for random-access patterns** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_40_simt_gather_random_access.md`
- **Pure-copy ops — TQueBind replaces the VECIN→VECOUT bridge** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_49_tquebind_pure_copy_ops.md`
- **Vector Counter mode replaces Normal mode to simplify tail handling** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_51_counter_mode_simplifies_tail_handling.md`
- **Permute — identity shortcut is the largest optimization lever** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_55_permute_identity_shortcut.md`
- **Sort algorithm selection — prefer the hardware Sort API** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_57_sort_prefer_hardware_sort_api.md`
- **SIMD/SIMT mode selection — scatter ops tail-dimension threshold** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_59_simd_simt_scatter_tail_dim_threshold.md`
- **Slice-D workspace — general splitting scheme for large-hidden-dim ops** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_60_slice_d_workspace_large_hidden_dim.md`
- **Pooling multi-strategy priority dispatch + UB shrink loop** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_61_pooling_priority_dispatch_ub_shrink.md`
- **GatherElements — compile-time axis specialization + axis merging** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_62_gather_elements_axis_specialization_merge.md`
- **Elementwise ops — tile-first UB allocation, queue-depth-second** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_63_elementwise_tile_first_queue_depth.md`
- **Hand-written math approximations can beat built-in functions** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_64_handwritten_math_beats_builtin.md`
- **Histogram two-phase int32 atomicAdd pattern** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_67_histogram_two_phase_int32_atomicadd.md`
- **Multi-input fused ops use independent TQues to avoid PB-9** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_71_multi_input_fused_independent_tques.md`
- **Identity-gradient outputs — use pybind clone() instead of a kernel** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_73_identity_gradient_outputs_pybind_clone.md`
- **Using TBuf for loop-persistent state is safe (does not contradict PB-11)** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_78_tbuf_loop_persistent_state_safe.md`
- **Multi-instance partition-dispatch when output rows partition by a host-knowable boundary** · 53 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/ol_93_multi_instance_partition_dispatch.md`
- **Boundary-clamped interpolation — a single OOB→0 gather reproduces zeros/border/reflection padding because the clamped coordinate's overflow neighbor always carries zero interpolation weight** · 58 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p105_boundary_clamped_interpolation_a_single_oob_0_ga.md`
- **Serial-in-L linear recurrence on AscendC — L-chunk to bound UB + Hillis-Steele parallel associative scan over the affine pair, with an L-major `[l*N+n]` layout so one L-shift is one wide Mul/Add acros** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p106_serial_in_l_linear_recurrence_on_ascendc_l_chunk.md`
- **Iterative top-K / argmax selection from a GM-resident buffer — valid ONLY for bounded top-K (k≪V); ANTI-PATTERN for nucleus/top-P (O(V²))** · 62 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p108_iterative_top_k_argmax_selection_from_a_gm_resid.md`
- **Hardware Sort + cumsum nucleus finding (O(V log²V)) for top-P / Branch-C sampling — the correct replacement for P-P108's O(V²) iterative selection** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p109_hardware_sort_cumsum_nucleus_finding_o_v_log_v_f.md`
- **Block Oversubscription** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p10_block_oversubscription.md`
- **Multi-core outer_blocks partitioning template (elementwise/fused)** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p114_multi_core_outer_blocks_partitioning_template_el.md`
- **Zero-copy strided split-input kernel template (chunk-then-compute)** · 64 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p115_zero_copy_strided_split_input_kernel_template_ch.md`
- **Adaptive tile size** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p11_adaptive_tile_size.md`
- **Multi-phase op via inheritance chain — phases share state via class-hierarchy, dispatch by TILING_KEY in same .so** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p122_multi_phase_op_via_inheritance_chain_phases_sh.md`
- **int32 for inner-loop counters instead of int64** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p12_int32_for_inner_loop_counters_instead_of_int64.md`
- **Cooperative-group parallel traversal + shuffle reduction** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p13_cooperative_group_parallel_traversal_shuffle_red.md`
- **Cooperative-group value transport** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p16_cooperative_group_value_transport.md`
- **Prefix-sum + block-level atomicAdd aggregation** · 28 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p17_prefix_sum_block_level_atomicadd_aggregation.md`
- **Kernel development must include UT (boundary dim + sorted/original consistency)** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p19_kernel_development_must_include_ut_boundary_dim.md`
- **Dynamic numBlocks** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p1_dynamic_numblocks.md`
- **Thread utilization rebalancing (Thread Utilization Analysis)** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p20_thread_utilization_rebalancing_thread_utilizatio.md`
- **Persistent-core dispatch (Persistent Kernel)** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p22_persistent_core_dispatch_persistent_kernel.md`
- **Contiguous-block partition vs grid-stride (adjacent-element comparison scenario)** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p23_contiguous_block_partition_vs_grid_stride_adjace.md`
- **Sort-to-Reuse — eliminate GM read amplification from indirect addressing** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p24_sort_to_reuse_eliminate_gm_read_amplification_fr.md`
- **Sorted-Edge First-Occurrence Dedup (atomicCAS-Free)** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p32_sorted_edge_first_occurrence_dedup_atomiccas_fre.md`
- **SIMT per-element indirect gather** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p34_simt_per_element_indirect_gather.md`
- **bf16 direct assign in SIMT (no Cast needed)** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p35_bf16_direct_assign_in_simt_no_cast_needed.md`
- **TQueBind for pure data-movement ops** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p36_tquebind_for_pure_data_movement_ops.md`
- **Vector Counter mode for auto tail handling** · 23 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/optimizations/p_p38_vector_counter_mode_for_auto_tail_handling.md`

下一页：page-017.md
