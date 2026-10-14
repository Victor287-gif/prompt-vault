# sync-scalar: 561–630 / 674

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **共享 UB 跨流水线复用须按方向显式同步** · 352 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/shared_ub_cross_pipeline_per_direction_sync.md`
- **骨架占位输出全 0：标量直算先打通正确性** · 244 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/skeleton_placeholder_output_zero_scalar_conv.md`
- **SyncAll 修复尾元素丢写无效反致误差恶化** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/syncall_not_fix_scalar_lost_write.md`
- **半精度 shadow 跨流竞态 — 转换与 kernel 两侧做设备同步** · 384 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_npu_async_temporaries_need_completion_before_raw_kernel.md`
- **有数据依赖的 VF 调用间仍需 PipeBarrier** · 92 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ub_dependent_vf_calls_keep_pipebarrier.md`
- **向量Compare/Select小A0结果错，改标量扫描** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vector_compare_select_small_a0_scalar_rewrite.md`
- **VF 归约落 UB 后 GetValue 读取陈旧标量** · 448 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vector_scalar_ub_handoff_requires_explicit_completion.md`
- **507035：多核 reduce SyncAll 回退单核** · 171 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/507035_syncall_multicore_reduce.md`
- **arch35 gather kernel TBuf 手动同步完整实现** · 409 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/arch35_gather_tbuf_manual_sync_recipe.md`
- **catlass Resource 内存模型与 event-ID 规则：一核一 Resource、GetBufferByByte 子分配、intra-core 同步用 AllocEventID** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/catlass_resource_memory_model_event_ids.md`
- **带标量属性 elementwise 算子 tiling 骨架配方** · 545 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/elementwise_scalar_attr_tiling_skeleton.md`
- **Softmax 大 R 分核：Phase1+SyncAll+Phase2 合并** · 336 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/softmax_group_2d_split_phase1_syncall_phase2.md`
- **SyncAll 须配 tiling SetScheduleMode 批调度** · 183 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/syncall_requires_batch_schedule_mode.md`
- **aclnn内部sync吞错误致ST假通过** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_internal_sync_false_pass.md`
- **tiling UT 标量输入须带 isConst/constValue** · 127 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/tiling_ut_scalar_const_value_injection.md`
- **PyTorch 接入 TensorList+scalar 算子 ST 工程** · 348 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/torch_tensorlist_scalar_st_adapter.md`
- **TTK warmup 编译失败须 --warmup false（0 无效）** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_warmup_compile_failure_disable_flag.md`
- **白盒 rank0-scalar 用例须补 expect_error** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_rank0_scalar_case_expect_error.md`
- **sync_block_set / wait / all 同步操作** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_sync.md`
- **HIVM 跨核同步操作（sync_block 系）** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/sync_block_sync.md`
- **HIVM 同步体系总览** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/sync_overview.md`
- **HIVM 管道同步操作（set_flag / wait_flag / pipe_barrier）** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/sync_pipe_sync.md`
- **HIVM 同步注入 Pass（InjectSync / GraphSyncSolver）** · 101 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/sync_sync_injection.md`
- **HIVM UnitFlag 同步机制** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/sync_unit_flag.md`
- **HIVM 向量操作标量降级** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/vector_scalar_lowering.md`
- **标量降级** · 208 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/avoid_scalar_lowering.md`
- **Scalar 转 Vector 优化模式** · 479 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/scalar_to_vector.md`
- **标量降级规避（910_95 / 910B）** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/scalar_degradation_avoidance.md`
- **同步与 Barrier（NPU 三层同步机制）** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/sync_and_barrier.md`
- **标量计算向量化 (Scalar-to-Vector Conversion)** · 440 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/optimization_patterns/scalar_to_vector.md`
- **FlashAttention 类算子 — Feature Flags 框架** · 216 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/flashattention/specialization/feature_flags.md`
- **# Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 87 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/scripts/backfill_resource_flags.py`
- **AscendC 同步机制指南** · 285 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/ascendc-sync-guide.md`
- **DSL: Cube 完成所有 tile 后一次性通知 Vector** · 166 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/dsl2Ascendc_cross_core_sync.md`
- **跨核同步 API 使用指南** · 259 行 · upstream
  `cannbot-skills/ops/ascendc-api-best-practices/references/api-crosscore-sync.md`
- **cv_sync_constants** · 32 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/assets/blaze_custom/epilogue/cv_sync_constants.h`
- **Ascend C 同步指令编码指导** · 814 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/fundamentals/blaze-sync-patterns.md`
- **CV Sync 与两阶段 Kernel 编排专题** · 156 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv-sync-and-two-phase-entry.md`
- **V+C 数据流、CV 同步与 L1 布局专题** · 103 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/weight-dequant-prologue-fusion/vc-sync-and-l1-layout-design.md`
- **Sync 与可见性规则** · 105 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/sync-and-visibility.md`
- **Scalar Bound / 小 case 优化策略** · 29 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/single-core-pipeline/scalar.md`
- **Scalar 高性能编码原则** · 476 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/scalar/coding_principles.md`
- **Conv 算子 Scalar 优化** · 165 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/scalar/conv_scalar.md`
- **RegBase API 同步** · 173 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/regbase_api_sync.md`
- **同步与数据一致性** · 314 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/sync-and-consistency.md`
- **多核数据交换与同步** · 60 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/multi-core-sync.md`
- **ascendc-sync-audit benchmark 最小集** · 181 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/README.md`
- **bm_good_a_standard_pipeline** · 57 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_a_standard_pipeline.asc`
- **bm_good_b_double_buffer_pingpong** · 49 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_b_double_buffer_pingpong.asc`
- **bm_good_sync01_loop_pingpong** · 52 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_sync01_loop_pingpong.asc`
- **bm_good_sync10_signed_no_report** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_sync10_signed_no_report.asc`
- **bm_real_blas1_correct** · 22 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_real_blas1_correct.asc`
- **bm_real_blas2_correct** · 21 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_real_blas2_correct.asc`
- **bm_sync01_wait_before_set** · 42 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync01_wait_before_set.asc`
- **bm_sync02_datacopy_then_adds** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync02_datacopy_then_adds.asc`
- **bm_sync03_single_core_residual_wait** · 39 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync03_single_core_residual_wait.asc`
- **bm_sync04_two_set_one_wait** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync04_two_set_one_wait.asc`
- **bm_sync05_conditional_pipe_wait** · 54 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync05_conditional_pipe_wait.asc`
- **bm_sync05_hardevent_typo** · 39 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync05_hardevent_typo.asc`
- **bm_sync06_pipebarrier_v_then_datacopy** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync06_pipebarrier_v_then_datacopy.asc`
- **bm_sync07_flagid_over_15** · 51 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync07_flagid_over_15.asc`
- **bm_sync08_return_skips_setflag** · 429 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync08_return_skips_setflag.asc`
- **bm_sync09_pipe_all_narrowable** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync09_pipe_all_narrowable.asc`
- **bm_sync10_unsigned_underflow** · 37 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync10_unsigned_underflow.asc`
- **bm_sync11_enque_deque_no_compute** · 42 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync11_enque_deque_no_compute.asc`
- **bm_sync12_syncall_asymmetric** · 39 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync12_syncall_asymmetric.asc`
- **bm_sync13_atomic_ordering** · 22 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync13_atomic_ordering.asc`
- **bm_sync14_output_bundle_index** · 41 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync14_output_bundle_index.asc`
- **# Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 251 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/eval_benchmark.py`
- **# Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 208 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/extract_change_lines.py`

下一页：page-010.md
