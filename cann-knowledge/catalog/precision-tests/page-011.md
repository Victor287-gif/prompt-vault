# precision-tests: 701–770 / 893

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Two-Product split 遇 inf 产 NaN 破坏 inf 传播** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/two_product_split_inf_nan_pollution.md`
- **有数据依赖的 VF 调用间仍需 PipeBarrier** · 92 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ub_dependent_vf_calls_keep_pipebarrier.md`
- **UDMA GET size=0 污染后续 GET 致精度失败** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/udma_get_size_zero_pollutes_engine_guard.md`
- **Uint32ToFloat 须用 IEEE754 位操作而非位移乘法** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/uint32tofloat_ieee754.md`
- **unroll s2 内层循环破坏 flash 在线 softmax 循环态** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/unroll_s2_flash_softmax.md`
- **VF UpdateMask 手动递减移除致输出全零** · 103 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/updatemask_manual_decrement_required_despite_doc.md`
- **upsample特化路径VECIN直拷GM致TilingKey=2全失败** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/upsample_specialization_vecin_gm_regression.md`
- **V351 AIV TQue depth-2 静默损坏** · 46 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_aiv_tque_depth2_001.md`
- **static Broadcast<T,N,axis,false>(...,workspace) sharing a workspace with RA-form ReduceSum undersizes it on V351 — switch to dynamic GetBroadcastTilingInfo API** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_broadcast_static_workspace_001.md`
- **V351 DataCopyPad 尾 tile 按 32B 上取整多读，读到的是下一行真实数据而非零区，污染真实输出列** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_datacopypad_32b_roundup_oob_001.md`
- **V351 DataCopyPad（DataCopyExtParams）GM 侧 srcStride/dstStride 单位是 1 字节，不是 32B 块——A3 习惯的 /32 写法在 A5 上静默错位读取** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_datacopypad_extparams_stride_byte_unit_001.md`
- **bare AscendC::Exp() defaults to ExpAlgo::INTRINSIC whose input-dependent error pushes softmax past frozen-case rtol on V351 — pass ExpConfig{PRECISION_1ULP_FTZ_TRUE} explicitly** · 49 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_exp_intrinsic_precision_001.md`
- **【阴性结果】V351 fp32 hi/lo 3-MMA GEMM '缺 MTE1_M/M_MTE1 硬事件围栏'假设——管线级实测证伪，勿再作为首选假设** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_fp32_hilo_3mma_mte1_m_fence_001.md`
- **V351 累加循环'Add(ubAcc,ubAcc,ubTmp) 后无屏障、下轮迭代 DataCopyPad 覆写 ubTmp'的 V→MTE2 跨迭代竞争 → 输出 run 间非确定（max_abs 可达 O(1)）；修复 = 循环末尾 PipeBarrier<PIPE_ALL>；诊断法 = 跨 run 确定性二分** · 57 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_loop_carried_v_mte2_accum_race_001.md`
- **V351 纯 PipeBarrier<PIPE_ALL> + TBuf 不排序跨 pipe** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_pipe_all_tbuf_stale_001.md`
- **reference/candidate weight seed via hash(torch.dtype) is address-based and process-unstable — golden/fixture and eval-process weights diverge** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_reference_weight_seed_nondeterminism_001.md`
- **向量Compare/Select小A0结果错，改标量扫描** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vector_compare_select_small_a0_scalar_rewrite.md`
- **Vector 结果搬出前须经 VECOUT 队列** · 177 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vector_results_need_vecout_queue_before_mte3.md`
- **VF 归约落 UB 后 GetValue 读取陈旧标量** · 448 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vector_scalar_ub_handoff_requires_explicit_completion.md`
- **bf16 输出 NaN 载荷 0x7FFF 需归一化为 0x7FC0** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vf_cast_bf16_nan_payload_0x7fff_vs_golden_7fc0_normalize.md`
- **双 VF 融合输出非预期 先排 CopyOut slot 勿用 PRINTF** · 120 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vf_fusion_output_unexpected_check_copyout_slot_not_printf.md`
- **VF乘法原位覆盖源buffer致bitwise失配** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vf_mul_inplace_dst_overwrites_source_bitwise_mismatch.md`
- **白盒 kernel 模式精度失配与 ST 黑盒通过并存** · 81 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/whitebox_kernel_mode_mismatch_aclnn_pass.md`
- **分块 wrapper 与 VF callee 不可重复叠加 offset** · 173 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/wrapper_and_vf_callee_must_not_double_apply_chunk_offset.md`
- **输出 z NZ 物理布局致 dim=1 间歇错位 banned** · 105 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/z_nz_dim_1_banned.md`
- **上板精度二分定位：恒等搬运+离线数值复现** · 204 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/precision_bisection_identity_copy_offline_repro.md`
- **GEIR 多输出精度串无法过 verify 严格 PASS 判定** · 182 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ttk_geir_verify_multi_output_empty_cst_precision.md`
- **白盒生成输出形状全错：golden 探测静默回退 shape_rule** · 448 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_gen_shape_rule_fallback_wrong_shapes.md`
- **白盒生成器 int 属性自动补 range(5) 致 step=0 用例** · 132 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_int_attr_range5_sampling.md`
- **白盒非默认 mode 分支结构覆盖塌缩：mode_struct_k 恢复** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_nondefault_mode_structural_collapse.md`
- **白盒 rank0-scalar 用例须补 expect_error** · 134 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_rank0_scalar_case_expect_error.md`
- **白盒 semantic_boundaries 无法物化 fp16 边界用例** · 94 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_semantic_boundaries_dtype_gap.md`
- **白盒 special 用例 attrs 空致误路由** · 179 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/whitebox_special_cases_empty_attrs_misdispatch.md`
- **FA 精度闸门与测量口径** · 150 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/flash_attention__precision_gate_and_measurement.md`
- **数据类型与精度保护（910_95 平台）** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/data_type_precision.md`
- **DumpTensor API Reference** · 53 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-precision-tuning/references/api-reference.md`
- **复杂 Cube+Vector 算子调试流程** · 298 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-precision-tuning/references/debug-workflow-complex.md`
- **简单 Vector 算子调试流程** · 87 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-precision-tuning/references/debug-workflow-simple.md`
- **错误模式参考** · 38 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-precision-tuning/references/error-patterns.md`
- **Precision Patterns for Pooling Operators** · 169 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/precision-patterns.md`
- **精度转换与混合精度指南** · 146 行 · upstream
  `cannbot-skills/ops/ascendc-api-best-practices/references/api-precision.md`
- **GLU 精度诊断** · 113 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/kernel-design/glu-precision-diagnosis.md`
- **Elementwise/Broadcast Epilogue 精度诊断** · 86 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/precision-diagnosis.md`
- **生态算子精度标准（迁移场景接入）** · 24 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/OPS_PRECISION_STANDARDS.md`
- **aclnn 接口调用指南** · 329 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/aclnn-interface-guide.md`
- **精度测试前置数值验证指南** · 146 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/precision-test-pre-validation-guide.md`
- **{{OP_NAME}} 算子精度验证报告（ascend950）** · 58 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/precision_report_template.md`
- **自定义算子 PyTorch 绑定构建指南（torch_aclnn_helper.h + EXEC_NPU_CMD 标准方式）** · 435 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/pytorch-binding-build-guide.md`
- **-----------------------------------------------------------------------------------------------------------** · 401 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/run_precision_report_aclnn_template.py.template`
- **算子源码逆向分析方法论** · 139 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/source-code-reverse-analysis.md`
- **-----------------------------------------------------------------------------------------------------------** · 204 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/test_op_precision_aclnn_template.py.template`
- **torch_aclnn_helper.h** · 567 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/references/precision-testing/torch_aclnn_helper.h.template`
- **阶段 4：精度验证（含 Subnormal 测试）** · 341 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/routes/2201-to-3510/stages/stage_4_precision.md`
- **DumpTensor API Reference** · 92 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/ascendc-dumptensor-refs/api-reference.md`
- **DumpTensor 视角下的错误模式** · 40 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/ascendc-dumptensor-refs/error-patterns.md`
- **AscendC DumpTensor Debugging** · 205 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/ascendc-dumptensor.md`
- **精度调试最佳实践** · 317 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/best-practices.md`
- **二分调试法详细指南** · 275 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/binary-search-debug.md`
- **实战调试案例** · 459 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/case-studies.md`
- **常见精度陷阱详解** · 492 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/common-traps.md`
- **数据对比法** · 308 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/data-comparison.md`
- **精度调试总结模板** · 198 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/debug-summary-template.md`
- **精度问题诊断工作流程** · 180 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/diagnosis-workflow.md`
- **Printf 调试法** · 296 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/printf-debug.md`
- **SIMT 精度调试技巧** · 76 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/simt-precision-debug.md`
- **工具和命令参考** · 364 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/references/tools-reference.md`
- **-*- coding: UTF-8 -*-** · 148 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/scripts/error_analysis.py`
- **gen_boundary_test** · 169 行 · upstream
  `cannbot-skills/ops/ascendc-precision-debug/scripts/gen_boundary_test.py`
- **精度失败** · 51 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/pitfalls/precision_failures.md`
- **精度指南** · 85 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/pitfalls/precision_guide.md`

下一页：page-012.md
