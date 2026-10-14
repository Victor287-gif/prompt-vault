# profiling-debug: 71–140 / 230

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **DataCopyPad UB→GM crashes on Ascend950PR (507035)** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_23_datacopypad_ub_gm_crashes_on_ascend950pr_507035.md`
- **`std::string` in pybind11 crashes with `basic_string null` SIGSEGV on V220 ARM64 (bisheng/GCC ABI mismatch)** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ec_63_std_string_in_pybind11_crashes_with_basic_string_n.md`
- **多 vendor 同名 libcust_opapi 加载源判定** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ld_debug_libcust_opapi_same_soname_multi_vendor.md`
- **CANN typed kernel entry _fp32 crashes with error 507035** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_12_cann_typed_kernel_entry_fp32_crashes_with_error_50.md`
- **SIMT 1024 threads crashes with 507035 on Ascend950PR** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/ol_56_simt_1024_threads_crashes_with_507035_on_ascend950.md`
- **msprof_perf_summary.py `_jsonl_scalar_value` crashes on attr inputs declared `dtype:"int"` whose value is a LIST — canonical quick run dies at the first such case [shared perf-infra, input-space gap]** · 25 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_179_msprof_perf_summary_py_jsonl_scalar_value_crashe.md`
- **msprof_perf_summary.py tensor random-synth crashes on `shape: null` optional tensor inputs — null shape must emit `None` (ABSENT), not a synthesized shape [shared perf-infra, input-space gap]** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_182_msprof_perf_summary_py_tensor_random_synth_crash.md`
- **msprof / torch_npu.profiler ANALYZE stage fails `Operation not permitted` (EPERM) in the `.171` npu_dev3 container despite Privileged=true → use NPU `aclrtEvent` elapsed_time device-time as the fallba** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_51_msprof_torch_npu_profiler_analyze_stage_fails_oper.md`
- **CANN merge_mix_obj.sh Crash with Empty --build-type** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/pb_7_cann_merge_mix_obj_sh_crash_with_empty_build_type.md`
- **msprof带宽利用率与聚合带宽口径差约10倍** · 90 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/msprof_bw_usage_rate_vs_aggregate_caliber.md`
- **msprof op 单次仅采单 kernel，按场景分跑** · 533 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/msprof_op_single_kernel_per_scene.md`
- **Always msprof the reference op BEFORE writing structural-impossibility framing — meta-process for perf claims** · 89 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_117_always_msprof_the_reference_op_before_writing_stru.md`
- **msprof vec_ratio ≠ atomicAdd share (SIMT scalar GM reads also go through the VEC pipe)** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_20_msprof_vec_ratio_atomicadd_share_simt_scalar_gm_re.md`
- **a profiler-launched benchmark binds its NPU at runtime init, BEFORE any wrapper-level env var applies — export** · 67 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_557_a_profiler_launched_benchmark_binds_its_npu_at_r.md`
- **An offline numerics simulation validates only the arithmetic layer it re-implements — storage/transfer-layer kernel defects are structurally invisible to it, so a green offline proof is zero evidence ** · 27 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_563_an_offline_numerics_simulation_validates_only_th.md`
- **msprof pipe-ratio metrics (PipeUtilization op-summary / sample-based ai_core_utilization) are UNAVAILABLE for raw-launch (aclrtBinary) kernels on this box — all-zero rows** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_678_msprof_pipe_ratio_metrics_pipeutilization_op_sum.md`
- **msprof scalar_ratio > 0.9 = scalar pipe saturated → replace with VEC instructions** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/performance/ol_82_msprof_scalar_ratio_0_9_scalar_pipe_saturated_repl.md`
- **fp16小值域MARE超标：golden未模拟存储精度** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/fp16_subnormal_golden_simulate_storage.md`
- **A prose/simulator invariant divergence is itself a defect carrier — regenerate prose invariants from the simulator's formula** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/ol_565_a_prose_simulator_invariant_divergence_is_itself.md`
- **GE 图执行 0x50100064：TBE 子进程 Python3.13 崩溃** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/geir_0x50100064_tbe_python313_crash.md`
- **rank>8 张量 ACL 抛 length_error 崩溃整跑** · 241 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/aclnn_rank_gt8_std_length_error_crash.md`
- **OOM 插桩是编译期：release 套件 OOB=UNKNOWN 属预期** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/oom_debug_config_compile_time_instrumentation.md`
- **UT 多文件同名参数结构体致 gtest 静态初始化段错误** · 188 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/ut_gtest_duplicate_param_struct_static_init_crash.md`
- **编译错误排查（MLIRCompileError）** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/compile_error_debug.md`
- **调试环境变量完整参考** · 116 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debug_env_variables.md`
- **Triton-Ascend 调试方法总览** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debug_overview.md`
- **调试、稳定性与 API 规范** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debugging.md`
- **常见编译/运行错误速查（报错-原因-修复）** · 77 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/debugging_common_errors.md`
- **Profiling 方法论** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/profiling.md`
- **Triton-Ascend 性能分析与瓶颈定位指南** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/profiling_guide.md`
- **运行时错误排查** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/runtime_error_debug.md`
- **调试检查清单** · 93 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/debug_checklist.md`
- **Triton-Ascend 验证与调试工作流** · 131 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/workflow_and_debugging.md`
- **Profiling Reference — Stage 2 LLM 诊断参考材料** · 85 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/profiling_reference/INDEX.md`
- **msprof op CSV 字段完整参考** · 260 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/profiling_reference/csv_fields_reference.md`
- **瓶颈优化速查表** · 265 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/profiling_reference/optimization_quickref.md`
- **# Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 965 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-world-model/scripts/profiling_evidence.py`
- **复杂 Cube+Vector 算子调试流程** · 298 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-precision-tuning/references/debug-workflow-complex.md`
- **简单 Vector 算子调试流程** · 87 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-precision-tuning/references/debug-workflow-simple.md`
- **SplitM Elementwise/Broadcast Epilogue 契约与定位** · 86 行 · upstream
  `cannbot-skills/ops/ascendc-blaze-best-practice/references/scenarios/elementwise-broadcast-epilogue-fusion/splitm-contract-and-debugging.md`
- **卡死/崩溃调试工作流程** · 193 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/references/crash_workflow.md`
- **mssanitizer 内存检测使用指南** · 137 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/references/memcheck/README.md`
- **AscendC Memcheck 自动化工作流指南** · 418 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/references/memcheck/automated_workflow.md`
- **msSanitizer 内存检测工具指南** · 181 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/references/memcheck/mssanitizer_guide.md`
- **memcheck_input.json** · 44 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/scripts/memcheck_input.json.template`
- **-*- coding: UTF-8 -*-** · 191 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/scripts/parse_plog.py`
- **----------------------------------------------------------------------------------------------------------** · 696 行 · upstream
  `cannbot-skills/ops/ascendc-crash-debug/scripts/run_memcheck_pre.sh`
- **Cube 类算子迁移：测试与 debug 排查手册** · 73 行 · upstream
  `cannbot-skills/ops/ascendc-cross-gen-migration/routes/2201-to-3510/references/cube-debug-lessons.md`
- **性能采集（msprof + L2 cache flush）参考** · 278 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/shared/profiling_mc2.md`
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
- **运行时错误调试工作流程** · 324 行 · upstream
  `cannbot-skills/ops/ascendc-runtime-debug/references/debug_workflow.md`
- **aclnn API 返回码完整表** · 102 行 · upstream
  `cannbot-skills/ops/ascendc-runtime-debug/references/error_codes.md`
- **Kernel 二进制与构建系统调试** · 265 行 · upstream
  `cannbot-skills/ops/ascendc-runtime-debug/references/kernel_binary_debug.md`
- **-*- coding: UTF-8 -*-** · 161 行 · upstream
  `cannbot-skills/ops/ascendc-runtime-debug/scripts/parse_plog.py`
- **调测接口 API 参考** · 528 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/debug-api.md`
- **采集工作流与命令规划** · 91 行 · upstream
  `cannbot-skills/ops/ops-profiling/msopprof-visualization/references/collection-workflow.md`

下一页：page-003.md
