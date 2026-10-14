# pipeline-memory: 631–700 / 731

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **Triton-Ascend 编译流水线阶段划分** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_pipeline_overview.md`
- **TritonToLinalg 转换** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_to_linalg.md`
- **AST → TTIR 生成** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/ttir_generation.md`
- **TTIR 优化 Pass** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/ttir_optimization.md`
- **HFusion 内存操作** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hfusion/memory_ops.md`
- **Bufferization — Tensor 到 MemRef 的转换** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/bufferization.md`
- **额外缓冲区 — Extra Buffer** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/extra_buffer.md`
- **内存对齐 — Memory Alignment** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/memory_alignment.md`
- **内存规划 — PlanMemory Pass** · 80 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/memory_planning.md`
- **多缓冲区 — Multi-Buffer** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/multi_buffer.md`
- **内存管理子系统总览** · 80 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/overview.md`
- **紧耦合缓冲区 — Tightly Coupled Buffer** · 66 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/tightly_coupled_buffer.md`
- **工作空间管理 — Workspace Management** · 78 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/workspace_management.md`
- **内存操作 API（tl.load / tl.store / block_ptr）** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_memory_ops.md`
- **paged_memory_copy 类算子设计约束（Paged Memory Copy）** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/paged_memory_copy_constraints.md`
- **Paged KV Cache Attention 算子（decode / prefill） 实现手册** · 230 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/paged_attention__paged_kv_cache_attention.md`
- **StorePagedKVCache 算子（Scatter-Copy） 实现手册** · 106 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/paged_memory_copy__store_paged_kv_cache.md`
- **Matmul 链中间 buffer dtype 优化** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/chained_matmul_buffer_dtype.md`
- **Cube/MTE3 分阶段批量解耦优化** · 463 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/cube_mte3_decoupling.md`
- **离散访存优化模式** · 270 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/discrete_memory_access.md`
- **内存访问优化实践** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/memory_access_practice.md`
- **Multibuffer / 双缓冲优化（通信-计算重叠）** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/multibuffer_and_double_buffering.md`
- **Bank Conflict 消除** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/bank_conflict.md`
- **CV 流水线优化（Cube-Vector 融合）** · 176 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/cv_pipeline_optimization.md`
- **内存访问优化** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/memory_access_optimization.md`
- **Double Buffering Pattern** · 81 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/optimization_patterns/double_buffering.md`
- **Memory Coalescing Pattern** · 84 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/optimization_patterns/memory_coalescing.md`
- **Pipeline Overlap Optimization** · 96 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/optimization_patterns/pipeline_overlap.md`
- **P133_ssm_stage1_cv_pipeline** · 24 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/preconditions/P133_ssm_stage1_cv_pipeline.yaml`
- **多输出归约的 Buffer 规划** · 115 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/multi-output-buffer.md`
- **UB 缓冲区管理指南** · 364 行 · upstream
  `cannbot-skills/ops/ascendc-api-best-practices/references/api-buffer.md`
- **DataCopy / DataCopyPad 使用指南** · 405 行 · upstream
  `cannbot-skills/ops/ascendc-api-best-practices/references/api-datacopy.md`
- **流水线同步机制指南** · 206 行 · upstream
  `cannbot-skills/ops/ascendc-api-best-practices/references/api-pipeline.md`
- **MTE 通信窗口地址获取方法** · 374 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/mte-address-access.md`
- **mte_dispatch_comm** · 324 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/mte_dispatch_comm.h`
- **双缓冲四区轮转协议** · 69 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/double-buffer-protocol.md`
- **MoE Dispatch/Combine 窗口内存布局** · 199 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/window-memory-layout.md`
- **mc2_matmul_block_l2cache** · 165 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/all_gather_matmul/x_all_gather_matmul/mc2_common/op_kernel/mc2_matmul_block_l2cache.h`
- **hccl-matmul 通算并行调优（tileCnt，910B / A2）** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/pipeline_tuning.md`
- **通算并行调优（tileCnt / headMSize）** · 232 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/shared/pipeline_tuning.md`
- **流水配平分析框架** · 491 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/comm-compute/pipeline_balancing.md`
- **访存 Bound 优化策略** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/single-core-pipeline/memory.md`
- **无 Bound 优化策略** · 66 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/single-core-pipeline/no-bound.md`
- **Scalar Bound / 小 case 优化策略** · 29 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/single-core-pipeline/scalar.md`
- **VEC Bound 优化策略** · 164 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/single-core-pipeline/vec.md`
- **实现②：DataCopyPad 紧凑搬入（外层广播，offset 寻址）** · 46 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/broadcast/datacopypad_design.md`
- **DataCopy Optimization 优化设计** · 351 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/common/datacopy_optimization_design.md`
- **UB/TBuf 常驻复用与 Bank 冲突规避优化设计** · 342 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/common/ub_resident_design.md`
- **Double Buffering / 双缓冲流水线优化设计** · 288 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/elementwise/double_buffer_design.md`
- **matmul MTE2 预加载（MTE2 Preload）优化设计** · 320 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/mte2_preload_design.md`
- **小 M、宽 N Matmul：从载入量出发选择分块与流水** · 137 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/matmul/small_m_wide_n_pipeline_design.md`
- **MC² 流水配平优化设计** · 502 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/mc2/pipeline_balancing_design.md`
- **融合算子 UB 状态 Buffer 跨循环常驻优化设计** · 337 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/reduce/templates/state_resident_design.md`
- **RegBase DataCopy 分层** · 139 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/datacopy_best_practices.md`
- **Pipeline 与 Buffer** · 105 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/pipeline_and_buffer.md`
- **broadcast 范式 standard DAG 分析和 Buffer 划分** · 285 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-dag-buffers.md`
- **DataCopyPad 规则** · 185 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/datacopypad-rules.md`
- **reduction 范式 binary-base DAG 分析和 Buffer 划分** · 612 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/reduction/references/reduction-template-binary-base/reduction-binary-base-dag-buffers.md`
- **reduction 范式 binary-group DAG 分析和 Buffer 划分** · 209 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/reduction/references/reduction-template-binary-group/reduction-binary-group-dag-buffers.md`
- **reduction 范式 empty DAG 分析和 Buffer 划分** · 292 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/reduction/references/reduction-template-empty/reduction-empty-dag-buffers.md`
- **bm_good_a_standard_pipeline** · 57 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_a_standard_pipeline.asc`
- **bm_good_b_double_buffer_pingpong** · 49 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/correct/bm_good_b_double_buffer_pingpong.asc`
- **bm_sync02_datacopy_then_adds** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync02_datacopy_then_adds.asc`
- **bm_sync06_pipebarrier_v_then_datacopy** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-sync-audit/benchmark/defects/bm_sync06_pipebarrier_v_then_datacopy.asc`
- **多输出归约的 Buffer 规划** · 115 行 · upstream
  `cannbot-skills/ops/ascendc-tiling-design/references/reduction/multi-output-buffer.md`
- **Channel 与 Buffer：明确槽位选择和实际访问** · 36 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/channel-buffer.md`
- **流水时序与同步契约** · 84 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-op-design/references/pipeline-design.md`
- **DataCopy Best Practices** · 87 行 · upstream
  `cannbot-skills/ops/ops-design-paradigms/references/knowledge/api/datacopy_best_practices.md`
- **Pipeline and Buffer** · 87 行 · upstream
  `cannbot-skills/ops/ops-design-paradigms/references/knowledge/api/pipeline_and_buffer.md`
- **上下文恢复与验证要点** · 54 行 · upstream
  `cannbot-skills/ops/ops-design-paradigms/references/knowledge/dev-experience/process_memory_cards.md`

下一页：page-011.md
