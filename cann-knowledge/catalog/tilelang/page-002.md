# tilelang: 71–140 / 320

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **dsl2Ascendc_compute_cube** · 80 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/dsl2Ascendc_compute_cube.md`
- **dsl2Ascendc_compute_cv** · 138 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/dsl2Ascendc_compute_cv.md`
- **dsl2Ascendc_compute_vector** · 118 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/dsl2Ascendc_compute_vector.md`
- **DSL: Cube 完成所有 tile 后一次性通知 Vector** · 166 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/dsl2Ascendc_cross_core_sync.md`
- **dsl2Ascendc_host** · 107 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/dsl2Ascendc_host.md`
- **Pooling Pattern Index** · 54 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/PoolingPatternIndex.md`
- **Alignment Guards for Vectorized Pooling Kernels** · 112 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/alignment-guards.md`
- **Backward Implementation: Pooling 反向梯度 AscendC 实现（陷阱与高性能落地）** · 80 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/backward-implementation.md`
- **avg_pool3d_grad v2 补课：反向 pooling 落地的新经验（相对 v1 文档的增量）** · 209 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/grad-v2-lessons.md`
- **NDHWC 布局的 AscendC 落地（Host Permute + Kernel 行地址）** · 41 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/layout-implementation.md`
- **Anti-Patterns: Common Pooling Kernel Mistakes** · 143 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/pooling-anti-patterns.md`
- **Precision Patterns for Pooling Operators** · 169 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/precision-patterns.md`
- **Depth-Only Fast Path 的 AscendC 落地（Direct TBuf + DataCopy）** · 113 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/reduce-d-fastpath-implementation.md`
- **TileLang → AscendC Translation Rules for Pooling** · 85 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/tilelang-translation.md`
- **UB Buffer Management: TQue vs Direct TBuf** · 108 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/references/pooling-patterns/references/ub-management.md`
- **----------------------------------------------------------------------------------------------------------** · 358 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/scripts/build_ascendc.py`
- **----------------------------------------------------------------------------------------------------------** · 138 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/scripts/evaluate_ascendc.sh`
- **----------------------------------------------------------------------------------------------------------** · 1028 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/scripts/validate_ascendc_impl.py`
- **----------------------------------------------------------------------------------------------------------** · 1510 行 · community
  `cannbot-skills/community/ops/tilelang2ascend-translator/scripts/verification_ascendc.py`
- **TileLang Ascend API 最佳实践** · 92 行 · upstream
  `cannbot-skills/ops/tilelang-api-best-practices/Ascend910/instructions.md`
- **计算原语：GEMM、归约与 Tile 扩展操作** · 707 行 · upstream
  `cannbot-skills/ops/tilelang-api-best-practices/Ascend910/references/api-compute.md`
- **Kernel 定义、内存分配与数据搬运** · 496 行 · upstream
  `cannbot-skills/ops/tilelang-api-best-practices/Ascend910/references/api-kernel-memory.md`
- **调度、同步与调试** · 263 行 · upstream
  `cannbot-skills/ops/tilelang-api-best-practices/Ascend910/references/api-schedule-sync.md`
- **TileLang-Ascend 环境检查** · 633 行 · upstream
  `cannbot-skills/ops/tilelang-env-check/Ascend910/instructions.md`
- **-----------------------------------------------------------------------------------------------------------** · 289 行 · upstream
  `cannbot-skills/ops/tilelang-env-check/Ascend910/scripts/check_env.sh`
- **-*- coding: utf-8 -*-** · 65 行 · upstream
  `cannbot-skills/ops/tilelang-env-check/Ascend910/scripts/quick_verify.py`
- **完成报告输出模板** · 31 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/examples/completion-report-template.md`
- **{算子名称} 算子设计文档** · 497 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/examples/design-template.md`
- **TileLang-Ascend 算子设计文档生成** · 222 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/instructions.md`
- **Block Level Design** · 310 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/BlockLevelDesign.md`
- **技术约束清单（必须遵守）** · 262 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/ascend-constraints.md`
- **Attention Pattern Index** · 52 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/AttentionPatternIndex.md`
- **Attention Head Sharing Pattern** · 145 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/head_sharing_pattern.md`
- **Attention Mask Causal Pattern** · 150 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/mask_causal_pattern.md`
- **Attention MLA Pattern** · 167 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/mla_pattern.md`
- **Attention Page Attention Pattern** · 133 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/page_attention_pattern.md`
- **Attention Sink Pattern** · 169 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/sink_pattern.md`
- **Attention TND Pattern** · 125 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/tnd_pattern.md`
- **Attention Topk KV Sparse Pattern** · 119 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/attention-patterns/references/topk_kv_sparse_pattern.md`
- **算子特征分析决策树（Ascend 版）** · 118 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/decision-tree.md`
- **信息收集与信息源优先级** · 61 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/info-sources.md`
- **Adaptive Pooling Lessons: adaptive_avg_pool3d 生成经验** · 91 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/pooling/adaptive-avg-pool3d-lessons.md`
- **Backward Patterns: Pooling 反向梯度算子设计模式（语义与策略）** · 165 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/pooling/backward-patterns.md`
- **Layout Strategy: NDHWC for Row-Granularity Pooling** · 63 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/pooling/layout-strategy.md`
- **Reduce-D Fast Path: Depth-Only Pooling Optimization** · 51 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/pooling/reduce-d-fastpath.md`
- **Row-Granularity Tile Design for Pooling** · 74 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/pooling/row-granularity.md`
- **质量自检清单** · 50 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend910/references/quality-checklist.md`
- **openai** · 3 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend950/agents/openai.yaml`
- **<算子名称>：Ascend950 TileLang 算子设计** · 143 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend950/assets/design-template.md`
- **TileLang Ascend950 算子方案设计** · 260 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend950/instructions.md`
- **Ascend950 后端技术参考** · 97 行 · upstream
  `cannbot-skills/ops/tilelang-op-design/Ascend950/references/Ascend950-backend.md`
- **代码骨架模板** · 191 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/examples/code-skeleton.md`
- **TileLang-Ascend 算子代码生成** · 231 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/instructions.md`
- **技术约束清单（生成代码时必须遵守）** · 62 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/ascend-constraints.md`
- **Checklist** · 81 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/checklist.md`
- **关键编码规范** · 394 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/coding-conventions.md`
- **GEMM 与 CV 融合编码规范** · 43 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/gemm-cv-fusion.md`
- **算子疑难解答** · 393 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/troubleshooting.md`
- **V 核并行化编码规范** · 76 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend910/references/vector-parallelism.md`
- **openai** · 3 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend950/agents/openai.yaml`
- **TileLang Ascend950 算子生成** · 89 行 · upstream
  `cannbot-skills/ops/tilelang-op-develop/Ascend950/instructions.md`
- **TileLang 算子上板性能采集与调优** · 380 行 · upstream
  `cannbot-skills/ops/tilelang-op-profiling/Ascend950/instructions.md`
- **TileLang msprof op CSV 字段完整参考** · 311 行 · upstream
  `cannbot-skills/ops/tilelang-op-profiling/Ascend950/references/csv_fields_reference.md`
- **多核链路计时** · 13 行 · upstream
  `cannbot-skills/ops/tilelang-op-profiling/Ascend950/references/kernel-chain-measurement.md`
- **TileLang 瓶颈优化速查表** · 255 行 · upstream
  `cannbot-skills/ops/tilelang-op-profiling/Ascend950/references/optimization_quickref.md`
- **----------------------------------------------------------------------------------------------------------** · 664 行 · upstream
  `cannbot-skills/ops/tilelang-op-profiling/Ascend950/scripts/perf_summary.py`
- **TileLang-Ascend 算子测试设计** · 787 行 · upstream
  `cannbot-skills/ops/tilelang-op-test-design/Ascend910/instructions.md`
- **TileLang-Ascend 测试覆盖矩阵（强制契约）** · 179 行 · upstream
  `cannbot-skills/ops/tilelang-op-test-design/Ascend910/references/coverage-matrix.md`
- **算子类别划分依据详解** · 471 行 · upstream
  `cannbot-skills/ops/tilelang-op-test-design/Ascend910/references/operator-category.md`
- **TileLang-Ascend 精度标准体系（混合容差）** · 169 行 · upstream
  `cannbot-skills/ops/tilelang-op-test-design/Ascend910/references/precision-standard.md`

下一页：page-003.md
