# pypto: 71–126 / 126

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **MACHINE 组件经验** · 346 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_classified/machine.md`
- **MATMUL 组件经验** · 105 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_classified/matmul.md`
- **PASS 组件经验** · 411 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_classified/pass.md`
- **VECTOR 组件经验** · 97 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_classified/vector.md`
- **视图类OP 经验** · 91 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_classified/view_op.md`
- **经验库预检 — 流程指南** · 155 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/experience_preflight.md`
- **固定规则 F1-F8** · 54 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/fixed_rules.md`
- **PyPTO DFX 问题查找表** · 208 行 · upstream
  `cannbot-skills/ops/pypto-op-knowledge/references/problem-lookup.md`
- **coding: utf-8** · 632 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py`
- **PyPTO 算子性能分析报告模板** · 235 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/templates/performance_report_template.md`
- **PyPTO 算子性能优化点索引库** · 557 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/shared/optimization_catalog.md`
- **典型调优案例库** · 11 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/cases/README.md`
- **案例：尾轴 Broadcast 合轴优化（combine_axis）** · 74 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/cases/combine-axis-broadcast.md`
- **案例：多 Matmul 独立 TileShape 优化** · 77 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/cases/per-matmul-tile-shapes.md`
- **案例：Decode Attention Vector 合轴 + 合图优化** · 79 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/cases/vector-axis-merge-softmax.md`
- **高级用法：A/B 矩阵独立设置 K 轴切分** · 144 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/references/basic-block-optimization.md`
- **✅ 正确：reshape 挪到算子入口，inplace=True** · 106 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/references/reshape-global-optimization.md`
- **典型调优案例库** · 9 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-incore/cases/README.md`
- **案例：权重矩阵批量 NONE_CACHEABLE** · 113 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-incore/cases/weight-none-l2-cacheable.md`
- **阶段交接摘要（PHASE_SUMMARY_F / PHASE_SUMMARY_S / PHASE_SUMMARY_I）** · 218 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-orchestrator/references/phase-handoff.md`
- **原来：N=4096, nL0=256, nL1=256 → 4096/256=16 个 N 轴任务，但 M=1 → 总共 16 个任务** · 159 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/core-usage-load-balancing.md`
- **faq** · 25 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/faq.md`
- **leafHash → 前端代码映射方法** · 183 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/leafhash-to-code-mapping.md`
- **matmul-l2-layout** · 53 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/matmul-l2-layout.md`
- **Step 0: 核使用率分析（⛔ 强制前置，详见第 3 节）** · 448 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/merge-optimization.md`
- **上下游 TileShape 对齐时，可以合并在一个子图** · 125 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/references/tileshape-deep-tuning.md`
- **coding: utf-8** · 489 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/analyze_aiv_dep_chains.py`
- **coding: utf-8** · 200 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/analyze_core_usage.py`
- **coding: utf-8** · 285 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/analyze_swimlane.py`
- **coding: utf-8** · 256 行 · upstream
  `cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/scripts/leafhash_to_code.py`
- **Layout / structure requirements** · 21 行 · upstream
  `cannbot-skills/ops/pypto-op-review/references/CI.md`
- **-*- coding: utf-8 -*-** · 116 行 · upstream
  `cannbot-skills/ops/pypto-op-review/scripts/extract_pypto_calls.py`
- **Intermediate-value snapshot automation** · 253 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/references/intermediate-snapshot-automation.md`
- **Copyright (c) 2026 Huawei Technologies Co., Ltd.** · 58 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/references/snapshot_manifest.example.yaml`
- **-*- coding: utf-8 -*-** · 162 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/detailed_tensor_compare.py`
- **-*- coding: utf-8 -*-** · 158 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/gen_cleanup.py`
- **-*- coding: utf-8 -*-** · 170 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/gen_module_interfaces.py`
- **-*- coding: utf-8 -*-** · 301 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/gen_module_test.py`
- **coding: utf-8** · 452 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/snapshot_bisect.py`
- **coding: utf-8** · 473 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/snapshot_generator.py`
- **coding: utf-8** · 199 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/snapshot_manifest_schema.py`
- **-*- coding: utf-8 -*-** · 272 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/scripts/validate_yaml.py`
- **-*- coding: utf-8 -*-** · 238 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/templates/adversarial_runner_template.py`
- **-*- coding: utf-8 -*-** · 189 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/templates/cancellation_stress_template.py`
- **test_cases_template** · 46 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/templates/test_cases_template.json`
- **-*- coding: utf-8 -*-** · 95 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/templates/test_inputs_template.py`
- **-*- coding: utf-8 -*-** · 182 行 · upstream
  `cannbot-skills/ops/pypto-op-verify/templates/test_template.py`
- **Agent Team — Dispatch Contract** · 219 行 · upstream
  `cannbot-skills/ops/pypto-orchestration-manual/references/agents.md`
- **Behavioral Principles** · 71 行 · upstream
  `cannbot-skills/ops/pypto-orchestration-manual/references/principles.md`
- **PyPTO kernel — mandatory rules** · 122 行 · upstream
  `cannbot-skills/ops/pypto-orchestration-manual/references/rules.md`
- **PIPE_ALL 同步问题定位流程** · 203 行 · upstream
  `cannbot-skills/ops/pypto-precision-compare/precision-pass/references/pipe_all.md`
- **coding: utf-8** · 565 行 · upstream
  `cannbot-skills/ops/pypto-precision-compare/precision-pass/scripts/binary_pipeall_sync.py`
- **-*- coding: utf-8 -*-** · 259 行 · upstream
  `cannbot-skills/ops/pypto-precision-compare/precision-pass/scripts/common.py`
- **-*- coding: utf-8 -*-** · 221 行 · upstream
  `cannbot-skills/ops/pypto-precision-compare/precision-pass/scripts/locate_source_line.py`
- **coding: utf-8** · 382 行 · upstream
  `cannbot-skills/ops/pypto-precision-compare/precision-verify/scripts/compare_accuracy.py`
- **coding: utf-8** · 178 行 · upstream
  `cannbot-skills/ops/pypto-precision-compare/precision-verify/scripts/plot_accuracy.py`
