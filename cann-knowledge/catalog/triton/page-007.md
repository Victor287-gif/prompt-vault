# triton: 421–465 / 465

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **NPU 分块策略（Tiling）** · 139 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/tiling_strategy.md`
- **Where 条件优化（单位置 False 场景）** · 96 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/where_optimization.md`
- **2D Broadcast Division 优化案例** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/elemwise_broadcast_2d.md`
- **中等规模 Amax 归约优化** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_amax_medium.md`
- **极小规模 Amax 归约优化** · 47 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_amax_small.md`
- **极大规模 1D Amin 归约优化** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_amin_large.md`
- **大规模 2D Amin 归约优化** · 65 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_amin_medium.md`
- **中等规模 1D Amin 归约优化** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_amin_small.md`
- **大规模 Mean 归约优化（行二次切分）** · 69 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_mean_large.md`
- **中等规模 Mean 归约优化（reduce第一根轴）** · 74 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_mean_medium.md`
- **小规模 Prod 归约优化** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_prod_small.md`
- **Reduction + Elementwise 融合算子优化** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_sum_fused.md`
- **大规模 Sum 归约优化** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_sum_large.md`
- **Weighted SwiGLU Backward 融合算子优化** · 84 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/reduction_weighted_swiglu.md`
- **NPU 架构代际说明** · 312 行 · upstream
  `cannbot-skills/ops/npu-arch/references/npu-arch-guide-triton.md`
- **优化点索引（triton-latency-optimizer）** · 137 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/references/Index.md`
- **算子特定优化经验（本地保留区）** · 15 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/references/operators/README.md`
- **Grouped MatMul（MoE 专家分组矩阵乘）Tile 合并优化经验** · 271 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/references/operators/grouped-matmul-tile-merge.md`
- **LightningIndexer 优化经验（拆分路径 + profile-driven tile 放大 + program 开销削减）** · 195 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/references/operators/lightning-indexer.md`
- **Attention / MLA 类算子优化点库（Triton-Ascend）** · 87 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/references/operators/mla-paged-attention-optimization.md`
- **Attention / Sparse Flash Attention（SFA）类算子优化点库（Triton-Ascend）** · 137 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/references/operators/sparse-flash-attention-optimization.md`
- **----------------------------------------------------------------------------------------------------------** · 446 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/scripts/run_and_extract.sh`
- **Hint 模式：参数空间配置指南** · 115 行 · upstream
  `cannbot-skills/ops/triton-op-designer/references/hint-mode.md`
- **UnifiedSketch DSL 扩展规范** · 382 行 · upstream
  `cannbot-skills/ops/triton-op-designer/references/sketch-design-extended.md`
- **UnifiedSketch 设计规范** · 308 行 · upstream
  `cannbot-skills/ops/triton-op-designer/references/sketch-design.md`
- **----------------------------------------------------------------------------------------------------------** · 194 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/_baseline_integrity.py`
- **----------------------------------------------------------------------------------------------------------** · 266 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/_common_utils.py`
- **----------------------------------------------------------------------------------------------------------** · 38 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/_log_utils.py`
- **----------------------------------------------------------------------------------------------------------** · 1438 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/benchmark.py`
- **----------------------------------------------------------------------------------------------------------** · 177 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/freeze_baseline.py`
- **----------------------------------------------------------------------------------------------------------** · 406 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/hidden_state_probe.py`
- **----------------------------------------------------------------------------------------------------------** · 429 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/lock_npu_frequency.py`
- **----------------------------------------------------------------------------------------------------------** · 379 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/npu_preflight.py`
- **----------------------------------------------------------------------------------------------------------** · 2250 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/validate_triton_impl.py`
- **----------------------------------------------------------------------------------------------------------** · 1405 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/scripts/verify.py`
- **Attention 低精度逐位契约 — 参考实现算术路径复刻** · 211 行 · upstream
  `cannbot-skills/ops/triton-precision-debug/references/attention-lowprec-contract.md`
- **NPU 浮点除法 scale 校正（坐标/权重查找表法）** · 221 行 · upstream
  `cannbot-skills/ops/triton-precision-debug/references/divide-scale-calibration.md`
- **Triton-Ascend 精度对齐调试完整案例 —— DynamicQuant** · 299 行 · upstream
  `cannbot-skills/ops/triton-precision-debug/references/precision-alignment-guide.md`
- **瓶颈诊断（pipe 占比规则）** · 146 行 · upstream
  `cannbot-skills/ops/triton-simulator-optimizer/references/bottleneck-diagnosis.md`
- **msprof op simulator 采集与解析** · 237 行 · upstream
  `cannbot-skills/ops/triton-simulator-optimizer/references/msprof-simulator.md`
- **----------------------------------------------------------------------------------------------------------** · 83 行 · upstream
  `cannbot-skills/ops/triton-simulator-optimizer/scripts/enumerate_kernels.py`
- **----------------------------------------------------------------------------------------------------------** · 164 行 · upstream
  `cannbot-skills/ops/triton-simulator-optimizer/scripts/parse_prof.py`
- **算子任务格式规范** · 163 行 · upstream
  `cannbot-skills/ops/triton-task-extractor/references/task-format.md`
- **----------------------------------------------------------------------------------------------------------** · 37 行 · upstream
  `cannbot-skills/ops/triton-task-extractor/scripts/_log_utils.py`
- **----------------------------------------------------------------------------------------------------------** · 501 行 · upstream
  `cannbot-skills/ops/triton-task-extractor/scripts/validate_task.py`
