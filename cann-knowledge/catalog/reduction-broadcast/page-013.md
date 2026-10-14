# reduction-broadcast: 841–910 / 1263

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **fp16 1-ULP失配：torch多维softmax不升fp32** · 185 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_multidim_fp16_softmax_no_internal_upcast.md`
- **torch.unique ±0 符号不稳定致 bit-exact 失败** · 118 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/torch_unique_negzero_sign_normalize.md`
- **unroll s2 内层循环破坏 flash 在线 softmax 循环态** · 73 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/unroll_s2_flash_softmax.md`
- **static Broadcast<T,N,axis,false>(...,workspace) sharing a workspace with RA-form ReduceSum undersizes it on V351 — switch to dynamic GetBroadcastTilingInfo API** · 55 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/v351_broadcast_static_workspace_001.md`
- **bf16 输出 NaN 载荷 0x7FFF 需归一化为 0x7FC0** · 111 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/precision/vf_cast_bf16_nan_payload_0x7fff_vs_golden_7fc0_normalize.md`
- **507035：多核 reduce SyncAll 回退单核** · 171 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/507035_syncall_multicore_reduce.md`
- **合轴 fuse R 无尾 A → tail-R key=1（非 key=0）** · 154 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/reduce_fuse_pattern_is_tail_r_branch_rule.md`
- **ReduceSum tmp 与 src 须分区：同缓冲偏移切分** · 122 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/reducesum_sharedtmp_offset_partition.md`
- **Softmax 大 R 分核：Phase1+SyncAll+Phase2 合并** · 336 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/softmax_group_2d_split_phase1_syncall_phase2.md`
- **softmax regbase VF计算链参考结构** · 308 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/softmax_regbase_vf_compute_chain_reference.md`
- **大规模M/E用例507035未解负结果** · 82 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/softmax_topk_large_scale_507035_unresolved.md`
- **黑盒CSV转ST清单：shape capping保rank广播** · 363 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/blackbox_csv_shape_capping_rank_broadcast.md`
- **fp16/bf16 位流转 float 零依赖转换** · 190 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/cxx_fp16_bf16_bitstream_to_float_conversion.md`
- **torch ST非连续用例：先npu()再转置才真非连续** · 126 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/torch_st_noncontiguous_npu_then_transpose.md`
- **x86 host fp16：探测 _Float16，兜底位级转换** · 291 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/validations/x86_host_fp16_type_probe_portable_conversion.md`
- **HFusion 归约操作** · 76 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hfusion/reduction_ops.md`
- **HIVM 归约运算（hivm.hir.vreduce）** · 70 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/vector_reduction_ops.md`
- **归约操作 API** · 112 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/tl_reduction_ops.md`
- **融合 Softmax 示例（Fused Softmax on NPU）** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/fused_softmax.md`
- **LayerNorm 示例（前向融合 + 反向自旋锁分组归约）** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/layer_norm.md`
- **归约操作模式集（Reduction Patterns）** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/reduction_pattern.md`
- **RMSNorm 融合代码骨架** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/rmsnorm_fused.md`
- **跨轴 3D Broadcast 优化案例** · 79 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_elemwise_broadcast_3d.md`
- **大规模 Amax 归约优化（reduce轴映射多核）** · 68 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_reduction_amax_large.md`
- **Amin 归约原子操作优化案例** · 143 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/case_reduction_amin_atomic.md`
- **Reduce 算子优化** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/coding_reduce.md`
- **normalization 类算子设计约束** · 187 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/normalization_constraints.md`
- **Normalization 归一化算子** · 44 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/normalization.md`
- **归约算子** · 45 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/reduction.md`
- **Softmax / LogSoftmax 算子** · 43 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/softmax.md`
- **ConvTranspose2d 算子（2D 转置卷积） 实现手册** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/convolution__convtranspose2d.md`
- **AdaptiveInstanceNorm2DBackward 算子（norm-backward / backward） 实现手册** · 232 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/normalization__adaptiveinstancenorm2dbackward.md`
- **GroupNormSwish 算子（norm-fused / forward） 实现手册** · 463 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/normalization__groupnormswish.md`
- **AttentionSoftmaxWithSoftcappingAndDropout 算子（reduce） 实现手册** · 198 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__attentionsoftmaxwithsoftcappinganddropout.md`
- **MoeGatingTopKSoftmax 算子（sort-topk） 实现手册** · 232 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/transformer_inference__moegatingtopksoftmax.md`
- **归一化优化经验聚合** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/normalization_experiences.md`
- **归约优化经验聚合** · 34 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/reduction_experiences.md`
- **Softmax 优化经验聚合** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/softmax_experiences.md`
- **Histogram-like / Small-Output-Table Reduction 优化** · 145 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/histogram_like_table_reduction.md`
- **Ascend Matmul Transpose 优化** · 340 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/matmul_transpose.md`
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
- **Reduction Operations Optimization Insights** · 108 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-knowledge/references/a3/algorithm_insights/reduction_ops.md`
- **Broadcast 族策略选型指南** · 146 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/families/broadcast_guide.md`
- **Conversion / Transpose 族策略选型指南** · 173 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/families/conversion_guide.md`
- **Reduction / Normalization 族策略选型指南** · 139 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/families/reduction_guide.md`
- **Conversion 优化工作流（Transpose / Permute / Split / Concat）** · 41 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/scenario-routing/conversion.md`
- **Reduction 系优化工作流（含 Normalization / Softmax）** · 29 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/scenario-routing/reduction.md`
- **Broadcast - 动态 UB Broadcast（DAV_3510）** · 87 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/broadcast/dynamic-ub-broadcast.md`
- **Broadcast - NDDMA Broadcast 分支（DAV_3510）** · 199 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/broadcast/nddma-broadcast.md`
- **Broadcast - OneDim 分支** · 152 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/broadcast/onedim.md`
- **Broadcast 类算子场景路由** · 163 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/broadcast/patterns.md`
- **Broadcast - UB Broadcast 静态接口（DAV_2201）** · 331 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/broadcast/ub-broadcast.md`
- **Conversion 类算子模式** · 261 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/conversion/patterns.md`
- **二分累加（Dichotomy Addition / Half-Interval）** · 55 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/alg-dichotomy.md`
- **Group Reduce（跨核归约）** · 100 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/alg-group-reduce.md`
- **Welford Online 算法（在线单遍）** · 74 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/alg-welford.md`
- **Reduction 类算子核心算法路由** · 35 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/algorithms.md`
- **Reduce 类算子 - AR Col-Split 分支** · 149 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/ar-colsplit.md`
- **Reduce 类算子 - AR 全载分支** · 151 行 · community
  `cannbot-skills/community/ops/ascendc-evolution-strategies/references/tiling-design/reduction/ar-fullload.md`

下一页：page-014.md
