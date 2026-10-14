# catlass: 1–70 / 79

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **catlass** · 21 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/index.md`
- **CATLASS Kernel Design** · 251 行 · upstream
  `cannbot-skills/ops/catlass-op-design/SKILL.md`
- **CATLASS Kernel Code Generation** · 159 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/SKILL.md`
- **Catlass Kernel Tuning** · 161 行 · upstream
  `cannbot-skills/ops/catlass-op-perf-tune/SKILL.md`
- **CATLASS FA 接口映射及参考配方冲突** · 40 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/attention_interface_mapping_and_conflicting_recipes.md`
- **CATLASS 编译架构与启动资源合同** · 35 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/compilation_architecture_and_launch_resources.md`
- **CATLASS 的 Tile、Block 与 Kernel 组件契约** · 48 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/component_contracts.md`
- **CATLASS 开发实例的装配与 Bias 所有权** · 41 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/development_assembly_and_bias_ownership.md`
- **CATLASS Epilogue 槽位与自定义 Tile 边界** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/epilogue_slot_topology_and_custom_tile_limits.md`
- **Grouped 融合事件 Credit 与排障证据** · 29 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/grouped_event_credits_and_diagnostic_limits.md`
- **Grouped Matmul 融合的分区与 Partial Max** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/grouped_fusion_partition_and_partial_max.md`
- **CATLASS 分组融合的跨块依赖与小 M 调度** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/grouped_fusions_cross_tile_dependencies_and_small_m.md`
- **CATLASS Host Device 与 Workspace 指针 ABI** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/host_device_workspace_and_pointer_abi.md`
- **独立 Golden 与整数 Dot 的精确性条件** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/independent_golden_and_integer_dot_exactness.md`
- **CATLASS 线性 Attention 的阶段合同与 Chunk 归属** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/linear_attention_stage_semantics_and_chunk_ownership.md`
- **CATLASS Matmul 模板索引与实例化合同** · 36 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/matmul_template_catalog_and_instantiation_contracts.md`
- **CATLASS 精度条件的并集与验证器漏判** · 37 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/precision_criteria_union_and_validator_failures.md`
- **CATLASS 参考版本与评测语义身份** · 30 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/reference_versions_and_evaluation_semantic_identity.md`
- **CATLASS 轮转 workspace 与 Callback 的收益条件** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/rotating_workspace_capacity_and_callback_evidence.md`
- **CATLASS Shape 合法性与布局实例化** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/shape_legality_and_layout_instantiation.md`
- **CATLASS tile 的 L1、L0 与 UB 分项预算** · 39 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/guides/catlass/tile_resource_budget.md`
- **catlass 坐标/形状等基础类型的头文件在库顶层（catlass/gemm_coord.hpp），不在 catlass/gemm/ 子目录下** · 32 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/compilation/catlass_vocabulary_headers_top_level.md`
- **catlass MIX kernel 确定性首发 507015 L0 冲突——先审 Resource 生命周期，再做几何二分** · 33 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/catlass_resource_lifetime_l0_conflict.md`
- **catlass Resource 内存模型与 event-ID 规则：一核一 Resource、GetBufferByByte 子分配、intra-core 同步用 AllocEventID** · 63 行 · upstream
  `cannbot-knowledge/knowledge/ops/ascendc/runbooks/runtime/catlass_resource_memory_model_event_ids.md`
- **Layer 0: 硬件与架构** · 110 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/architecture/00-hardware-arch.md`
- **Layer 1: Tile 原语层** · 54 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/architecture/01-tile-layer.md`
- **Layer 2: Block 组件层** · 182 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/architecture/02-block-layer.md`
- **Layer 3: Kernel 组装层** · 199 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/architecture/03-kernel-layer.md`
- **自定义 Tile Epilogue（设计阶段）** · 115 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/custom-epilogue.md`
- **Catlass 章节模板（嵌入 DESIGN.md 的 catlass 部分）** · 179 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/design-document.md`
- **Catlass 后处理（Epilogue）组件清单** · 54 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/epilogue-components.md`
- **Attention 子场景：GDN / KDA** · 46 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/attention/gdn-kda.md`
- **Kernel 路由：Linear Attention 类算子** · 220 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/attention/linear-attention.md`
- **Kernel 路由：Epilogue 后处理模式** · 37 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/epilogue-patterns.md`
- **FlashAttention 标杆接口参考规范** · 55 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/flash-attention-npu-reference.md`
- **Kernel 路由：FlashAttention 类算子** · 151 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/flash-attention.md`
- **flash-linear-attention-npu 参考规范** · 62 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/flash-linear-attention-npu-reference.md`
- **Kernel 路由：Grouped Matmul 类算子** · 220 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/grouped-matmul.md`
- **Kernel 路由：Linear Attention 类算子** · 11 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/linear-attention.md`
- **Kernel 路由：Matmul 类算子** · 179 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/matmul.md`
- **线性 Attention 参考资产地图** · 117 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/open-source-linear-attention-map.md`
- **Kernel 路由：Quant Matmul 类算子** · 40 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/kernels/quant-matmul.md`
- **Catlass 矩阵乘法模板清单** · 83 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/matmul-templates.md`
- **选到最优的 BlockMmad 与 BlockEpilogue（影响精度 + 性能）** · 124 行 · upstream
  `cannbot-skills/ops/catlass-op-design/references/mmad-epilogue-selection.md`
- **00: Kernel 组装全景** · 95 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/architecture/00-overview.md`
- **01: Kernel Assembly — using 链标准模式** · 142 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/architecture/01-kernel-assembly.md`
- **02: Device Calling** · 121 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/architecture/02-device-calling.md`
- **03: Compilation Requirements** · 85 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/architecture/03-compilation.md`
- **自定义 Tile Epilogue 实现骨架** · 84 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/custom-epilogue.md`
- **A2/A3 FlashAttention Stage 设计经验** · 116 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md`
- **A2/A3 线性 Attention Stage 设计经验** · 239 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/a2-a3-linear-attention-stage-design.md`
- **纯 Matmul — 完整代码骨架** · 76 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/basic-matmul.md`
- **Branch Instantiation — 多分支实例化** · 88 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/branch-instantiation.md`
- **FA 族手搓 kernel 坑清单（AIC/AIV 双 launch / 双子核 / online softmax / 多 batch）** · 69 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-handcraft-pitfalls.md`
- **FA kernel 复合特性叠加与避坑（sink / varlen / q8 / causal / paged / 测量）** · 169 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md`
- **FA kernel 手搓实现手册（从零写出 FAInferKernel 级别的 kernel）** · 401 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-kernel-handcraft.md`
- **FA kernel 算子配方库（FFA / s1s2 / S1-sink / kfc 直调）** · 661 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md`
- **MLA 手搓完整配方（kernel+tiling+host 端到端逐步骤，闭卷可生成）** · 511 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-mla-handcraft-recipe.md`
- **MLA Paged Attention — 实现契约** · 325 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-mla-paged.md`
- **FA-Paged 手搓完整配方（四文件端到端逐步骤，NO_MASK 全量验证 PASS）** · 696 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-paged-handcraft-recipe.md`
- **FA 族性能评测与 gate 判定方法论（计时口径陷阱 / 官方标杆 / 时长分布泛化套件）** · 104 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-perf-gate-evaluation.md`
- **FA-SageAttention 完整架构规格（Ascend950 / dav-3510）** · 261 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-sageattention-architecture.md`
- **精度/挂死取证方法论（已固化验证的完整流程）** · 124 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-sageattention-precision.md`
- **fa-sageattention（Ascend950/dav-3510 量化注意力）开发流程与逐步骤配方** · 108 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-sageattention-recipe.md`
- **FA-SageAttention 硬规则（不变量 + 运行纪律）** · 32 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-sageattention-rules.md`
- **Ascend950 (dav-3510) 实测平台坑全集（20 个，每个都有真实事故）** · 190 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-sageattention-troubleshooting.md`
- **fa-sink / 标准 FA（S1 模板）手搓完整配方（S1 单遍组织逐步骤）** · 307 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-sink-handcraft-recipe.md`
- **fa-varlen（TND 变长）手搓完整配方（s1s2 双层组织逐步骤）** · 172 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/fa-varlen-handcraft-recipe.md`
- **FFA / FIA（双 KV TensorList + sharedPrefix）算子配方** · 105 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/ffa-handcraft-recipe.md`
- **FlashAttention — BNSD QKV 类实现注意事项** · 267 行 · upstream
  `cannbot-skills/ops/catlass-op-develop/references/patterns/flash-attention.md`

下一页：page-002.md
