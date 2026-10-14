# engineering: 421–490 / 823

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **参考工程（all_to_all_matmul/）改造食谱** · 342 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md`
- **通信层（SHMEM / UDMA）参考** · 280 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md`
- **计算层（Blaze Matmul）参考** · 259 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md`
- **MC2 通算融合架构总览** · 199 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md`
- **blaze-shmem 路线代码审查验收条件** · 29 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/review-checklist.md`
- **CANNBot Step 1→7 在 MC2 场景下的工作流映射** · 215 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md`
- **预定义变量** · 319 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/foundations/hccl-matmul/all_gather_matmul/build.sh`
- **Scalar 高性能编码原则** · 476 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/scalar/coding_principles.md`
- **算术与归约** · 228 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/arithmetic_and_reduce.md`
- **MemBase 与 RegBase 计算 API 对照** · 107 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/compute_api_membase_vs_regbase.md`
- **RegBase DataCopy 分层** · 139 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/datacopy_best_practices.md`
- **Pipeline 与 Buffer** · 105 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/pipeline_and_buffer.md`
- **Ascend C RegBase API 参考** · 303 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/regbase_api_reference.md`
- **RegBase API 同步** · 173 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/regbase_api_sync.md`
- **RegBase API 白名单** · 142 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/api/regbase_api_whitelist.md`
- **RegBase 构建笔记** · 49 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/dev-experience/regbase_build_notes.md`
- **RegBase 性能实践** · 52 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/dev-experience/regbase_performance_practices.md`
- **RegBase 编程笔记** · 120 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/dev-experience/regbase_programming_notes.md`
- **Tiling 审查笔记** · 86 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/dev-experience/tiling_review_notes.md`
- **DAV_3510 成功/失败案例对照** · 49 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/dev-experience/working_vs_failed_dav_3510_cases.md`
- **范式开发指导** · 190 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/README.md`
- **adam_apply_one_assign_def** · 102 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_host/adam_apply_one_assign_def.cpp`
- **adam_apply_one_assign_tiling_arch35** · 597 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp`
- **adam_apply_one_assign_tiling_arch35** · 82 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.h`
- **adam_apply_one_assign** · 44 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_kernel/adam_apply_one_assign.cpp`
- **adam_apply_one_assign_kernel** · 361 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`
- **adam_apply_one_assign_struct** · 27 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_struct.h`
- **adam_apply_one_assign_tiling_struct** · 53 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h`
- **AdamApplyOneAssign 接口分析** · 143 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-design/adam_apply_one_assign_analysis.md`
- **AdamApplyOneAssign 详细设计** · 2865 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/examples-adam-design/adam_apply_one_assign_design.md`
- **-----------------------------------------------------------------------------------------------------------** · 328 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/CMakeLists.txt`
- **xdivy_custom_package** · 163 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/README.md`
- **xdivy_graph_infer** · 38 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_graph/xdivy_graph_infer.cpp`
- **xdivy_proto** · 42 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_graph/xdivy_proto.h`
- **xdivy_tiling_arch35** · 601 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_host/arch35/xdivy_tiling_arch35.cpp`
- **xdivy_tiling_arch35** · 89 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_host/arch35/xdivy_tiling_arch35.h`
- **xdivy_def** · 60 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_host/xdivy_def.cpp`
- **xdivy_infershape** · 58 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_host/xdivy_infershape.cpp`
- **xdivy_kernel** · 331 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_kernel/arch35/xdivy_kernel.h`
- **xdivy_struct** · 28 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_kernel/arch35/xdivy_struct.h`
- **xdivy_tiling_struct** · 57 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_kernel/arch35/xdivy_tiling_struct.h`
- **xdivy** · 42 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/example/xdivy_custom_package/op_kernel/xdivy.cpp`
- **Broadcast 类算子 DESIGN.md 生成规范** · 324 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/patterns.md`
- **broadcast 范式 Kernel 入口** · 95 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-kernel-entrance.md`
- **broadcast 模板总览** · 252 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-overview.md`
- **broadcast 范式 standard 计算流伪码** · 214 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-compute-flow.md`
- **broadcast 范式 standard DAG 分析和 Buffer 划分** · 285 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-dag-buffers.md`
- **broadcast 范式 Kernel 辅助函数** · 209 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-kernel-helpers.md`
- **broadcast 范式 standard 模板** · 611 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-kernel-template.md`
- **broadcast 范式 standard Tiling 预处理** · 545 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-tiling-preprocess.md`
- **broadcast 范式 standard Tiling 切分策略** · 471 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/broadcast/references/broadcast-template-standard/broadcast-standard-tiling.md`
- **Broadcast 规则** · 126 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/broadcast-rules.md`
- **Cast 规则** · 414 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/cast-rules.md`
- **DataCopyPad 规则** · 185 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/datacopypad-rules.md`
- **NDDMA 规则** · 255 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/nddma-rules.md`
- **同步与数据一致性** · 314 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/sync-and-consistency.md`
- **值依赖分析** · 40 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/value-depend-process.md`
- **VF 编程规则** · 150 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/common/vf-programming-rules.md`
- **DynamicShape 值依赖机制原理** · 382 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/dynamic_shape/knowledge.md`
- **DynamicShape 类算子场景路由** · 260 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/dynamic_shape/patterns.md`
- **EleWise 类算子场景路由** · 46 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/elewise/patterns.md`
- **EleWise - Tiling 详细计算** · 215 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/elewise/tiling.md`
- **example-routes.yaml — 范式 → 参考实现路由表** · 17 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/example-routes.yaml`
- **Ascend C 算子 Tiling 通用设计方法论** · 94 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/general-methodology.md`
- **MaskPredicate 门禁校验与场景路由** · 99 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/mask_predicate/patterns.md`
- **数值稳定性核心知识** · 233 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/numerical_stable/knowledge.md`
- **数值稳定性实现模式** · 390 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/numerical_stable/patterns.md`
- **-----------------------------------------------------------------------------------------------------------** · 289 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/reduction/example/euclidean_norm_package/CMakeLists.txt`
- **euclidean_norm_package** · 148 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/reduction/example/euclidean_norm_package/README.md`
- **euclidean_norm_graph_infer** · 39 行 · upstream
  `cannbot-skills/ops/ascendc-regbase-best-practice/references/paradigms/reduction/example/euclidean_norm_package/op_graph/euclidean_norm_graph_infer.cpp`

下一页：page-008.md
