# triton: 1–70 / 465

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **API 参考** · 38 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/index.md`
- **NPU 硬件架构（IR 视角）** · 6 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/index.md`
- **编译流水线** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/index.md`
- **HACC 方言** · 10 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hacc/index.md`
- **HFusion 方言** · 13 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hfusion/index.md`
- **HIVM 方言** · 72 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hivm/index.md`
- **IR 分析与 AscendNPU-IR** · 24 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/index.md`
- **内存管理** · 10 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/memory_management/index.md`
- **其他方言** · 9 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/other_dialects/index.md`
- **速查参考** · 6 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/reference/index.md`
- **Triton 方言** · 10 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/triton_dialects/index.md`
- **代码示例** · 20 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/examples/index.md`
- **设计期约束（stage: design）** · 59 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/constraints/index.md`
- **Triton 指南** · 11 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/index.md`
- **IR 工作流** · 7 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/ir/index.md`
- **GPU→NPU 迁移对照** · 7 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/migration/index.md`
- **算子专项设计** · 19 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_design/index.md`
- **算子实现手册（playbook）** · 142 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/operator_playbooks/index.md`
- **triton-ascend** · 19 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/triton_ascend/index.md`
- **工作流** · 31 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/guides/workflows/index.md`
- **Triton 算子知识** · 13 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/index.md`
- **优化经验** · 61 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/experiences/index.md`
- **Triton 优化方法** · 7 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/index.md`
- **优化期改写（stage: optimize）** · 95 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/operator_optimization/index.md`
- **优化技术** · 52 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/optimizations/techniques/index.md`
- **Triton Runbook** · 5 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/index.md`
- **实测性能案例（perf_card）** · 17 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/runbooks/performance/index.md`
- **Latency Optimizer Skill** · 157 行 · upstream
  `cannbot-skills/ops/triton-latency-optimizer/SKILL.md`
- **Triton Ascend 代码生成 Skill** · 346 行 · upstream
  `cannbot-skills/ops/triton-op-coding/SKILL.md`
- **Triton Ascend 算法草图设计 Skill** · 206 行 · upstream
  `cannbot-skills/ops/triton-op-designer/SKILL.md`
- **Kernel Verifier Skill** · 604 行 · upstream
  `cannbot-skills/ops/triton-op-verifier/SKILL.md`
- **Precision Debugger Skill** · 302 行 · upstream
  `cannbot-skills/ops/triton-precision-debug/SKILL.md`
- **Triton Simulator-Driven Optimizer** · 117 行 · upstream
  `cannbot-skills/ops/triton-simulator-optimizer/SKILL.md`
- **算子任务提取 Skill** · 257 行 · upstream
  `cannbot-skills/ops/triton-task-extractor/SKILL.md`
- **API 支持矩阵（API Support Matrix）** · 199 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/api_support_matrix.md`
- **parallel / compile_hint / multibuffer 辅助操作** · 170 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_aux_ops.md`
- **Buffer 编程模型（bl.alloc / to_buffer / to_tensor / subview）** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_buffer_model.md`
- **自定义算子注册与调用（register_custom_op / custom）** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_custom_op.md`
- **fixpipe 操作（L0C → UB 搬运与融合后处理）** · 164 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_fixpipe.md`
- **libdevice 数学函数库（HMF 硬件加速）** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_libdevice.md`
- **内存操作扩展（index_put / gather_out_to_ub / scatter_ub_to_out / index_select_simd）** · 168 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_mem_ops.md`
- **Ascend 扩展 API 总览与导入方式** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_overview.md`
- **PIPE / CORE / MODE / IteratorType 枚举** · 128 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_pipe_core.md`
- **sub_vec_id / sub_vec_num 子向量核操作** · 141 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_sub_vec_ops.md`
- **sync_block_set / wait / all 同步操作** · 166 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_sync.md`
- **向量操作扩展（insert_slice / extract_slice / get_element / sort / flip / cast）** · 146 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ascend_ext_vec_ops.md`
- **数据类型支持矩阵** · 169 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/data_type_matrix.md`
- **数据布局详解** · 102 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/data_layout.md`
- **内存层次详解** · 125 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/memory_hierarchy.md`
- **NPU 硬件架构总览** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/npu_hardware_overview.md`
- **Pipeline 执行模型** · 109 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/architecture/pipeline_execution_model.md`
- **Ascend 特有 Pass** · 121 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/ascend_passes.md`
- **前端 → HFusion 转换** · 88 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/frontend_to_hfusion.md`
- **HFusion → HIVM 转换** · 91 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hfusion_to_hivm.md`
- **HFusion 变换 Pass** · 114 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hfusion_transforms.md`
- **HIVM → 后端转换** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hivm_to_backend.md`
- **HIVM 变换 Pass** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/hivm_transforms.md`
- **Linalg IR → 设备二进制** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/linalg_to_binary.md`
- **编译流水线总览** · 83 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/overview.md`
- **Pass 依赖关系与执行顺序** · 151 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/pass_dependencies.md`
- **Pipeline 选项参考** · 159 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/pipeline_options.md`
- **Triton-Ascend 编译流程主线** · 113 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_compile_flow.md`
- **Triton-Ascend 编译选项与环境变量** · 133 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_compile_options.md`
- **Triton IR 编译路径** · 104 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_ir_compilation.md`
- **Triton-Ascend 编译流水线阶段划分** · 140 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_pipeline_overview.md`
- **TritonToLinalg 转换** · 108 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/triton_to_linalg.md`
- **AST → TTIR 生成** · 138 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/ttir_generation.md`
- **TTIR 优化 Pass** · 97 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/compilation_pipeline/ttir_optimization.md`
- **HACC 设备规格** · 87 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hacc/device_specification.md`
- **HACC 函数管理** · 98 行 · upstream
  `cannbot-knowledge/knowledge/ops/triton/apis/ir/hacc/function_management.md`

下一页：page-002.md
