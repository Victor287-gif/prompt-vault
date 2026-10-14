# other-operator-knowledge: 2031–2100 / 2176

文档ID可直接传给 read_knowledge；通过 GitHub 读取时，在包根路径后加 third_party/ 和文档ID。

- **MC2 通信路径决策（决策栈 L1 层）** · 51 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/requirement-analysis/comm-path-decision.md`
- **需求拷问协议（Grill Protocol）** · 202 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/requirement-analysis/grill-protocol.md`
- **MC2 需求文档自检清单** · 57 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/requirement-analysis/quality-checklist.md`
- **{算子名称} 需求文档** · 208 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/references/requirement-analysis/template.md`
- **----------------------------------------------------------------------------------------------------------** · 96 行 · upstream
  `cannbot-skills/ops/ascendc-mc2-best-practice/scripts/fetch_apace.sh`
- **MC2单算子性能抖动分析skill** · 117 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/comm-compute/MC2_JitterDiagnose.md`
- **Bound 判定方法** · 92 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/comm-compute/bound_diagnosis.md`
- **膨胀分析** · 60 行 · upstream
  `cannbot-skills/ops/ascendc-perf-optimize/references/comm-compute/expansion_analysis.md`
- **Vector 计算效率优化设计** · 203 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/elementwise/vector_efficiency_design.md`
- **SIMT 算子性能优化指南** · 110 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/simt/optimization-guide.md`
- **选择/基数排序类算子性能优化实践** · 354 行 · upstream
  `cannbot-skills/ops/ascendc-performance-best-practices/references/sort/radix_sort.md`
- **计算指令映射** · 34 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/compute-operators.md`
- **数据搬入搬出方案** · 110 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/data-transfer.md`
- **索引计算 API** · 83 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/index-calculation.md`
- **核内多线程共享内存** · 70 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/intra-core-shared.md`
- **内置宏参考** · 65 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/macro-api.md`
- **SIMT C API 总览** · 128 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/overview.md`
- **细粒度随机访存：写侧（散写）模式** · 43 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/random-access-patterns.md`
- **TensorList 动态输入处理** · 92 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/tensorlist-handling.md`
- **线程排布统一编程模式** · 122 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/thread-stride-pattern.md`
- **VF_CALL 启动模式** · 66 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/vf-call.md`
- **VF 函数声明模式** · 47 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/vf-declaration.md`
- **编译与运行** · 175 行 · upstream
  `cannbot-skills/ops/ascendc-simt-best-practices/references/编译与运行.md`
- **环境依赖安装指南** · 77 行 · upstream
  `cannbot-skills/ops/cann-env-setup/references/env_dependence.md`
- **组件错误码源码索引** · 47 行 · upstream
  `cannbot-skills/ops/cann-error-code-standard/references/component-error-code-source-index.md`
- **CANN算子错误码开发使用规范** · 98 行 · upstream
  `cannbot-skills/ops/cann-error-code-standard/references/development-standard.md`
- **CANN官方文档错误码清单** · 97 行 · upstream
  `cannbot-skills/ops/cann-error-code-standard/references/documented-error-code-catalog.md`
- **documented-error-codes** · 4026 行 · upstream
  `cannbot-skills/ops/cann-error-code-standard/references/documented-error-codes.json`
- **错误码来源与维护规则** · 79 行 · upstream
  `cannbot-skills/ops/cann-error-code-standard/references/source-and-maintenance.md`
- **整体实现：统一计算入口与必备导出入口** · 226 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/overall-implementation.md`
- **aclnn 适配：先对齐参数语义，再接真实接口** · 34 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/aclnn-adaptation.md`
- **AICPU 负载均衡 Metadata：把不均匀任务变成可消费的调度记录** · 116 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/aicpu-load-balance-metadata.md`
- **编译与启动：复用程序，绑定本次数据** · 144 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/compilation-launch.md`
- **Host 约束校验：先检查，再使用** · 34 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/host-validation.md`
- **布局与尾块：逻辑范围和物理地址分别计算** · 53 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/layout-tail.md`
- **Torch 注册：复用公开入口的薄适配** · 66 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/torch-registration.md`
- **VF：先写清寄存器内的计算和有效 lane** · 144 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/vf.md`
- **Workspace：容量按物理布局算，复用按生命周期算** · 67 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/patterns/workspace.md`
- **生产代码：清楚的结构与可检查的执行边界** · 140 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-best-practice/references/production-code.md`
- **CANNbot DSL 代码检视方法论** · 106 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/core/methodology.md`
- **生产执行约束** · 36 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/references/execution-rules.md`
- **可读性规则（DSL-R01 ~ R10）** · 280 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/references/readability-rules.md`
- **检视子 Agent prompt 模板（阶段1 派发时填入 {} 占位）** · 59 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/steps/clause-review.md`
- **检视报告格式** · 63 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/steps/report-write.md`
- **文件检视场景（轻量流程）** · 49 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/workflows/file-review.md`
- **PR 检视场景（轻量流程）** · 58 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-code-reviewer/workflows/pr-review.md`
- **检查定义与故障处理** · 96 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/references/checks-and-troubleshooting.md`
- **env_selected_soc.yaml 结构** · 43 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/references/env-selected-soc-schema.md`
- **environment.yaml 结构** · 112 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/references/environment-schema.md`
- **----------------------------------------------------------------------------------------------------------** · 34 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/scripts/check_env.sh`
- **-*- coding: UTF-8 -*-** · 946 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/scripts/env_check.py`
- **-*- coding: UTF-8 -*-** · 99 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/scripts/probe_compile_stage.py`
- **-*- coding: UTF-8 -*-** · 33 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-env-check/scripts/probe_kernel.py`
- **阶段成本与瓶颈分析** · 77 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-op-design/references/stage-cost-model.md`
- **-*- coding: UTF-8 -*-** · 211 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/assets/case_contract.py`
- **-*- coding: UTF-8 -*-** · 600 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/assets/test.py`
- **testcase.csv 用例契约** · 29 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/references/cases.md`
- **TDD 输入、触发条件、设计证据与调用生命周期** · 69 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/references/evidence.md`
- **脚本生成与执行验收** · 25 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/references/execution.md`
- **直接编写最终 Golden** · 14 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/references/golden.md`
- **-*- coding: UTF-8 -*-** · 71 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/scripts/casebook.py`
- **-*- coding: UTF-8 -*-** · 95 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/scripts/generator_support.py`
- **-*- coding: UTF-8 -*-** · 395 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/scripts/prepare_golden.py`
- **-*- coding: UTF-8 -*-** · 144 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-test-generator/scripts/recipe_validation.py`
- **可实现性与边界检查** · 46 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-unit-planner/references/decomposition-checks.md`
- **Unit YAML 格式** · 85 行 · upstream
  `cannbot-skills/ops/cannbot-dsl-unit-planner/references/unit-format.md`
- **既有算子错误码重构路由** · 132 行 · upstream
  `cannbot-skills/ops/ops-error-code-remediation/references/remediation-routing.md`
- **算子错误码返回状态报告规范** · 67 行 · upstream
  `cannbot-skills/ops/ops-error-code-remediation/references/status-report-schema.md`
- **E*9***触发入口覆盖契约** · 55 行 · upstream
  `cannbot-skills/ops/ops-error-code-remediation/references/trigger-coverage.md`
- **整改验证清单** · 61 行 · upstream
  `cannbot-skills/ops/ops-error-code-remediation/references/validation.md`

下一页：page-031.md
