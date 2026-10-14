---
type: Guide
title: Counter Mask 的作用域与尾部计数
description: 保留总元素模式、手动 mask 重载、模式恢复与混合 dtype 边界。
tags:
- task.implementation
- topic.implementation.masking
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P67_vector_counter_mode.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P67_vector_counter_mode.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P67.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_count.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/mask_operations/set_mask_norm.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/guides/operator_practice/simd_operator_optimization/vector_compute/vector_counter_mode.md
  role: constraint
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# Counter Mask 的作用域与尾部计数

Normal模式用repeat次数和每repeat的mask描述有效计算，Counter模式直接描述总元素数，由支持该模式的Vector重载推断迭代和尾部。来源希望省去主尾块的Scalar分支；对于half的256字节repeat，15000元素为117个满repeat加24个尾元素，而float满repeat为64元素，不能混用宽度。

显式SetMaskCount和SetVectorMask须与isSetMask=false等实际允许外部mask控制的重载配合；若API默认自行设置mask，外部写入可能被覆盖。Counter的计数、mask参数形状、repeat占位值及stride依所用API规定，不能把所有调用的repeat统一改成1，或假设所有接口均支持2^32-1元素。Counter不会扩充UB容量，也不自动把超大输入分配到多个tile。

模式选择和mask值是两份状态。ResetMask恢复默认mask值不等于SetMaskNorm恢复Normal模式；手动Counter区域结束后，按后续代码合同恢复模式和值。可以在有明确状态约定的helper间使用，不要求物理上必须同一函数，但异常分支、提前返回、其他helper修改与后续普通Vector均需覆盖。每条操作使用的有效count必须明确，不能默认前一条调用后的状态仍适合当前dtype/长度。

来源P67把一组归约API统一列为不支持Counter，P68又在Counter中调用WholeReduceSum；这些相互冲突的列表不能充当版本支持矩阵，应逐个重载核对。Cast链允许有显式转换的混合类型，但count语义、源目的容量及精度要相应处理。

Normal尾mask也有边界：`(1<<tail)-1`可能发生整型宽度或移位溢出，half尾部超过64位时不能只生成一个64位bitmask。连续mask和逐bitmask不能混为一谈。静态长度可能已被编译器折叠，小任务模式切换成本也可能抵消节省；来源约20条Scalar及0–10%性能变化未在本批复现。验证0长度、整repeat、跨64/128边界、多helper混用和Counter后首个Normal计算。
