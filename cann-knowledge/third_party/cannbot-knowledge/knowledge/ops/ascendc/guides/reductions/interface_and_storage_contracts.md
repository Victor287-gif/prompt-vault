---
type: Guide
title: Reduce 的有效范围、存储布局与临时空间契约
description: 区分逐行 count 型归约与多维 Pattern 归约，明确有效 count、补齐行步长和不同重载的临时空间要求。
tags:
- task.design
- topic.compute.reduction
- topic.memory
- topic.alignment
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-reduce.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-reduce-pattern.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_max.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/reduction_operations/reduce_max_interface/reduce_max.md
  role: constraint
created_at: '2026-09-10T12:27:52Z'
updated_at: '2026-09-10T12:27:52Z'
platforms:
- '950'
- a2
- a3
---

# Reduce 的有效范围、存储布局与临时空间契约

## 先选择接口契约

count 型 Reduce 处理一段逻辑输入；Pattern 型 Reduce 按多维 shape 指定 R 轴和保留的 A 轴。AR 表示归约内轴，RA 表示归约外轴。选择取决于数据布局、归约轴和资源成本，不能仅凭“多行”就断定某种接口更快。

count 型接口的输入允许非整齐元素数，不等于地址无对齐要求。已有 ReduceMax 契约要求源地址和工作区起始地址满足 32 字节对齐，目的地址还受对应输出布局约束。

## 有效数据和存储数据分开计算

假设一行有效长度为 L，UB 中补齐后行步长为 S，则第 r 行的地址偏移为 `r × S`，count 型归约的参与数量仍为 L。若使用 Pattern 型接口处理补齐 shape，padding 必须符合该接口和归约的数学语义。字节长度与元素数量不可混传。

多行或多组归约的输出槽位也有对齐要求。仅因为每组产生一个标量，不能推断目的地址可按一个标量的字节数紧密推进；写入布局与后续读取索引应一起确定。

## 临时空间

两个同名重载未必使用相同类型的工作区：不能把 count 型重载的模板化 LocalTensor 临时区与 Pattern 重载的 `LocalTensor<uint8_t>` 按名称混用。

对提供空间查询接口的高阶 API，应按目标 shape、dtype 和合法参数调用相应 Tiling 接口计算需求。省略 sharedTmpBuffer 参数仅表示由框架取得临时空间，不能将其视为零 UB 占用；框架获取方式需要遵循目标 SDK 的临时空间契约。随意分配一个 TBuf 不能独立证明它会被框架使用。

当前知识库的 Pattern ReduceMax 契约将 `isReuseSource` 标为预留且要求 false，因此不能直接套用 reference 示例中的 true。临时区大小也不能统一写死为 4KB 或 32KB。

## 验证

覆盖多行、非对齐列数、多个紧邻输出槽位和最大 tile。检查结果、输入是否允许被复用、UB 峰值以及每个重载的实际声明。参见 [基础 ReduceMax](../../apis/simd_api/basic_api/memory_vector_compute/reduction_compute/reduce_max.md)、[Pattern ReduceMax](../../apis/simd_api/adv_api/reduction_operations/reduce_max_interface/reduce_max.md)和[临时空间查询](../../apis/simd_api/adv_api/reduction_operations/reduce_max_interface/get_reduce_max_max_min_tmp_size.md)。
