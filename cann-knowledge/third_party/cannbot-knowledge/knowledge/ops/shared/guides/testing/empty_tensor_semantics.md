---
type: Guide
title: 空张量、零维标量与非空输出的区别
description: 区分 rank 为零与元素数量为零，并按算子 shape 约束判断空输入的输出和计算语义。
tags:
- task.testing
- task.design
- topic.programming_model
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-st-design/references/empty-tensor-guide.md
  role: primary
created_at: '2026-09-10T12:59:46Z'
updated_at: '2026-09-10T12:59:46Z'
platforms:
- agnostic
---

# 空张量、零维标量与非空输出的区别

零维标量的 shape 为 `[]`，通常包含一个元素；空张量的 shape 含长度为零的维度，例如 `[3,0,5]`，元素数量为零。两者不能用“维度为 0”混称。

空输入不一定产生空输出。对矩阵乘 `[M,K] × [K,N] → [M,N]`，M 或 N 为零时输出为空；K 为零而 M、N 非零时，输出仍有 M×N 个元素，其数值由算子对空归约的定义决定。融合 bias、残差或系数时还应保留这些项的语义，不能简单提前返回而留下未初始化输出。

从正常 shape 改出空 shape，需要共同维护耦合维度：例如矩阵乘两侧 K 必须一起修改，输出 M/N 与输入对应关系也需保持。广播、参数范围和算子是否允许空维度仍要重新检查，不能认为正常用例通过就保证置零后自然合法。

| 情况 | 需要区分的行为 |
|---|---|
| 入空出空 | 元信息、返回结构及合法的空输出处理 |
| 入空出非空 | 空归约结果、融合项及完整输出写入 |
| 部分输入为空 | 多输入 shape 关系与可选项语义 |
| 算子不支持该空维度 | 明确的参数拒绝行为 |

覆盖数量应由实际分支和约束决定。每类只保留一个用例可以是一种测试生成策略，但不证明所有 dtype、属性和空维组合都已覆盖。
