# Triton 指南

任务式开发指南：环境/调试/迁移工作流、设计期约束、单算子实现手册与上游平台契约。

- [IR 工作流](ir/index.md)：Triton-Ascend 编译 IR 的提取、分析与迁移工作流。
- [开发工作流](workflows/index.md)：环境搭建、调试、Profiling、GPU→NPU 迁移。
- [设计期约束](constraints/index.md)：跨算子与分类别的设计期硬约束（stage: design，53 张）。
- [单算子实现手册](operator_playbooks/index.md)：按算子类别组织的实现 playbook（59 张）。
- [算子专项设计](operator_design/index.md)：按算子类型的设计页，含经验路由与适用条件（14 张）。
- [GPU→NPU 迁移对照](migration/index.md)：含实测 Diff 点与收益的迁移对照卡（5 张）。
- [上游平台契约](triton_ascend/index.md)：上游 triton-ascend 锁定 commit 的能力与 API 契约（4 张）。
