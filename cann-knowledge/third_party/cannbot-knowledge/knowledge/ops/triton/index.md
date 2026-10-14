# Triton 算子知识

Triton-Ascend 算子开发知识：上游 API 契约、AscendNPU-IR 方言语义、设计期约束、优化期改写、单算子实现手册与实测性能案例。

- [API](apis/index.md)：tl.* 核心 API、al.* Ascend 扩展、API/dtype 支持矩阵与 AscendNPU-IR 方言契约。
- [指南](guides/index.md)：开发工作流、设计期约束、单算子实现手册、算子专项设计与 GPU→NPU 迁移对照。
- [优化方法](optimizations/index.md)：跨算子优化技术专题、优化期改写点库与实测经验聚合。
- [Runbook](runbooks/index.md)：按 shape 特征驱动的实测性能案例。
- [示例](examples/index.md)：代码骨架模板与逐步教学 walkthrough。

> **设计期与优化期知识不可混用**：两者适用前提互斥，检索时按 `task.design` /
> `task.optimization` 标签过滤（设计期见 guides/constraints/（`guides/constraints/index.md`），
> 优化期见 optimizations/operator_optimization/（`optimizations/operator_optimization/index.md`））。
