# IR 工作流

Triton-Ascend 编译 IR 的提取、分析与迁移工作流。IR 方言各操作的语义契约见 apis/ir/（`../../apis/ir/index.md`）。

- [IR 提取（run_and_extract.sh）](ir_extraction.md)：从 Triton-Ascend 编译过程提取 TTIR / TTAdapter / last_pass.mlir 的完整流程。
- [IR 分析工作流](ir_analysis_workflow.md)：IR 优先的性能分析决策流程。
- [GPU Triton → Ascend 910_95 迁移工作流](migration_agent_workflow.md)：最小变更的迁移工作流。
