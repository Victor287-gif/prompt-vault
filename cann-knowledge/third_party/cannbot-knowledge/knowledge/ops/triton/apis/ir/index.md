# IR 分析与 AscendNPU-IR

本分区治理自 `ops/triton-latency-optimizer` 的 IR 知识源（官方上游源码仓：[AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)），覆盖两个层面：

- **IR 工作流**：如何从 Triton-Ascend 编译过程提取 `last_pass.mlir`，并以 IR 为主、msprof 为辅做性能分析与迭代优化。
- **AscendNPU-IR 方言参考**：HIVM/HFusion/HACC/Triton 方言的操作语义、编译流水线各阶段变换、内存管理机制——分析提取出的 IR 时按 op 名检索查阅。

## 核心工作流卡

* ir_extraction（`../../guides/ir/ir_extraction.md`） — 从 Triton-Ascend 编译过程提取 TTIR / TTAdapter / last_pass.mlir 的完整流程：环境变量、bishengir-compile 参数、dump flag 探测、autotune 去重与验证诊断。
* ir_analysis_workflow（`../../guides/ir/ir_analysis_workflow.md`） — 以 last_pass.mlir 为主、msprof 为辅的性能分析决策流程：IR 四项检查点、msprof 指标判读与 caveat、瓶颈→优化动作映射，以及 Phase 4 IR 多轮迭代规则。
* migration_agent_workflow（`../../guides/ir/migration_agent_workflow.md`） — GPU Triton kernel 迁移到 Ascend 910_95 NPU 的三阶段工作流：Phase 1 最小变更跑通、Phase 2 单点迭代优化、Phase 3 910_95 专属优化，附精度验证模板。

## 方言与参考分区

- [hivm](hivm/index.md)：HIVM 方言完整操作语义（45 卡）：DMA、向量、宏操作、同步、Custom、属性类型系统与内建指令
- [hfusion](hfusion/index.md)：HFusion 方言（9 卡）：六大操作族、枚举属性与 40+ 变换 Pass
- [hacc](hacc/index.md)：HACC 方言（6 卡）：Host/Device 函数管理、设备规格、Kernel 参数与 Host-Device 绑定
- [triton_dialects](triton_dialects/index.md)：Triton 三层方言（6 卡）：tt/TTIR、ttg/TTGIR 与 Gluon 的操作、类型与布局编码
- [other_dialects](other_dialects/index.md)：辅助方言（7 卡）：Scope/Symbol/MemRefExt/MathExt/HMAP/Annotation/AscendDPX
- [compilation_pipeline](compilation_pipeline/index.md)：编译流水线（17 卡）：阶段变换、Pass 依赖、Pipeline 选项、last_pass.mlir 定位，以及 Triton-Ascend 前端编译链（AST→TTIR→Linalg→二进制）
- [architecture](architecture/index.md)：NPU 硬件架构（4 卡）：内存层级、Pipeline 执行模型与 Fractal 数据布局
- [memory_management](memory_management/index.md)：内存管理（8 卡）：Bufferization、内存规划、多缓冲、对齐与 Workspace
- [reference](reference/index.md)：速查参考（4 卡）：操作索引、枚举速查、硬件规格精确值与术语表
