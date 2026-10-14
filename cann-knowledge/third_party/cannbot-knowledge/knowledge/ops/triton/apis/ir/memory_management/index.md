# 内存管理

- [overview](overview.md)：内存管理完整变换链（Bufferization→作用域推断→PlanMemory→对齐→后端 Lowering）总览
- [bufferization](bufferization.md)：Tensor→MemRef 的 Bufferization 阶段与配套 Pass
- [memory_planning](memory_planning.md)：hivm-plan-memory：两种规划模式、生命周期分析、Inplace 复用与 UB 溢出检测
- [multi_buffer](multi_buffer.md)：多缓冲流水线优化：mark/enable 两 Pass、MultiBufferStrategy 与 cv-pipelining
- [extra_buffer](extra_buffer.md)：临时缓冲区 temp_buffer 分配机制与相关 Pass
- [memory_alignment](memory_alignment.md)：硬件对齐约束的编译期保障：四个对齐 Pass 与 UB/L1/L0C 对齐数值
- [tightly_coupled_buffer](tightly_coupled_buffer.md)：Mix CV 场景 Cube↔Vector 免 GM 中转的紧耦合缓冲机制
- [workspace_management](workspace_management.md)：跨函数全局内存 Workspace 机制与三个配套 Pass
