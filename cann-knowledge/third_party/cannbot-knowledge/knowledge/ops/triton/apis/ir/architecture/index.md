# NPU 硬件架构（IR 视角）

- [npu_hardware_overview](npu_hardware_overview.md)：AI Core 组成、全型号规格表、Reg-based/Mem-based 架构分类（入口卡）
- [memory_hierarchy](memory_hierarchy.md)：各级存储容量与对齐、AddressSpace 枚举、地址空间→Pipeline 映射与数据通路
- [pipeline_execution_model](pipeline_execution_model.md)：多 Pipeline 并行执行模型：PIPE 枚举、操作→Pipe Trait 绑定与跨 Pipeline 同步
- [data_layout](data_layout.md)：Cube 单元 Fractal 布局体系：DataLayout 枚举、nd2nz/nz2nd/fixpipe 布局转换
