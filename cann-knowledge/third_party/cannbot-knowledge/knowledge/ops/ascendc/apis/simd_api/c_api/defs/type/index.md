# 寄存器类型

- [Reg数据类型定义](data_type_definition.md)：矢量数据寄存器用于存储矢量数据，其长度为VL（Vector Length）。在Ascend 950PR/Ascend 950DT产品中，VL为256字节，可存储VL / sizeof(dtype)个dtype类型的元素。例如，vector_float可存储256 / sizeof(float) = 64个float类型的元素。
