# asccopygm2l1

- [asc_copy_gm2l1_arch_3510（2D矩阵搬运模式）](asc_copy_gm2l1_2d_arch_3510.md)：将数据从Global Memory搬运到L1 Buffer。以512字节的分形为单位，从GM中按行列方向的起始位置、步长切分提取多个分形矩阵搬运到L1 Buffer，搬运过程中不支持分形格式转换，用于矩阵计算所需的2D格式数据加载。
- [asc_copy_gm2l1](asc_copy_gm2l1_arch_2201.md)：将矩阵数据从Global Memory搬运到L1 Buffer中，并支持不同类型的pad模式。
- [asc_copy_gm2l1_arch_3510（高维切分数据搬运）](asc_copy_gm2l1_highdim_split_arch_3510.md)：将数据从Global Memory搬运到L1 Buffer，数据搬运时格式和内容保持不变。高维切分是指能够通过配置数据块个数、单个数据块长度、地址偏移等搬运参数实现非连续搬运。并且此接口支持在搬运过程中按通道padding模式对32字节粒度的数据插入或移除padding。其中padding值需要通过asc_set_gm2l1_pad设置。
