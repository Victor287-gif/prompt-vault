# asccopyl12l0b

- [asc_copy_l12l0b_arch_3510（2D矩阵搬运）](asc_copy_l12l0b_2d_arch_3510.md)：负责完成矩阵计算所需的2D格式数据的搬运，搬运粒度为大小为512字节的数据分形，支持非转置与转置矩阵从L1 Buffer到L0B buffer的数据搬运。
- [asc_copy_l12l0b_arch_3510（3D矩阵搬运）](asc_copy_l12l0b_3d_arch_3510.md)：3D卷积数据搬运是用于将NC1HWC0格式的Feature Map完成Image to Column展开，按卷积步长、卷积核大小、膨胀系数配置生成img2col二维矩阵，再从展开后的矩阵中选取指定数据块搬运到L0B Buffer，用于3D卷积场景的矩阵计算数据加载。
- [asc_copy_l12l0b](asc_copy_l12l0b_arch_2201.md)：用于搬运存放在L1 Buffer里的512B大小的矩阵到L0B Buffer里。包含2D格式搬运、3D格式搬运。
