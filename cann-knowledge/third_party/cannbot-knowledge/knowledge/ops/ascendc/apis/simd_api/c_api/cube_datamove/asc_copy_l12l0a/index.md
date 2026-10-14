# asccopyl12l0a

- [asc_copy_l12l0a_arch_3510（2D矩阵搬运）](asc_copy_l12l0a_2d_arch_3510.md)：负责完成矩阵计算所需的2D格式数据的搬运，搬运粒度为大小为512字节的数据分形，支持非转置与转置矩阵从L1 Buffer到L0A Buffer的数据搬运。
- [asc_copy_l12l0a_arch_3510（3D矩阵搬运）](asc_copy_l12l0a_3d_arch_3510.md)：3D卷积数据搬运是用于将NC1HWC0格式的Feature Map完成Image to Column展开，然后再从展开后的二维矩阵中选取指定数据块搬入对应内存位置。见图1。
- [asc_copy_l12l0a](asc_copy_l12l0a_arch_2201.md)：用于搬运存放在L1 Buffer里的512B大小的矩阵到L0A Buffer里。包含2D格式搬运、3D格式搬运。
