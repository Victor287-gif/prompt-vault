# asccopygm2l1nd2nz

- [asc_copy_gm2l1_nd2nz](asc_copy_gm2l1_nd2nz_arch_2201.md)：将数据从Global Memory (GM)搬运到Level 1 cache (L1)，支持在数据搬运时进行ND格式到NZ格式的转换。
- [asc_copy_gm2l1_nd2nz_arch_3510](asc_copy_gm2l1_nd2nz_arch_3510.md)：将数据从Global Memory搬运到L1 Buffer，并在搬运过程中将源端按N维连续排布的ND格式数据转换为L1 Buffer侧的Nz分形排布，使其满足后续矩阵计算对分形列连续性的要求。该接口也适用于卷积格式NHWC（ND）转换为NC1HWC0格式等转换场景。
