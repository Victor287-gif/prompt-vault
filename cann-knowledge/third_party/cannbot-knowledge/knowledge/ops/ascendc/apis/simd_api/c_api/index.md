# API列表

- [原子操作](atomic/index.md)：共 19 张知识卡（如 原子操作概述、关键特性说明 等）。
- [API列表](c_api.md)：C API文档按最细粒度公共头文件归类。除试验接口外，整体使用时可以包含c_api/asc_simd.h。
- [缓存控制](cache_ctrl/index.md)：共 9 张知识卡（如 asc_datacache_preload、AscDcci、asc_dci（缓存控制） 等）。
- [矩阵计算](cube_compute/index.md)：共 13 张知识卡（如 asc_disable_hf32、asc_disable_hif8、AscEnableFp8 等）。
- [矩阵计算搬运](cube_datamove/index.md)：共 69 张知识卡（如 asc_copy_gm2l0a、asc_copy_gm2l0b、asc_copy_gm2l1_align 等）。
- [基础定义](defs/index.md)：共 14 张知识卡，分布在 4 个子目录（constant、enum、type 等）。
- [废弃接口](deprecated_interface/index.md)：废弃接口索引页，汇总矢量计算asc_abs_sync与原子操作asc_set_atomic_add_int等已废弃接口。（共 219 张知识卡：AscAbsSync、asc_add_relu_sync（废弃）、asc_add_scalar_sync（废弃） 等）
- [其他操作](experimental/index.md)：本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。
- [通用说明和约束（C API）](general_description_and_constraints.md)：AI Core内外包含多级存储单元，各存储单元的主要用途如下：
- [Reg矢量计算](reg_compute/index.md)：共 187 张知识卡，分布在 19 个子目录（load、reg_addr_reg、reg_arith 等）。
- [标量计算](scalar_compute/index.md)：共 12 张知识卡，分布在 4 个子目录（scalar_bit、scalar_convert、scalar_load 等）。
- [特殊寄存器访问](spr/index.md)：共 7 张知识卡（如 asc_get_ctrl、asc_get_saturation_flag、asc_get_saturation_strategy 等）。
- [同步控制](sync/index.md)：共 22 张知识卡（如 AscLock、asc_sync（同步）、AscSyncBlockArrive 等）。
- [工具接口](utils/index.md)：共 17 张知识卡，分布在 3 个子目录（sys_init、sys_misc、sys_var）。
- [Memory矢量计算](vector_compute/index.md)：共 100 张知识卡，分布在 11 个子目录（vector_arith、vector_broadcast、vector_compare 等）。
- [矢量数据搬运](vector_datamove/index.md)：共 23 张知识卡（如 asc_copy、asc_copy_ub2l1、asc_copy_ub2ub 等）。

- [C API](c_api_reference.md)：C API开放芯片完备编程能力，支持以数组形式分配内存，一般基于指针编程。提供与业界一致的C语言编程体验。
