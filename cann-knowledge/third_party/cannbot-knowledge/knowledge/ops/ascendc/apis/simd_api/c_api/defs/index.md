# 基础定义

- [常量](constant/index.md)：共 2 张知识卡（如 ASC_C_API_MRGSORT_ELEMENT_LEN、ASC_VDEQ_SIZE）。
- [枚举](enum/index.md)：共 6 张知识卡（如 asc_hf32_round_mode、asc_l13d_repeat_direction、asc_load_l2_cache_mode 等）。
- [寄存器类型](type/index.md)：矢量数据寄存器用于存储矢量数据，其长度为VL（Vector Length）。在Ascend 950PR/Ascend 950DT产品中，VL为256字节，可存储VL / sizeof(dtype)个dtype类型的元素。例如，vector_float可存储256 / sizeof(float) = 64个float类型的元素。
- [联合体](union/index.md)：共 5 张知识卡（如 asc_fill_value_config、asc_l13d_fmatrix_config、asc_load3d_v2_config 等）。
