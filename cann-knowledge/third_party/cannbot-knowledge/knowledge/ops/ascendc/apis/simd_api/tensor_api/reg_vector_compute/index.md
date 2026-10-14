# reg矢量计算（Tensor API）

- [基础算术](basic_arithmetic/index.md)：共 5 张知识卡（如 log、max、operator+ 等）。
- [概述](compare_and_select/index.md)：头文件路径："include/tensor_api/experimental/arch/vector/compare_and_select.h"。
- [数据填充](data_padding/index.md)：头文件路径："include/tensor_api/experimental/arch/vector/compare_and_select.h"。
- [数据重排](data_reorder/index.md)：头文件路径："include/tensor_api/experimental/arch/vector/data_reorder.h"。
- [逻辑计算](logical_compute/index.md)：头文件路径："tensor_api/experimental/vector_compute.h"。
- [mask寄存器计算](mask_reg_compute/index.md)：共 6 张知识卡（如 all_mask、deinterleave(掩码解交织)、make_mask 等）。
- [寄存器数据搬入](reg_data_load/index.md)：共 2 张知识卡（如 load、load_broadcast）。
- [寄存器数据搬出](reg_data_store/index.md)：头文件路径为："tensor_api/experimental/arch/vector/reg_data_store.h"，命名空间为asc::te::experimental。
- [reg_tensor](reg_tensor/index.md)：头文件路径为："tensor_api/arch/vector/experimental/reg_tensor.h"，命名空间为asc::te::experimental。（共 2 张知识卡：reg_pair）
- [类型转换](type_conversion/index.md)：共 2 张知识卡（如 cast、trunc）。
- [矢量计算的搬入](vector_compute_load/index.md)：共 3 张知识卡（如 copy（Global Memory到Unified Buffer的数据搬运）、copy（Unified Buffer到L1 Buffer的数据搬运）、copy（Unified Buffer到Unified Buffer的数据搬运））。
- [矢量计算的搬出](vector_compute_store/index.md)：头文件路径为：tensor_api/tensor.h，命名空间为asc::te。
