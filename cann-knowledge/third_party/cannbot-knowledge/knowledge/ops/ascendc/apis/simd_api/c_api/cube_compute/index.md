# 矩阵计算

- [asc_disable_hf32](asc_disable_hf32.md)：本接口用于设置Mmad计算关闭HF32模式，其作用与asc_enable_hf32相反，两个接口不同时生效。关闭HF32模式后，L0A Buffer与L0B Buffer中的float数据在参与Mmad计算之前不做舍入处理。
- [asc_disable_hif8](asc_disable_hif8.md)：本接口用于设置Mmad计算关闭HiF8模式（与asc_enable_hif8相反），调用此接口后L0A Buffer/L0B Buffer中的FP8数据在参与Mmad计算之前不会被转化为hifloat8_t类型数据。
- [AscEnableFp8](asc_enable_fp8.md)：用于设置Mmad计算开启FP8模式，关闭HiF8模式，使L0A/L0B Buffer中的FP8数据在参与Mmad计算前不转化为hifloat8_t类型。
- [AscEnableHf32](asc_enable_hf32.md)：设置Mmad计算开启HF32模式，开启后Mmad计算FP32数据性能提升但会带来一定精度损失。
- [AscEnableHf32Trans](asc_enable_hf32_trans.md)：设置HF32模式舍入方式。使用该接口前需要先使用asc_enable_hf32开启HF32模式。
- [AscEnableHif8](asc_enable_hif8.md)：用于设置Mmad计算开启HiF8模式，开启后L0A Buffer/L0B Buffer中的8bit数据将在参与矩阵乘法运算前被转化为hifloat8_t类型数据。
- [AscMmad](asc_mmad.md)：Ascend C矩阵乘加（Mmad）核心计算接口，封装昇腾NPU硬件矩阵乘加能力，用于全连接层、卷积层等神经网络算子开发。
- [AscMmadMx](asc_mmad_mx.md)：完成带量化系数的矩阵乘加计算（MmadMx），左矩阵A和右矩阵B分别配有量化系数矩阵ScaleA和ScaleB，按K维度每32个元素共享一个量化系数。
- [AscMmadSparse](asc_mmad_sparse.md)：完成稀疏矩阵乘加计算，左矩阵A为稠密矩阵、右矩阵B为稀疏矩阵，B在输入前按稠密算法完成稠密化并生成索引矩阵用于A的稠密化。
- [AscSetFp32Mode](asc_set_fp32_mode.md)：本接口用于设置Mmad计算关闭HF32模式，其作用与asc_enable_hf32相反，两个接口不同时生效。关闭HF32模式后，L0A Buffer与L0B Buffer中的float数据在参与Mmad计算之前不做舍入处理。
- [AscSetHf32RoundMode](asc_set_hf32_round_mode.md)：设置HF32模式舍入方式。使用该接口前需要先使用asc_enable_hf32开启HF32模式。
- [AscSetMmadDirectionM](asc_set_mmad_direction_m.md)：设置矩阵乘加计算（Mmad）时逐列生成结果分形，只改变L0C Buffer上分形生成顺序，不改变计算结果。适用于UnitFlag场景下与搬出顺序一致以提升性能。
- [AscSetMmadDirectionN](asc_set_mmad_direction_n.md)：设置矩阵乘加计算（Mmad）时逐行生成结果分形，只改变L0C Buffer上分形生成顺序，不改变计算结果。适用于UnitFlag开启且开启Nz2ND随路格式转换或B8/B4量化触发Channel Merge的场景。
