# API 参考

Triton Ascend API 速查与参考。

## 速查与矩阵

- [triton_api_cheatsheet](triton_api_cheatsheet.md)：Triton/Ascend API 完整速查：Kernel 声明、Program Identity、内存、计算、同步、Ascend 扩展（dl/tle）
- [api_support_matrix](api_support_matrix.md)：全部 Triton Python API 在 Ascend NPU 上的支持状态矩阵（8 大类 + 通用约束）
- [data_type_matrix](data_type_matrix.md)：数据类型在 load/store/dot/数学/归约/cast/原子操作上的支持矩阵与选型建议

## 核心 API（tl.*）

- [内存操作 API（tl.load / tl.store / block_ptr）](tl_memory_ops.md)：Triton-Ascend 内存操作 API：tl.load、tl.store、tl.make_block_ptr、tl.advance 的签名、参数语义、dtype 支持与 NPU 特有差异（care_padding、离散访存）。
- [tl_math_ops](tl_math_ops.md)：算术、初等函数与融合运算（fma/sigmoid/softmax）签名与 NPU 精度提升规则
- [tl_reduction_ops](tl_reduction_ops.md)：reduce/sum/max/argmax/xor_sum；窄整数不提升 int32、bf16 提升 fp32 等适配差异
- [tl_linear_algebra_ops](tl_linear_algebra_ops.md)：tl.dot / tl.dot_scaled：精度模式映射（tf32→hf32）、dtype 约束与 ND2NZ 布局转换
- [tl_atomic_ops](tl_atomic_ops.md)：atomic_add/max/min/cas 等签名、NPU dtype 支持表、sem/scope 仅默认值限制
- [tl_shape_ops](tl_shape_ops.md)：broadcast/reshape/trans/join/split 等形状操作签名与 can_reorder 语义
- [tl_scan_sort_ops](tl_scan_sort_ops.md)：associative_scan/cumsum/histogram/sort（bitonic 实现，2 的幂限制）
- [tl_comparison_logical_ops](tl_comparison_logical_ops.md)：比较/逻辑/移位/where：有序浮点比较、where 非短路求值、bf16 比较提升 fp32

## Ascend 扩展（al.*）

- [ascend_ext_overview](ascend_ext_overview.md)：扩展 API 全景：导入方式、与标准 Triton API 的差异、完整分类索引
- [ascend_ext_pipe_core](ascend_ext_pipe_core.md)：PIPE / CORE / MODE / IteratorType 枚举取值与语义
- [ascend_ext_fixpipe](ascend_ext_fixpipe.md)：fixpipe：L0C→UB 搬运（仅 910_95）、NZ↔ND 布局转换、预量化与预 ReLU 融合
- [ascend_ext_sync](ascend_ext_sync.md)：sync_block_set/wait/all 跨核 Event/Flag 同步原语与 Vector 内部 debug_barrier
- [ascend_ext_sub_vec_ops](ascend_ext_sub_vec_ops.md)：sub_vec_id/sub_vec_num：CV 融合算子中多 Vector 核负载划分
- [ascend_ext_custom_op](ascend_ext_custom_op.md)：register_custom_op / al.custom 自定义算子注册调用与 4 个内置索引算子
- [ascend_ext_buffer_model](ascend_ext_buffer_model.md)：Buffer 编程模型：bl.alloc / to_buffer / to_tensor / subview 显式地址空间控制
- [ascend_ext_aux_ops](ascend_ext_aux_ops.md)：parallel（bind_sub_block）/ compile_hint / multibuffer 辅助操作
- [ascend_ext_vec_ops](ascend_ext_vec_ops.md)：insert_slice/extract_slice/get_element/sort/flip 与增强 cast（overflow_mode）
- [ascend_ext_mem_ops](ascend_ext_mem_ops.md)：index_put / gather_out_to_ub / scatter_ub_to_out / index_select_simd 索引读写
- [ascend_ext_libdevice](ascend_ext_libdevice.md)：libdevice 数学函数库（HMF 硬件加速）、TRITON_ENABLE_LIBDEVICE 双路径与精度注意

## AscendNPU-IR 方言契约

- [ir/](ir/index.md)：AscendNPU-IR：HIVM/HFusion/HACC 等方言操作语义、编译流水线与内存管理（109 张）
