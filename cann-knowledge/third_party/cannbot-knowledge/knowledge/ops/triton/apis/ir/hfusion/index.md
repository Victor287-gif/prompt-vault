# HFusion 方言

HFusion 是前端方言到 HIVM 之间的结构化中间层，以 OpDSL 声明式定义，lowering 到 HIVM 向量/宏指令。

- [overview](overview.md)：方言定位、六大操作分类（42 个操作）与类型系统
- [elementwise_ops](elementwise_ops.md)：六个逐元结构化操作与函数选择属性（unary_fn/binary_fn/compare_fn 等）
- [reduction_ops](reduction_ops.md)：reduce_with_index 带索引归约与 cumsum/cumprod 累积操作
- [matmul_ops](matmul_ops.md)：matmul_mx 微缩放 FP8 矩阵乘与 group_matmul MoE 分组矩阵乘
- [data_movement_ops](data_movement_ops.md)：数据重排：interleave/deinterleave/flip/gather/arange 与 Linalg 承载方式
- [memory_ops](memory_ops.md)：九个 memref 级访存操作：聚集/散射、间接访存、SIMT gatherT 与原子操作
- [special_ops](special_ops.md)：十一个特殊操作：print/assert、barrier、sort、histogram、embedding_gather 等
- [attributes_enums](attributes_enums.md)：全部控制枚举与属性：函数选择、舍入、原子类型、FusionKind、访存修饰
- [transforms](transforms.md)：40+ 变换 Pass 分类清单与 fuse-ops/auto-vectorize/flatten-ops 关键选项
