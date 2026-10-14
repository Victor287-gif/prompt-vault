# Triton 方言

Triton 原生编译链的方言体系，对应 IR 提取产物中的 TTIR（`tt.*`）与 TTGIR（`ttg.*`）层级。

- [overview](overview.md)：三层方言定位、tt→ttg→LLVM 转换链与分类速查
- [triton_ops](triton_ops.md)：Triton 方言（tt 前缀）全部操作：load/store/dot/reduce/原子/形状变换/TMA
- [triton_types_attrs](triton_types_attrs.md)：类型体系（PointerType/TensorDescType）与 12 组枚举属性
- [tritongpu_ops](tritongpu_ops.md)：TritonGPU 方言（ttg 前缀）：convert_layout、异步拷贝、共享内存管理
- [tritongpu_encodings](tritongpu_encodings.md)：布局编码属性：分布式/共享布局、swizzle 公式与昇腾 fractal_shared
- [gluon_dialect](gluon_dialect.md)：Gluon 实验性布局推断：gluon.set_auto_layout
