# 其他方言

- [scope_dialect](scope_dialect.md)：`scope.scope` 区域作用域抽象：tcore_type 属性与 outline/inline 变换
- [symbol_dialect](symbol_dialect.md)：动态形状表示：symbolic_int 符号整数与 bind_symbolic_shape 绑定
- [memrefext_dialect](memrefext_dialect.md)：`memref_ext.alloc_workspace` 工作空间分配与 HACC 协作
- [mathext_dialect](mathext_dialect.md)：NPU 特有数学操作：ilogb/ldep/divfhp 高精度除法
- [hmap_dialect](hmap_dialect.md)：多卡集合通信：hmap.all_to_allv 不均匀分片全互联通信
- [annotation_dialect](annotation_dialect.md)：`annotation.mark` 值标注机制：驱动 Bufferization、缓存 IO 与多缓冲标记
- [ascend_dpx_dialect](ascend_dpx_dialect.md)：SIMT 虚拟 ISA：线程索引、Warp Shuffle/归约、原子操作，lowering 到 LLVM 前最后 IR 层
