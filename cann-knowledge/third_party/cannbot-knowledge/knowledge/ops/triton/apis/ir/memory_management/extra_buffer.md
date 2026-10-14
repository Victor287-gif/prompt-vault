---
type: API
title: 额外缓冲区 Extra Buffer（temp_buffer 机制与缓冲区尺寸推断 Pass）
description: 需要临时缓冲区的 HIVM 操作的 temp_buffer 分配机制：hivm-alloc-extra-buffer / outline-alloc-in-VF
  / constantize-buffer-size / auto-infer-buffer-size 等 Pass、temp_buffer IR 形态与 ExtraBufferOpInterface。
tags: [task.lookup, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# 额外缓冲区 — Extra Buffer

某些 HIVM 操作执行时需要额外临时缓冲区（temp_buffer）：类型转换的中间存储、broadcast 操作的展开缓冲区等。Extra Buffer 相关 Pass 负责分配和管理这些临时存储。

## 相关 Pass

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hivm-alloc-extra-buffer` | FuncOp | 为声明了额外缓冲区需求的操作（带 broadcast 的二元 op、部分类型转换）自动分配 temp_buffer |
| `hivm-outline-alloc-in-VF` | ModuleOp | 把 VF（Vector Function）中静态形状的 `memref.alloc` 提到函数外，减少 VF 内部分配开销 |
| `hivm-constantize-buffer-size` | FuncOp | 动态形状缓冲区常量化：对原始形状做上界估计，成功则新建静态形状分配，再用 subview 转回原始形状 |
| `hivm-set-buffer-size` | FuncOp | 设置缓冲区大小 |
| `hivm-auto-infer-buffer-size` | FuncOp | 通过插入 `annotation.mark` 自动推断缓冲区大小 |

## temp_buffer 机制

操作通过 `temp_buffer` 操作数声明额外临时存储：

```mlir
// 带 temp_buffer 的 vsub（broadcast 场景）
hivm.hir.vsub ins(%alloc, %alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>,
                                    memref<64x1xf32, #hivm.address_space<ub>>)
  outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>)
  temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>)
  broadcast = [1]
```

典型场景：

1. **Broadcast**：二元操作某输入需 broadcast 时，temp_buffer 存 broadcast 后的中间结果；
2. **类型转换**：某些转换需要中间存储；
3. **Decompose**：`hivm-decompose-op` 在动态情况下创建带 `buffer_size_in_byte` 标记的额外缓冲区。

## ExtraBufferOpInterface

HIVM 操作接口，用于声明操作是否需要额外缓冲区及其大小。实现该接口的操作在 `hivm-alloc-extra-buffer` Pass 中被自动分配 temp_buffer。

内存规划侧注意：temp_buffer 与主缓冲区独立分配偏移，Inplace 策略互不影响，见 [memory_planning.md](memory_planning.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L320-L324](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L320-L324)
- [Passes.td#L326-L330](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L326-L330)
- [Passes.td#L339-L350](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L339-L350)
- [Passes.td#L409-L412](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L409-L412)
- [Passes.td#L636-L642](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L636-L642)
- [ExtraBufferOpInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.td)
