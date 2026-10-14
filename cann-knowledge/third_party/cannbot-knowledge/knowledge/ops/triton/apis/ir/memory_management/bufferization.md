---
type: API
title: Bufferization（Tensor→MemRef 转换与相关 Pass）
description: 内存管理第一阶段 Bufferization：Tensor 与 MemRef 语义对比、BufferizableOpInterface、变换前后
  IR 形态，以及 hivm-clone-tensor-empty 等四个配套 Pass 的作用。
tags: [task.lookup, topic.implementation.memory_management, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# Bufferization — Tensor 到 MemRef 的转换

Bufferization 是内存管理的第一阶段：把基于 Tensor 的 SSA 值转换为基于 MemRef 的缓冲区操作，消除 Tensor 值语义、引入 `memref.alloc` 显式分配逻辑缓冲区，为后续 [内存规划](memory_planning.md) 与物理地址分配奠基。

## Tensor vs MemRef

| 特性 | Tensor | MemRef |
|------|--------|--------|
| 语义 | 值语义（SSA） | 引用语义（指针） |
| 内存分配 | 隐式 | 显式 `memref.alloc` |
| 地址空间 | 无 | `#hivm.address_space<...>` |
| 可变性 | 不可变 | 可变 |
| 生命周期 | SSA 支配 | alloc/dealloc 决定 |

MLIR Bufferization 框架通过 `BufferizableOpInterface` 让每个操作自定义转换行为：Tensor 操作数如何映射到 MemRef、是否支持 inplace（输出复用输入缓冲区）、转换后的 IR 结构。

## 变换效果

变换前（Tensor IR）：
```mlir
%0 = tensor.empty() : tensor<16x16xf16>
%1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>)
                  outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
```

变换后（MemRef IR）：
```mlir
%0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
hivm.hir.vadd ins(%arg0, %arg1 : memref<16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16xf16, #hivm.address_space<ub>>)
              outs(%0 : memref<16x16xf16, #hivm.address_space<ub>>)
```

## 相关 Pass

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hivm-clone-tensor-empty` | FuncOp | 为不同 HIVM 操作的输出克隆各自的 `tensor.empty`，保证每个输出独立分配，避免 Bufferization 时 inplace 冲突 |
| `hivm-clone-scf-if-yield-operand` | FuncOp | 克隆 `scf.if` 的 yield 操作数：当多个 yield 返回同值或该值在 if 之后仍被使用时，防止 PlanMemory 的 inplace 优化出错 |
| `hivm-opt-func-output` | ModuleOp | Bufferization 后优化函数输出：返回的 MemRef 已经由参数传入时，移除多余的地址返回 |
| `hivm-inline-otf-load-store` | FuncOp | 即时内联 Load/Store，减少不必要的缓冲区中间拷贝 |

## Inplace Bufferization

关键优化：操作输出可复用输入缓冲区时省掉额外 `memref.alloc`。HIVM 结构化操作经 `HIVMStructuredOpInterface` 支持 inplace 分析；到 PlanMemory 阶段，inplace 进一步扩展到物理地址级——生命周期不重叠的缓冲区共享同一物理偏移（见 [memory_planning.md](memory_planning.md)）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td#L102-L107](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L102-L107)
- [Passes.td#L121-L128](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L121-L128)
- [Passes.td#L352-L357](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L352-L357)
- [Passes.td#L660-L666](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L660-L666)
- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td)
- [buffer-opt.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/buffer-opt.mlir)
