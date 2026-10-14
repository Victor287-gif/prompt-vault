---
type: API
title: HIVM DMA Load（hivm.hir.load）
description: hivm.hir.load 操作语义：GM→UB 的 MTE2 数据加载，操作数与 pad_mode/pad_value/eviction_policy/init_out_buffer
  属性表、验证约束与 lowering 规则。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM DMA Load（hivm.hir.load）

`hivm.hir.load` 把数据从全局内存（GM）加载到统一缓冲区（UB），映射到 MTE2 Pipeline，是大多数 kernel 中最高频的 DMA 操作。当前仅支持 GM→UB 一条通路（加载到 L1 用 [hivm.hir.nd2nz](dma_nd2nz.md)，本地层级间搬运用 [hivm.hir.copy](dma_copy.md)）。对应 Triton 前端的 `tl.load`。

## MLIR 语法

```mlir
hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
              outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)

%res = hivm.hir.load ins(%src : tensor<16x16xf32>)
                     outs(%dst : tensor<16x16xf32>)
                     pad_mode = #hivm.padmode<PadValue>
                     pad_value = %val : f32
                     -> tensor<16x16xf32>
```

## 操作数与属性

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源缓冲区，memref 语义下必须为 GM 地址空间 |
| `dst` | TensorOrMemref | 是 | 目标缓冲区，memref 语义下必须为 UB 地址空间 |
| `pad_mode` | HIVM_PadModeAttr | 否 | `PadNull`(0) / `PadFirstElem`(1) / `PadValue`(2) |
| `pad_value` | AnyType | 否 | 填充值，元素类型须与 src/dst 相同；仅给 pad_value 时自动推断为 PadValue 模式 |
| `left_padding_num` / `right_padding_num` | Index/AnyType | 否 | 左/右填充元素数，仅在 pad_mode 设置时有意义 |
| `init_out_buffer` | BoolAttr | 否 | 加载前将目标缓冲区初始化（默认 false） |
| `init_condition` | AnyType | 否 | 条件化初始化 |
| `eviction_policy` | HIVM_EvictionPolicyAttr | 否 | `EvictFirst`(0) 优先驱逐（一次性数据）/ `EvictLast`(1) 最后驱逐（复用数据） |
| `result_tensor` | Optional<AnyRankedTensor> | 结果 | Tensor 语义下的结果 |

元素类型约束（操作数 0）：`i8/ui8/i16/ui16/f16/bf16/i32/ui32/f32/ui64/i64/f8E4M3FN/f8E5M2`。最大 rank 3（`StaticMaxRankTrait<3>`）。

## IR 示例

带 Padding（源 16x15 补到 16x16，用首元素填充）：

```mlir
hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
              outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
              pad_mode = #hivm.padmode<PadFirstElem>
```

带驱逐策略：

```mlir
hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
              outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
              eviction_policy = #hivm.evictionpolicy<EvictLast>
```

## 验证器约束

1. src 与 dst 元素类型、rank 必须一致；未设 pad_mode 时形状必须相同。
2. `pad_mode = PadValue` 时 `pad_value` 必填且类型与 dst 元素类型一致。
3. memref 语义下 src 必须为 GM、dst 不得为 GM（报错样例：`only support src == gm and dst != gm currently!`）。
4. Tensor 语义下 result_tensor 的元素类型、rank 与 dst 一致，未设 pad_mode 时形状兼容。

## Lowering 与映射

- `tt.load` 的 `padding_option` / `other` 参数映射为 `pad_mode` / `pad_value`。
- 最终 lowering 为库函数 `load_gm_to_ubuf_{rank}d_{datatype}`。
- HFusion 加载按目标地址空间与布局需求选择生成 `hir.load` 或 `hir.nd2nz`。

## 排障与选型

- 数据供 Vector 核心使用 → `hir.load`（GM→UB）；供 Cube 矩阵乘使用 → `hir.nd2nz`（GM→L1 + NZ 布局）。
- `PadFirstElem` 用源首元素填充，适合边界复制场景；`PadValue` 用指定值填充。
- `init_out_buffer = true` 用于加载数据无法完全覆盖目标缓冲区时保证缓冲区干净。

相关：[dma_store.md](dma_store.md)、[dma_overview.md](dma_overview.md)、[dma_padding_quantization.md](dma_padding_quantization.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMDMAOps.td#L53-L136](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L53-L136)
- [HIVMAttrs.td#L330-L349](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L330-L349)
- [HIVMAttrs.td#L356-L372](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L356-L372)
- [HIVMDMAOps.cpp#L219-L272](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L219-L272)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
- [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir)
