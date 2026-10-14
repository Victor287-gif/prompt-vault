---
type: API
title: TritonGPU 方言操作详解（ttg.*）
description: TritonGPU 方言（ttg 前缀）全部操作的签名与语义：ttg.convert_layout 布局转换、异步拷贝（async_copy_global_to_local
  / async_wait / async_commit_group）、共享内存管理（local_alloc/load/store）、memdesc 子视图、软件流水线谓词、fp4_to_fp
  与 Warp 特化操作。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.implementation.synchronization]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# TritonGPU 方言操作详解（ttg.*）

TritonGPU 方言在 Triton（tt）层之上引入并行布局编码与共享内存管理（方言体系与转换链见 [overview.md](overview.md)，布局编码语义见 [tritongpu_encodings.md](tritongpu_encodings.md)）。

## 布局转换

### ttg.convert_layout

将张量从一种布局编码转换为另一种，是 TritonGPU 最核心的操作之一——实际的数据移动（如共享内存读写）在此发生。`Pure`，形状与元素类型不变。

```mlir
%result = ttg.convert_layout %src : tensor<128xf32, #blocked> -> tensor<128xf32, #dot_op>
```

## 异步拷贝（计算能力 >= 80）

| 操作 | 语义 |
|------|------|
| `ttg.async_copy_global_to_local` | 全局内存 → 共享内存异步拷贝；输入 `src: TT_PtrTensor`（MemRead）、`result: memdesc`（MemWrite）、可选 `mask`/`other`；属性 `cache`/`evict`/`isVolatile`；返回 `TTG_AsyncToken`。有效加载字节数 4/8/16 |
| `ttg.async_wait` | 等待异步拷贝完成，属性 `num`（等待的组数），返回新 token |
| `ttg.async_commit_group` | 把当前未提交的异步操作标记为一组，返回 token |

```mlir
%token = ttg.async_copy_global_to_local %src, %dst mask %mask other %other cacheModifier = #tt<cache ca> : tensor<128x!tt.ptr<f32>> -> !ttg.memdesc<128xf32, #shared>
%token2 = ttg.async_wait %token1 {num = 0 : i32}
```

## 共享内存操作

| 操作 | 语义 |
|------|------|
| `ttg.local_alloc` | 分配共享内存缓冲区，返回 memdesc；可选初始化值与 `alignment`。辅助方法 `isSharedMemoryAlloc()`、`getAlignmentOrDefault()` |
| `ttg.local_dealloc` | 显式释放（可选；未显式释放的缓冲区在最后使用后的第一个后支配点隐式释放） |
| `ttg.local_load` | 共享内存 → 分布式张量，可挂 async token（`LocalLoadTrait`） |
| `ttg.local_store` | 分布式张量 → 共享内存 |

```mlir
%buf = ttg.local_alloc %src : () -> !ttg.memdesc<128xf32, #shared>
%result = ttg.local_load %buf token %async_token : !ttg.memdesc<128xf32, #shared> -> tensor<128xf32, #blocked>
ttg.local_store %tensor, %buf : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared>
```

## 内存描述符视图操作

均为 `Pure` + `MemDescViewTrait`，只改视图不动底层内存：

| 操作 | 语义 | 示例 |
|------|------|------|
| `ttg.memdesc_index` | 取第 0 维第 i 个子视图 | `2x4x16xf16` index=1 → `4x16xf16` |
| `ttg.memdesc_subslice` | 按各维 `offsets` 取子视图 | `32x16` offsets=[2,1] → `8x16` 覆盖 `[2:10, 1:17]` |
| `ttg.memdesc_trans` | 转置视图（`order` 属性） | `8x16` → `16x8` |
| `ttg.memdesc_reshape` | 改形状视图 | `128xf16` → `8x16xf16` |
| `ttg.memdesc_reinterpret` | 重解释类型与形状（要求原描述符连续） | `128xf16` → `64xf32` |

## 软件流水线操作

- `ttg.predicate_stage %iv, %ub, %step maxStage N stage S : index -> i1`：流水线阶段谓词。
- `ttg.mask`：带区域的流水线掩码操作，区域内以 `ttg.mask.return` 终止。

## 类型转换

`ttg.fp4_to_fp %src {axis}`：把打包为 i8 的 FP4（E2M1）上转换为浮点张量——每个 i8 低 4 位为第一个 FP4 元素、高 4 位为第二个。

```mlir
%result = ttg.fp4_to_fp %src {axis = 1 : i32} : tensor<16x8xi8> -> tensor<16x16xf32>
```

## 全局内存

`ttg.global_scratch_alloc {nbytes, alignment} : !tt.ptr<i8>`：分配当前程序私有的全局内存缓冲区（MemAlloc\<GlobalMemory\>）。

## Warp 特化

`ttg.warp_specialize` 在不同 warp 组上异步执行不同代码：默认区域可隐式捕获外层值，分区区域与上方隔离（`IsolatedFromAbove`）。属性 `partitionNumWarps`、可选 `warpGroupStartIds` / `requestedRegisters` / `actualRegisters`。

```mlir
%0 = ttg.warp_specialize(%a, %b)
  default {
    %out = some_operation(%a)
    ttg.warp_yield %out : i32
  }
  partition0(%arg0: i32, %arg1: i32) num_warps(8) {
    some_async_dispatch(%arg0, %arg1)
    ttg.warp_return
  }
: (i32, i32) -> i32
```

| 操作 | 角色 |
|------|------|
| `ttg.warp_specialize.partitions` | 隔离分区区域的容器（Terminator，HasParent WarpSpecializeOp） |
| `ttg.warp_yield` | 默认区域终止符，操作数作为 warp_specialize 的 SSA 结果 |
| `ttg.warp_return` | 分区区域隐式终止符，无操作数（分区不能返回值） |

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)
