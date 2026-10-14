---
type: API
title: HMAP 方言（hmap.all_to_allv 集合通信）
description: HMAP（Hybrid Mesh Aware Parallelism）方言基于 MLIR Mesh 方言提供多卡集合通信操作，核心是 hmap.all_to_allv——支持不均匀分片的全互联向量通信，与
  HACC 的 mesh_arg 参数类型协作完成 Host 编排与 Device 数据交换。
tags: [task.lookup, topic.compilation, topic.data_movement.memory_access, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HMAP 方言

HMAP（Hybrid Mesh Aware Parallelism，混合 Mesh 感知并行）方言（方言名 `hmap`，C++ 命名空间 `::mlir::hmap`，依赖 `mesh::MeshDialect`）提供集合通信操作，用于多卡/多设备间数据交换，支持 Mesh 感知的并行策略。

## 操作基类

- `HMAP_Op`：所有 HMAP 操作的基类。
- `HMAP_CollectiveCommunicationOpBase`：集合通信操作基类，自动附加 `SymbolUserOpInterface` 与 `OpAsmOpInterface`。

| 公共参数 | 类型 | 说明 |
|----------|------|------|
| `mesh` | `FlatSymbolRefAttr` | Mesh 符号引用 |
| `mesh_axes` | `DenseI16ArrayAttr`（默认 `{}`） | 通信使用的 Mesh 轴 |

## hmap.all_to_allv

在设备间执行全互联向量（All-to-All Vector）通信：将张量分片发送到所有设备并接收来自所有设备的分片。`Pure`，输入/输出均为非 0 秩张量。

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `input` | `AnyNon0RankedTensor` | 输入张量 |
| `input_splits` | `AnyTensor` | 输入分片大小 |
| `output_splits` | `AnyTensor` | 输出分片大小 |
| `result` | `AnyNon0RankedTensor` | 输出张量 |

```mlir
%result = hmap.all_to_allv %input on @mesh
  input_splits = %in_splits : tensor<4xi32>
  output_splits = %out_splits : tensor<4xi32>
  : tensor<1024xf32> -> tensor<1024xf32>

// 带 mesh_axes
%result = hmap.all_to_allv %input on @mesh mesh_axes = [0, 1]
  input_splits = %in_splits : tensor<4xi32>
  output_splits = %out_splits : tensor<4xi32>
  : tensor<1024xf32> -> tensor<1024xf32>
```

## 与 MLIR Mesh 方言的关系

HMAP 在 MLIR Mesh 之上提供更高级抽象：`mesh` 参数引用 Mesh 方言定义的 Mesh 资源，`mesh_axes` 指定通信维度。

| HMAP | Mesh | 差异 |
|------|------|------|
| `hmap.all_to_allv` | `mesh.all_to_all` | HMAP 支持不均匀分片（v = variable），由 input_splits / output_splits 张量描述 |

## 与 HACC 方言的协作

HMAP 集合通信操作在 Host 端编排；Mesh 参数通过 `hacc.arg_type = #hacc.arg_type<mesh_arg>` 传递给 Device Kernel 用于多卡数据交换。HACC 参数类型机制见 [kernel_args](../hacc/kernel_args.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HMAPOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HMAP/IR/HMAPOps.td)
- [HMAPBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HMAP/IR/HMAPBase.td)
