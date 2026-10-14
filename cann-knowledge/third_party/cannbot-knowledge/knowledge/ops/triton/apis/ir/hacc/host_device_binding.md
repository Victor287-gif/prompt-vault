---
type: API
title: HACC Host-Device 函数绑定（HostFuncType 与 FuncRefAttr 系列）
description: HostFuncType 8 类 Host 辅助函数角色（tiling_function / infer_output_shape / infer_workspace_shape
  / sync_block_lock 推断等）与 FuncRefAttr 符号绑定属性，及完整 Host-Device 绑定 IR 形态。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HACC Host-Device 函数绑定

Device Kernel 执行前，Host 端需先跑一批辅助函数（算 Tiling、推输出形状、推 workspace 大小等）。HACC 用 `HostFuncType` 标记 Host 函数角色，用一组 `FuncRefAttr` 属性把这些 Host 函数符号绑定到 Device 函数上。

## HostFuncType 完整列表

| 枚举值 | 整数 | 助记符 | 绑定属性（挂在 Device 函数上） |
|--------|------|--------|--------------------------------|
| `kEntry` | 1 | `host_entry` | —（Host 入口，自身角色标记） |
| `kTilingFunction` | 2 | `tiling_function` | `hacc.tiling_function` |
| `kInferOutputShapeFunction` | 3 | `infer_output_shape_function` | `hacc.infer_output_shape_function` |
| `kInferWorkspaceShapeFunction` | 4 | `infer_workspace_shape_function` | `hacc.infer_workspace_shape_function` |
| `kInferSyncBlockLockNumFunction` | 5 | `infer_sync_block_lock_num_function` | `hacc.infer_sync_block_lock_num_function` |
| `kInferSyncBlockLockInitFunction` | 6 | `infer_sync_block_lock_init_function` | `hacc.infer_sync_block_lock_init_function` |
| `kInferVFModeFunction` | 7 | `infer_vf_mode_function` | `hacc.infer_vf_mode_function` |
| `kGetTilingStructSizeFunction` | 8 | `get_tiling_struct_size_function` | `hacc.get_tiling_struct_size_function` |

Host 函数自身用 `hacc.host_func_type = #hacc.host_func_type<tiling_function>` 声明角色；绑定属性均为 `FuncRefAttr` 基类（参数为 `FlatSymbolRefAttr:$funcName`），语法 `#hacc.tiling_function<@tiling_func>`。

## 各绑定函数职责

| 函数 | 职责 |
|------|------|
| TilingFunction | 按输入形状与设备规格计算 Tiling 参数（分块大小、迭代次数），结果作为 `tiling_data` 参数传给 Device Kernel |
| InferOutputShapeFunction | 推断动态形状 Kernel 的输出张量形状，供 Host 预分配输出内存 |
| InferWorkspaceShapeFunction | 推断 workspace 大小（workspace 参数语义见 [kernel_args.md](kernel_args.md)，管理策略见 [workspace_management](../memory_management/workspace_management.md)） |
| InferSyncBlockLockNumFunction | 推断同步锁数量（每个原子操作需一个全 Block 共享的 `<1xi64>` memref 锁） |
| InferSyncBlockLockInitFunction | 推断各锁初始值，Kernel 执行前完成初始化 |
| InferVFModeFunction | 推断 VF（Vector Function）模式，控制向量化执行策略 |
| GetTilingStructSizeFunction | 返回 Tiling 结构体大小（单位 i64），供 `hacc-pack-tiling-data` 打包散装 tiling_data |

调用关系：`Host: tiling_func(input_shape, device_spec) -> (tile_m, tile_n, ...)` → `Device: kernel(input, output, tile_m, tile_n, ...)`。

## 完整绑定 IR 示例

```mlir
func.func @tiling_func(%arg0: tensor<?x?xf16>) -> (i64, i64)
  attributes {hacc.function_kind = #hacc.function_kind<HOST>,
              hacc.host_func_type = #hacc.host_func_type<tiling_function>} { ... }

func.func @infer_shape(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>,
              hacc.host_func_type = #hacc.host_func_type<infer_output_shape_function>} { ... }

func.func @device_kernel(
  %input: tensor<?x?xf16>, %output: tensor<?x?xf16>,
  %t1: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>},
  %t2: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
) attributes {
  hacc.function_kind = #hacc.function_kind<DEVICE>,
  hacc.tiling_function = #hacc.tiling_function<@tiling_func>,
  hacc.infer_output_shape_function = #hacc.infer_output_shape_function<@infer_shape>
} { ... }

// Host 入口：组织 Device Kernel 调用
func.func @host_entry(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16>
  attributes {hacc.function_kind = #hacc.function_kind<HOST>,
              hacc.host_func_type = #hacc.host_func_type<host_entry>} {
  %r = func.call @device_kernel(%arg0, ...) : (tensor<?x?xf16>, ...) -> tensor<?x?xf16>
  return %r : tensor<?x?xf16>
}
```

相关：[function_management.md](function_management.md)、[kernel_args.md](kernel_args.md)、[overview.md](overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HACCAttrs.td#L110-L225](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L110-L225)
