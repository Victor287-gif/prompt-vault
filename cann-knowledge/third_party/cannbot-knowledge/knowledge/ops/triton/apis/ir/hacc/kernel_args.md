---
type: API
title: HACC Kernel 参数类型（hacc.arg_type 与参数 marshaling 规则）
description: KernelArgType 12 类参数语义（input/output/workspace/sync_block_lock/tiling_key/tiling_data/tiling_struct
  等）、out-param 输出传递约定、workspace 与同步锁的 Host 端推断机制及 tiling 参数打包。
tags: [task.lookup, topic.compilation, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HACC Kernel 参数类型

`hacc.arg_type` 附加在 Device 函数的每个参数上，声明该参数在 Kernel 启动时的语义角色。运行时按此 marshaling 参数，Host 端辅助函数据此计算衍生值（workspace 大小、锁数量等）。

## KernelArgType 完整列表

| 枚举值 | 整数 | 助记符 | 说明 |
|--------|------|--------|------|
| `kFFTSBaseAddr` | 0 | `ffts_base_address` | FFTS（Fast Function Task Schedule）基地址，Host 预配置执行参数以降低启动延迟 |
| `kInput` | 1 | `input` | 只读输入数据 |
| `kOutput` | 2 | `output` | 只写输出数据 |
| `kInputAndOutput` | 3 | `input_and_output` | 读写数据（原地操作） |
| `kWorkspace` | 4 | `workspace` | 工作空间内存（临时存储） |
| `kSyncBlockLock` | 5 | `sync_block_lock` | 多 Block 原子同步锁 |
| `kTilingKey` | 6 | `tiling_key` | Tiling 策略选择键 |
| `kTilingData` | 7 | `tiling_data` | 单个动态 Tiling 参数（tile 大小、步长等） |
| `kTilingStruct` | 8 | `tiling_struct` | 打包后的 Tiling 结构体 |
| `kMeshArg` | 9 | `mesh_arg` | Mesh 参数（多卡通信，配合 HMAP 集合通信） |
| `kSanitizerAddr` | 10 | `sanitizer_addr` | Sanitizer 地址（Device 端内存调试/越界检测） |
| `kGMAddr` | 11 | `gm_addr` | 直接传递的全局内存地址 |

## Marshaling 规则要点

1. **out-param 输出约定**：NPU Kernel 的输出不作为返回值，而是作为输入参数传入（`kOutput`），`hacc.output_idx` 标记哪个参数对应哪个输出值（见 [function_management.md](function_management.md)）。
2. **workspace**：大小由 Host 端 `hacc.infer_workspace_shape_function` 计算；生命周期管理见 [workspace_management](../memory_management/workspace_management.md)。
3. **sync_block_lock**：每个原子操作需要所有 Block 共享一个 `<1xi64>` memref 作锁；锁数量由 `hacc.infer_sync_block_lock_num_function` 推断，初始值由 `hacc.infer_sync_block_lock_init_function` 给出（绑定机制见 [host_device_binding.md](host_device_binding.md)）。
4. **tiling 三件套**：`tiling_key` 由 Host 按输入形状计算用于选策略；`tiling_data` 为散装的动态参数；`hacc-pack-tiling-data` Pass 可把多个 `tiling_data` 打包为单个 `tiling_struct`（结构体大小由 `get_tiling_struct_size_function` 给出，单位 i64）。

## MLIR 示例

```mlir
func.func @device_kernel(
  %input: memref<?x?xf16> {hacc.arg_type = #hacc.arg_type<input>},
  %output: memref<?x?xf16> {hacc.arg_type = #hacc.arg_type<output>},
  %workspace: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
  %lock: memref<1xi64> {hacc.arg_type = #hacc.arg_type<sync_block_lock>},
  %tiling_key: i64 {hacc.arg_type = #hacc.arg_type<tiling_key>},
  %tiling_data: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}
) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  return
}
```

C++ 侧用接口判定参数角色：

```cpp
if (funcOp.isKernelArg(argIdx, hacc::KernelArgType::kInput)) { ... }
```

相关：[host_device_binding.md](host_device_binding.md)、[function_management.md](function_management.md)、[sync_block_sync](../hivm/sync_block_sync.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HACCAttrs.td#L77-L104](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L77-L104)
