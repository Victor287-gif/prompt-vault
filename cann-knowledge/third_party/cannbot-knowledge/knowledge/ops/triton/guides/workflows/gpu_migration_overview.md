---
type: Guide
title: GPU→NPU 迁移概览：架构差异与迁移工作流
description: GPU SIMT 与 Ascend NPU SIMD 架构差异速查（计算核心/内存层次/调度/数据搬运/数据类型），迁移检查清单、常见错误速查与三阶段迁移工作流。
tags: [task.migration, topic.quality.compatibility.version_migration, topic.parallelism, topic.data_movement.memory_access, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']
---
# GPU→NPU 迁移概览：架构差异与迁移工作流

触发条件：将 `device='cuda'` 上运行的 Triton kernel 适配到 Ascend NPU，或排查 GPU→NPU 迁移后的编译/运行错误。

## 1. 计算核心与执行模型

| 维度 | GPU (NVIDIA) | NPU (Ascend) |
|------|--------------|--------------|
| 计算核心 | SM | AI Core（Cube + Vector，1:2 配比） |
| 执行模型 | SIMT（单指令多线程） | SIMD（单指令多数据）；910_95 额外支持 SIMT 模式 |
| 矩阵单元 | Tensor Core | Cube Unit（均映射 `tl.dot()`，最小 16x16x16） |
| 向量单元 | CUDA Core | Vector Unit |
| 最小调度单位 | Warp（32 线程） | AI Core（一个 Block 绑定一个 AI Core） |
| 线程组织 | Thread→Warp→Block→Grid | 无线程概念（SIMD）；910_95 SIMT 模式下有线程概念 |

SIMT 下每个线程有独立上下文，Warp 锁步执行、分支发散串行化；SIMD 下一条指令处理整个向量，无分支发散但也不支持线程级独立执行。910_95 为 Reg-based 架构，可用 `compile_mode="simt_only"` / `"unstructured_in_simt"` 启用 SIMT 模式，适合控制流密集操作。

## 2. 内存层次

| 层级 | GPU | NPU (910B A2/A3) | NPU (910_95) |
|------|-----|------------------|--------------|
| 全局内存 | HBM 40-80GB | GM 32-64GB | GM 32-64GB |
| 片上共享内存 | Shared Memory 48-164KB/SM | UB 192KB/AI Core | UB 256KB/AI Core |
| 私有缓存 | L1 Cache（与 SMEM 共享） | L1 Buffer（与 UB 独立） | L1 Buffer（与 UB 独立） |
| 寄存器 | RF 256KB/SM | - | RF 128KB + DCache 32~120KB（SIMT 模式） |

关键差异：UB 由编译器自动管理、超限即编译报 `ub overflow`；Vector 算子访存需 32B 对齐、CV 融合算子需 512B 对齐；NPU 强偏好连续访存，离散访存会导致性能严重下降甚至 scalar 退化。

## 3. 并行调度

| 维度 | GPU | NPU |
|------|-----|-----|
| Grid 本质 | 逻辑任务维度，与物理核解耦 | 物理核组映射，绑定 AI Core 拓扑 |
| Grid 限制 | 无硬限制 | Grid ≤ AI Core 总数，coreDim ≤ 65535 |
| 超核调度 | 硬件自动，开销小 | 分批调度，设备侧开销大 |
| num_warps / num_stages | 有效 | 无效（无 Warp / 软件流水线概念） |
| Grid 维度 | 1D/2D/3D | 优先 1D，2D 会被合并为 1D |
| 核数取值 | 编译器/硬件决定 | 纯 Vector 算子 = Vector Core 数；含 tl.dot = AI Core 数 |

物理核数获取：`driver.active.utils.get_device_properties(0)["num_aicore"]`。

## 4. 数据搬运流水线

```text
GPU:  GM → Shared Memory → Register → 计算 → GM
NPU:  GM ──MTE2──> UB ──Vector/Cube──> UB ──MTE3──> GM
                   ↑                      |
                   └── multibuffer 双缓冲 ──┘
```

MTE2 搬入、MTE3 搬出、MTE1 负责 L1→UB；`multibuffer` 是 NPU 特有的存算并行优化（910B 默认开启，910_95 默认关闭），双缓冲时可用 UB 减半。

## 5. 数据类型支持

| 类型 | GPU | NPU (910B) | NPU (910_95) | 说明 |
|------|-----|-----------|--------------|------|
| fp16/bf16/fp32/int8/int16/int32 | 支持 | 支持 | 支持 | - |
| int64 | 支持 | 支持 | 支持 | Vector ADD/CMP 退化为 scalar |
| uint8 | 支持 | 部分不支持 | 部分不支持 | Block Pointer 场景不支持 |
| uint16/uint32/uint64 | 支持 | 不支持 | 不支持 | 硬件限制，换对应 int 类型 |
| fp64 | 支持 | 不支持 | 不支持 | 换 fp32 |
| fp8 | 支持 | 不支持 | 支持 | 910_95 支持 dot_scaled |

## 6. 迁移检查清单（必须修改的代码模式）

1. **设备标识**：`import torch_npu`；`device='cuda'`→`'npu'`；删除 `get_active_torch_device()` 与设备一致性断言。
2. **移除 GPU 专属参数**：autotune Config 中的 `num_warps`/`num_stages` 删除；NPU 特有参数 `multibuffer`。
3. **Grid 对齐物理核数**：`grid = (num_cores,)`，配合核内循环；或 `kernel[n, 1, 1](...)`。
4. **数据类型替换**：fp64→fp32；uint8→int8；uint16/32/64→int16/32/64（注意符号位与范围）。
5. **tl.load/tl.store 参数**：移除 `cache_modifier`/`eviction_policy`/`volatile`；可加 `care_padding=False` 提升并行度。
6. **核内 Tiling 防 UB 溢出**：核间 `BLOCK_SIZE` 控 coreDim，核内 `BLOCK_SIZE_SUB` 循环控 UB 用量。
7. **Block Pointer**：stride 必须反映真实内存布局；转置只能通过 `order` 表达；`tl.advance` 与复杂控制流搭配可能编译失败，改为循环内重建 `tl.make_block_ptr`；不支持 uint 系列/fp64。
8. **离散访存→连续访存**：一维 `(1024,), strides=(32,)` 改为二维 `(1024, 32), strides=(32, 1), order=(1, 0)`，保证最内轴连续。
9. **Vector CMP 类型转换**：`tl.where` 条件中 int64/int32 索引比较退化为 scalar，转 fp32 后再比较。

## 7. 910_95 特别注意

| 特性 | 910_95 | 910B |
|------|--------|------|
| UB 容量 | 256KB（可用约 248KB） | 192KB |
| L0C→UB 直通（FixPipe） | 支持 | 不支持 |
| MultiBuffer 默认 | 关闭 | 开启 |
| FP8 | 支持 dot_scaled | 不支持 |
| SIMT 模式 | 支持 | 不支持 |
| 同步方式 | SetFlag/WaitFlag | FFTS |

## 8. 常见迁移错误速查

| 错误 | 典型信息 | 根因 | 解决 |
|------|---------|------|------|
| UB 溢出 | `ub overflow, requires xxxx bits while ... available!` | 单核数据量超 UB | 核内 Tiling；减 BLOCK_SIZE_SUB；关 multibuffer |
| coreDim 超限 | `coreDim=xxxx can't be greater than UINT16_MAX` | Grid > 65535 | 增大 BLOCK_SIZE；`TRITON_ALL_BLOCKS_PARALLEL=1` |
| 复合问题 | 增大 BLOCK_SIZE 后 UB 溢出 | coreDim 与 UB 同时受限 | 核间大块 + 核内小块双重切分 |
| 精度差异 | 结果与 GPU/CPU 不一致 | 浮点顺序、类型退化 | `TRITON_INTERPRET=1` 取 CPU 基准；`tl.device_print` |
| Scalar 退化 | 性能大幅下降 | int64/int32 CMP | 比较操作数转 fp32 |
| Block Pointer 编译失败 | MLIR 编译错误 | advance + 复杂控制流 | 重建 make_block_ptr 或手动指针算术 |

调试环境变量：`TRITON_INTERPRET=1`（CPU 解释执行）、`MLIR_ENABLE_DUMP=1`（转储 IR）、`ENABLE_PRINT_UB_BITS=1`（打印 UB 占用）、`TRITON_ALL_BLOCKS_PARALLEL=1`（逻辑核自动对齐物理核，要求 kernel 对执行顺序不敏感）。

## 9. 迁移工作流三阶段

1. **先跑通**：设备替换 → 移除 GPU 专属参数 → 数据类型替换 → Grid 对齐核数 → `TRITON_INTERPRET=1` 验证正确性。
2. **再优化**：核内 Tiling → 访存连续化 → `multibuffer` / `care_padding=False` → CMP 类型转 fp32 → `msprof op` 定位瓶颈。
3. **910_95 专项**：评估 SIMT 模式；FP8 混合精度；L0C→UB 直通减少 Cube→Vector 搬运。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [architecture_difference.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/architecture_difference.md)
- [migrate_from_gpu.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/migration_guide/migrate_from_gpu.md)
