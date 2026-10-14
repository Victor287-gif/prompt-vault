---
type: API
title: HFusion → HIVM 转换（ConvertHFusionToHIVM 与 mm-map-mode）
description: ConvertHFusionToHIVM（convert-hfusion-to-hivm）的语义与 mm-map-mode（core_op
  / macro_instr）矩阵乘映射策略，ND→Fractal 布局转换、同步插入规则，及 ConvertHFusionToVector / ConvertTensorToHIVM
  辅助路径。
tags: [task.lookup, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion → HIVM 转换

HFusion→HIVM 是 BiShengIR 流水线的关键一跳：把融合后的高层算子图（命名算子 + linalg.generic + 调度/tiling 信息）转换为硬件指令映射图，决定算子落到 Cube / Vector / DMA 哪个执行单元。HIVM 方言总览见 [overview](../hivm/overview.md)。

## ConvertHFusionToHIVM（`convert-hfusion-to-hivm`，ModuleOp）

- 构造函数 `mlir::createConvertHFusionToHIVMPass()`。
- 依赖方言：func、hivm、memref、arith、affine、scf、vector、linalg。
- 唯一选项 `mm-map-mode`（默认 `core_op`），控制矩阵乘法如何映射：

| 模式 | 说明 | 适用 |
|------|------|------|
| `core_op` | 映射为 HIVM Core 操作（Cube 单元），数据布局转 Fractal（Z 形/N 形） | 大规模矩阵乘 |
| `macro_instr` | 映射为宏指令（如 `hivm.hir.matmul`，见 [macro_matmul](../hivm/macro_matmul.md)），抽象更高，允许编译器更激进优化 | 需要灵活调度的场景 |

## 算子映射规则

| HFusion 算子 | HIVM 侧映射 | 执行单元 |
|-------------|-------------|----------|
| 矩阵乘法（core_op / macro_instr） | Cube Core 操作 / 宏指令（`hir.matmul`、`hir.mmadL1` 族，见 [macro_overview](../hivm/macro_overview.md)） | Cube |
| 逐元素运算 | 向量操作（`hir.vadd` 等，见 [vector_overview](../hivm/vector_overview.md)） | Vector |
| 归约运算 | 向量归约（`hir.vreduce` 族） | Vector |
| 数据搬运 | DMA 拷贝（`hir.copy` / `hir.load` / `hir.store`，见 [dma_copy](../hivm/dma_copy.md)） | DMA |
| 广播 / 转置 | 向量广播 / 转置拷贝 | Vector / DMA |

## 内存布局转换

转换把 ND 布局改为硬件特定格式（布局概念见 [data_layout](../architecture/data_layout.md)）：

| 场景 | 输入 | 输出 |
|------|------|------|
| Cube A 矩阵 | ND (M×K) | Fractal (zN) |
| Cube B 矩阵 | ND (K×N) | Fractal (nZ) |
| Cube C 矩阵 | Fractal (zN) | ND (M×N) |
| Vector 运算 | ND | ND（保持） |

## 同步插入规则

转换过程自动插入同步（机制详解见 [sync_sync_injection](../hivm/sync_sync_injection.md)）：

1. **Cube→Vector 依赖**：插同步确保 Cube 计算完成后再被 Vector 消费。
2. **DMA→计算依赖**：插同步确保数据搬运完成后再启动计算。
3. **多缓冲切换**：插同步确保前一个缓冲实例使用完毕（多缓冲机制见 [multi_buffer](../memory_management/multi_buffer.md)）。

## 辅助转换 Pass

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| ConvertHFusionToVector | `convert-hfusion-to-vector` | HFusion → vector 方言（寄存器模式），SIMT 路径用，跳过 HIVM 的 Cube 映射；依赖 vector、linalg、memref、arith、scf |
| ConvertTensorToHIVM | `convert-tensor-to-hivm` | tensor 直接 → HIVM，跳过 HFusion 层 |

## 转换后的典型结构

```
%buf_a = memref.alloc ...                    ; Cube 输入缓冲
%dma_a = hivm.hir.load/copy ... -> %buf_a    ; DMA 搬运
hivm.hir.set_flag / wait_flag ...            ; 等 DMA 完成
hivm.hir.matmul / mmadL1 ...                 ; Cube 矩阵乘
hivm.hir.set_flag / wait_flag ...            ; 等 Cube 完成
hivm.hir.vadd ...                            ; Vector 逐元素
hivm.hir.store ...                           ; DMA 写回
```

多缓冲场景下同一逻辑缓冲展开为多个物理实例，DMA 搬运下一片与当前片计算重叠，由 set_flag/wait_flag 维持依赖。

## 与后续阶段的衔接

`ConvertHFusionToHIVM` 完成后进入 HIVM 变换阶段：类型/内存作用域/布局推断、内存规划、同步注入（见 [hivm_transforms.md](hivm_transforms.md)），再经 SIMD / SIMT / MIX 三条路径之一 lowering 到后端（见 [hivm_to_backend.md](hivm_to_backend.md)）。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Conversion/Passes.td)
