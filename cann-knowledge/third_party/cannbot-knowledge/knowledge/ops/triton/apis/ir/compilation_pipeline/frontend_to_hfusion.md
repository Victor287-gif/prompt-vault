---
type: API
title: 前端 → HFusion 转换（六个 Conversion Pass 与执行顺序）
description: arith/math/linalg/gpu/tensor/torch 六个前端方言到 HFusion 的转换 Pass（convert-*-to-hfusion）的
  CLI 参数、转换语义与推荐执行顺序，附 ConvertTorchToSymbol 与辅助转换。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# 前端 → HFusion 转换

前端→HFusion 转换把各前端方言操作统一映射为 HFusion 方言操作，为后续算子融合与调度做准备。HFusion 方言本身的操作分类见 [overview](../hfusion/overview.md)，转换完成后的变换阶段见 [hfusion_transforms.md](hfusion_transforms.md)。

## 六个主转换 Pass

### ConvertArithToHFusion（`convert-arith-to-hfusion`）

- 构造函数 `mlir::createArithToHFusionConversionPass()`；依赖 arith、linalg、hfusion。
- 语义：`arith.addf/addi/mulf/muli` 等逐元素算术 → hfusion 命名算子或 `linalg.generic`。

### ConvertMathToHFusion（`convert-math-to-hfusion`）

- 构造函数 `mlir::createMathToHFusionConversionPass()`；依赖 linalg、hfusion。
- 语义：`math.exp/log/sqrt/sin/cos` 等数学函数 → hfusion 命名算子。

### ConvertLinalgToHFusion（`convert-linalg-to-hfusion`）

最核心的前端转换 Pass。依赖 arith、linalg、hfusion、bufferization（部分 linalg 操作转换前需缓冲化）。

| 源操作 | 目标 |
|--------|------|
| `linalg.matmul` / `linalg.batch_matmul` | hfusion 矩阵乘 / 批量矩阵乘命名算子（见 [matmul_ops](../hfusion/matmul_ops.md)） |
| `linalg.generic`（逐元素） | hfusion 逐元素命名算子（见 [elementwise_ops](../hfusion/elementwise_ops.md)） |
| `linalg.generic`（归约） | hfusion 归约命名算子（见 [reduction_ops](../hfusion/reduction_ops.md)） |
| `linalg.fill` | hfusion 填充算子 |
| `linalg.conv_2d` | hfusion 卷积命名算子 |

### ConvertGPUToHFusion（`convert-gpu-to-hfusion`）

- 依赖 gpu、hfusion。
- 语义：`gpu.launch_func` → HFusion kernel 启动逻辑；GPU 并程相关操作 → HFusion 调度操作。

### ConvertTensorToHFusion（`convert-tensor-to-hfusion`）

- 依赖 tensor、hfusion、linalg。
- 语义：`tensor.extract_slice/insert_slice` → HFusion 切片/切片写入；`tensor.empty` → HFusion 空张量；`tensor.expand_shape/collapse_shape` → HFusion 形状变换。

### ConvertTorchToHFusion（`convert-torch-to-hfusion`）

- 依赖 linalg、math、func、tensor、arith、hfusion；条件编译宏 `BISHENGIR_ENABLE_TORCH_CONVERSIONS`。
- 选项 `ensure-no-implicit-broadcast`（bool，默认 false）：确保无隐式广播语义。
- 语义：ATen 算子 → linalg 或 hfusion 命名算子，支持动态形状和符号维度。

## ConvertTorchToSymbol（`convert-torch-to-symbol`）

Torch 前端配套 Pass：`torch.symbolic_int` → `symbol.symbolic_int`。依赖 func、tensor、symbol、arith，同样受 `BISHENGIR_ENABLE_TORCH_CONVERSIONS` 控制。Symbol 方言见 [symbol_dialect](../other_dialects/symbol_dialect.md)。

## 辅助转换 Pass

| Pass | CLI 参数 | 说明 |
|------|----------|------|
| ConvertArithToAffine | `convert-arith-to-affine` | arith → affine |
| ConvertTensorToHIVM | `convert-tensor-to-hivm` | tensor 直接 → HIVM，跳过 HFusion 层（见 [hfusion_to_hivm.md](hfusion_to_hivm.md)） |

## 推荐执行顺序

1. `ConvertTorchToSymbol`（Torch 前端时）
2. `ConvertTorchToHFusion`（Torch 前端时）
3. `ConvertLinalgToHFusion`
4. `ConvertArithToHFusion`
5. `ConvertMathToHFusion`
6. `ConvertTensorToHFusion`
7. `ConvertGPUToHFusion`

全部完成后进入 HFusion 变换阶段（融合、调度、向量化），随后由 `ConvertHFusionToHIVM` 进入硬件映射层（见 [hfusion_to_hivm.md](hfusion_to_hivm.md)）。各阶段在完整 pipeline 中的位置见 [pass_dependencies.md](pass_dependencies.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Conversion/Passes.td)
