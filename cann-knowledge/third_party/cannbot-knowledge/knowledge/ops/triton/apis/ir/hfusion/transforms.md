---
type: API
title: HFusion 变换 Pass 总览（融合 / 向量化 / 合法化 / Tiling 打包等 40+ Pass）
description: HFusion 方言 40+ 变换 Pass 分类清单（融合调度、向量化、类型合法化、Tiling 打包、形状参数变换、操作变换、缓存符号、Host
  与 Triton 适配）及 fuse-ops / auto-vectorize / flatten-ops 关键选项。
tags: [task.lookup, topic.compilation]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HFusion 变换 Pass 总览

40+ 个变换 Pass 覆盖操作融合、自动调度、自动向量化、类型合法化、Tiling 打包等编译阶段。定义于 `Dialect/HFusion/Transforms/Passes.td`。HACC 侧的设备规格附加与函数重命名见 [transforms](../hacc/transforms.md)。

## 融合与调度

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-fuse-ops` | ModuleOp | 张量操作融合，可融合操作 outline 为独立函数 |
| `hfusion-auto-schedule` | ModuleOp | 自动调度融合 Kernel，生成 Tiling 参数 |
| `hfusion-infer-func-fusion-kind` | FuncOp | 推断函数融合类型 |
| `hfusion-infer-out-shapes` | ModuleOp | 生成输出形状推断函数 |
| `hfusion-compose-multi-reduce` | FuncOp | 多归约组合为单操作 |
| `hfusion-decompose-multi` | FuncOp | 多操作分解为单操作 |
| `hfusion-pre-vectorization-fusion` | FuncOp | 向量化前的逐元融合 |
| `hfusion-eliminate-duplicate-funcs` | ModuleOp | 消除融合后重复函数 |

## 向量化

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-auto-vectorize` | ModuleOp | 自动向量化 Linalg 命名操作 |
| `hfusion-auto-vectorize-v2` | ModuleOp | 第二版：Tile + Fuse + Vectorize |
| `hfusion-generic-unroller` | FuncOp | 非可向量化 Linalg 操作展开 |
| `hfusion-pull-slice-into-vector-function` | ModuleOp | extract/insert_slice 拉入 VF 函数 |
| `outline-vector-function` | ModuleOp | Outline 向量函数 |
| `hfusion-vectorize-ops` | ModuleOp | 向量化 HFusion 操作 |
| `remove-mask-from-unaligned-reduction-loop` | FuncOp | 移除未对齐归约循环掩码 |

## 类型合法化

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-legalize-scalar` | FuncOp | 标量操作合法化为张量操作 |
| `hfusion-legalize-bf16` | FuncOp | BF16 → FP32 |
| `hfusion-legalize-fp8` | FuncOp | FP8 → FP32 |
| `hfusion-legalize-bool` | ModuleOp | int8/int1 布尔合法化 |
| `hfusion-downgrade-fp64` | FuncOp | FP64 常量降级 FP32 |
| `hfusion-normalize-ops` / `hfusion-normalize-slice-ops` | FuncOp | 规范化 HFusion / Slice 操作 |

## Tiling 与数据打包

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-pack-tiling-data` | ModuleOp | 动态 Tiling 信息打包为结构体（参数语义见 [kernel_args](../hacc/kernel_args.md)） |
| `hfusion-constantize-tiling-data` | ModuleOp | Tiling 与 Device 函数间传播常量 |
| `hfusion-add-ffts-addr` | ModuleOp | 添加 FFTS 基地址参数 |

## 形状与参数变换

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-tensor-results-to-out-params` | ModuleOp | 张量结果移至输出参数（out-param 约定） |
| `hfusion-hoist-tensor-empty` | ModuleOp | tensor.empty 提升为函数参数并合并 |
| `hfusion-simplify-vf-arg` / `hfusion-merge-vf` | ModuleOp | 简化 / 合并 VF 函数 |
| `hfusion-fold-unit-dims` | FuncOp | 折叠 Linalg 操作单位维度 |

## 操作变换

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-convert-generic-to-named` | FuncOp | linalg.generic → 命名操作 |
| `hfusion-flatten-ops` | FuncOp | 展平 Linalg / HFusion 操作 |
| `hfusion-inline-brc` | FuncOp | 内联广播类操作 |
| `hfusion-outline-single-op` | FuncOp | Outline 单 Linalg 操作为 Kernel |
| `hfusion-simplify-ops` / `hfusion-reorder-ops` | FuncOp | 简化 / BFS 重排操作 |
| `hfusion-generalize` | FuncOp | HFusion 操作 → linalg.generic |
| `hfusion-decompose` | FuncOp | 分解 AggregatedOpInterface 操作 |

## 缓存与符号

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-cache-io` / `hfusion-cache-io-for-return-arg` / `hfusion-recache-io` / `hfusion-remove-cache-io` | FuncOp | IO 缓存的添加 / 返回参数缓存 / 重缓存 / 移除 |
| `hfusion-fold-symbolic-dim` / `hfusion-unfold-symbolic-dim` | FuncOp | tensor.dim ↔ SymbolicDimOp 互换（操作见 [special_ops.md](special_ops.md)） |
| `hfusion-drop-symbols` | FuncOp | 丢弃 ranked tensor 符号 |

## Host 函数与 Triton 适配

| Pass | 作用域 | 功能 |
|------|--------|------|
| `hfusion-wrap-host-func` | ModuleOp | Host 辅助函数包装（绑定机制见 [host_device_binding](../hacc/host_device_binding.md)） |
| `adapt-triton-kernel` | ModuleOp | 适配 Triton 入口 Kernel |

## 关键 Pass 选项

### hfusion-fuse-ops

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `output-mode` | OutputMode | Multiple | Outline 函数输出模式 |
| `fusion-mode` | FusionKind | Unknown | 融合类型（取值见 [attributes_enums.md](attributes_enums.md)） |
| `always-inline` | bool | false | 始终内联 outline 函数 |
| `move-out-to-param` | bool | true | 张量输出移至参数 |
| `max-horizontal-fusion-size` | int | -1 | 最大水平融合大小 |
| `multi-kernel` | bool | false | 多 Kernel outline |
| `enable-symbol-analysis` | bool | false | 启用符号分析 |

### hfusion-auto-vectorize / auto-vectorize-v2

| 选项 | 默认 | 说明 |
|------|------|------|
| `vector-length` | 256 | 向量长度（字节） |
| `peel-loops` | false | 尝试剥离循环 |
| `max-vectorize-axes` | -1 | 最大向量化轴数 |
| `tree-reduce` | false | 树形归约 |
| `max-fused-ops`（v2） | 15 | 单融合节点最大操作数 |
| `enable-multiple-consumer-fusion`（v2） | false | 多消费者融合 |

### hfusion-flatten-ops

| 选项 | 默认 | 说明 |
|------|------|------|
| `flatten-mode` | Greedy | 展平模式（Greedy/Tidy） |
| `skip-host` | false | 跳过 Host 函数 |
| `multi-dynamic-shape` | true | 折叠多个动态形状 |
| `register-based` | false | 基于寄存器的展平 |
| `skip-scope` | true | 存在 scope 时跳过 |

相关：[overview.md](overview.md)、[transforms](../hacc/transforms.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td)
- [Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.h)
