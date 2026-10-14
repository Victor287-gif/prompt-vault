---
type: API
title: Pipeline 选项参考（HIVMPipelineOptions / HFusionPipelineOptions 全字段）
description: HIVMPipelineOptions、ConvertToHIVMPipelineOptions、HFusionPipelineOptions
  的全部字段、默认值与典型配置组合（纯 SIMD / 纯 SIMT / MIX / 调试），及 buildHFusionPipelines / buildLowerHIVMPipelines
  等 pipeline 入口函数签名。
tags: [task.lookup, topic.compilation, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]

---

# Pipeline 选项参考

Pipeline 选项是各变换 Pass 的总开关：下表字段如何映射到具体 Pass 见 [hivm_transforms.md](hivm_transforms.md) 与 [hfusion_transforms.md](hfusion_transforms.md)，条件启用关系见 [pass_dependencies.md](pass_dependencies.md)。

## HIVMPipelineOptions

### 特性与优化控制

| 选项 | 默认 | 说明 |
|------|------|------|
| `enable-triton-kernel-compile` | false | 启用 Triton kernel 编译 |
| `enable-dot-scaled-compile` | false | 启用 dot_scaled 编译 |
| `enable-mixed-cv` | false | 启用混合 Cube-Vector 编译（MIX 路径） |
| `enable-layout-optimization` | false | 启用布局优化 |
| `pure-simt` | false | 纯 SIMT 编译（不用 Cube） |
| `enable-auto-multi-buffer` | false | 自动多缓冲（机制见 [multi_buffer](../memory_management/multi_buffer.md)） |
| `limit-auto-multi-buffer-only-for-local-buffer` | false | 自动多缓冲仅对 local buffer 生效 |
| `limit-auto-multi-buffer-of-local-buffer` | `no-l0c` | local buffer 多缓冲策略：`no-limit`/`no-l0c` |
| `limit-mix-auto-multi-buffer-buffer` | `only-cube` | Mix 多缓冲策略：`no-limit`/`only-cube`/`only-vector` |
| `set-workspace-multibuffer` | 2 | workspace 多缓冲数量 |
| `enable-auto-bind-sub-block` | true | 自动绑定子块 |
| `enable-auto-cv-balance` | false | CV 流水线化自动平衡 |
| `enable-auto-storage-align` | true | 自动存储对齐标记/启用 |
| `enable-global-workspace-reuse` | false | 全局 workspace 重用 |
| `enable-print-memory-allocated-size` | false | 打印内存分配大小 |
| `disable-tightly-coupled-buffer-reuse` | false | 禁用紧耦合缓冲重用 |

### 同步控制

| 选项 | 默认 | 说明 |
|------|------|------|
| `enable-hivm-inject-barrier-all-sync` | false | barrier-all 模式同步注入（仅调试，见 [sync_sync_injection](../hivm/sync_sync_injection.md)） |
| `enable-hivm-inject-block-all-sync` | false | 全块同步注入 |
| `disable-auto-inject-block-sync` | false | 禁用自动 sync block wait/set |
| `enable-hivm-graph-sync-solver` | false | 启用图同步求解器 |
| `enable-hivm-unit-flag-sync` | false | unit-flag 同步模式 |

### 编译策略与调优

| 选项 | 默认 | 说明 |
|------|------|------|
| `target` | `Ascend910B1` | 目标设备名称 |
| `enable-code-motion` | true | 代码移动/子集提升 |
| `enable-vf-merge-level` | 1 | VF 合并级别（0=不合并，1=无依赖合并，2=全部合并） |
| `enable-direct-hivm-lowering` | false | 直接 HIVM lowering |
| `enable-fused-multiply-add` | false | 融合乘加 |
| `enable-hivm-nd2nz-on-vector` | false | ND→NZ 在 Vector 上执行 |
| `enable-auto-blockify-loop` | false | 逻辑块数 > 物理块数时自动循环 |
| `enable-bishengir-simt-optimization` | 900101 | BishengIR SIMT 优化级别 |
| `use-dpx` | true | SIMT 经 DPX 方言 lowering |
| `disable-decompose-reduction` | 0 | 禁用归约分解 |
| `disable-reorder-instruction` | 0 | 禁用指令重排 |
| `enable-simt-reorder-instruction` | 0 | SIMT 指令重排模式 |
| `simt-vf-dynamic-size` | 216 | SIMT VF 动态 UB 大小（KB） |
| `max-reduction-split` | 1 | 归约循环最大分裂次数 |
| `inject-ir-from-file` | `""` | 注入 IR 文件路径（调试用） |

### MultiBufferStrategy 枚举

`NO_LIMIT`（无限制）、`CUBE_NO_L0C`（禁止 L0C 多缓冲）、`ONLY_CUBE`（仅 Cube）、`ONLY_VECTOR`（仅 Vector）。

## ConvertToHIVMPipelineOptions

`enable-triton-kernel-compile`（false）、`enable-auto-blockify-loop`（false，逻辑块 > 物理块自动循环）、`enable-regbase-hivmpipe`（false，基于寄存器的 HIVM pipeline）。

## HFusionPipelineOptions

### 特性与优化控制

| 选项 | 默认 | 说明 |
|------|------|------|
| `enable-manage-host-resources` | false | Host 函数资源管理 |
| `enable-triton-kernel-compile` | false | Triton kernel 编译 |
| `disable-ffts` | false | 强制禁用 FFTS |
| `disable-hfusion-vectorize` | false | 禁用 HFusion 自动向量化 |
| `multi-kernel` | false | 多 kernel 融合 |
| `enable-symbol-analysis` | false | 符号分析 |
| `enable-auto-vectorize-v2` | true | 第二代自动向量化器 |
| `enable-vf-fusion` | false | VF 融合 |
| `enable-tree-reduce` | false | 树形归约 |
| `enable-multiple-consumer-fusion` | false | AutoVectorizeV2 多消费者融合 |
| `skip-scope` | true | 存在 scope 时跳过 flattenOps 等 |
| `target` | `Ascend910B1` | 目标设备 |
| `enable-high-precision` | true | sin/cos 高精度计算 |
| `enable-layout-optimization` | false | 布局优化 |
| `enable-mixed-cv` | false | 混合 Cube-Vector 编译 |
| `enable-fuse-reduction-into-loop` | false | 后循环归约融合进循环体 |
| `enable-auto-multi-buffer` | false | 自动多缓冲 |
| `enable-drop-unit-dims` | true | 移除单位维度 |
| `enable-flatten` | true | 展平 Pass |
| `enable-auto-bind-sub-block` | true | 自动绑定子块 |
| `enable-deterministic-computing` | true | 确定性计算（禁用可提性能但结果不确定） |
| `enable-ops-reorder` | true | 操作重排序 |

### 调优

| 选项 | 默认 | 说明 |
|------|------|------|
| `max-horizontal-fusion-size` | -1 | 最大水平融合大小 |
| `max-fused-elementwise-ops` | -1 | PreVectorizationFusion 最大逐元素融合数 |
| `max-fused-ops-in-auto-vectorize-v2` | -1 | AutoVectorizeV2 最大融合算子数 |
| `max-buffer-count-tuning` | 0 | 调优最大缓冲计数 |
| `cube-tiling-tuning` | `[]` | Cube tiling 参数调优 |
| `enable-count-buffer-dma-opt` | false | DMA 缓冲不被 Vector 重用 |
| `external-tiling-func-path` | `"-"` | 外部 tiling 函数路径 |
| `inject-ir-from-file` | `""` | 注入 IR 文件（调试） |
| `block-dim` | 1 | 块维度大小 |

内部状态 `insertFFTS`（bool，默认 true，非 CLI 选项）控制是否插入 FFTS。

## Pipeline 入口函数

```cpp
// HFusion
void buildHFusionPipelines(OpPassManager &pm, const HFusionPipelineOptions &options);
void buildHFusionRegBasePipeline(OpPassManager &pm, const HFusionPipelineOptions &options);
void registerLowerHFusionPipelines();
bool enableSIMDVFFusion(const HFusionPipelineOptions &options);
// HIVM
void buildConvertToHIVMPipeline(OpPassManager &pm, const ConvertToHIVMPipelineOptions &options);
void buildHIVMTensorOptimizations(OpPassManager &pm, const HIVMPipelineOptions &options);
void buildLowerHIVMPipelines(OpPassManager &pm, const HIVMPipelineOptions &options);
void registerConvertToHIVMPipelines();
void registerLowerHIVMPipelines();
void canonicalizationHIVMPipeline(OpPassManager &pm);
```

## 典型配置组合

| 场景 | 关键开关 |
|------|----------|
| 纯 SIMD（Cube） | `enable-triton-kernel-compile=true pure-simt=false enable-mixed-cv=false target=Ascend910B1` |
| 纯 SIMT（Vector） | `pure-simt=true enable-mixed-cv=false use-dpx=true target=Ascend910B1` |
| MIX（Cube+Vector） | `enable-mixed-cv=true enable-auto-multi-buffer=true enable-auto-cv-balance=true target=Ascend910B1` |
| 调试 | `enable-hivm-inject-barrier-all-sync=true enable-print-memory-allocated-size=true inject-ir-from-file=<path>` |

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/Pipelines/Passes.h)
- [Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HFusion/Pipelines/Passes.h)
