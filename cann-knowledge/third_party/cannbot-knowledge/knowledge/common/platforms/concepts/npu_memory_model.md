---
type: Concept
title: NPU 内存层次与数据通路模型
description: Ascend NPU 的 GM/L1/UB/L0A/L0B/L0C 存储层次、容量与对齐约束、MTE/FIX 数据通路及 tl.load/tl.store/tl.dot
  的硬件映射。
tags: [task.lookup, topic.architecture, topic.implementation.memory_management, topic.data_movement]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/cv_autotune/hardware_consts.py
  role: implementation
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.td
  role: constraint
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/technical_appendix/concepts_and_terms/glossary.md
  role: constraint
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-12T03:07:12Z'
platforms: [a2, a3, '950']
---
# NPU 内存层次与数据通路模型

昇腾 NPU 存在两条核心数据通路：Cube 路径 `GM -> L1 -> L0A/L0B -> L0C -> (FIX) -> GM/L1/UB` 与 Vector 路径 `GM -> UB -> GM`。`tl.load`/`tl.store` 映射为 GM↔UB 的 DMA 搬运，`tl.dot` 走完整 Cube 通路。片上存储由编译器自动管理（与 GPU 手动管理 Shared Memory 不同）。

## 存储层次与容量（显式标注平台）

| 存储空间 | IR 标识符 | 910B/93 | 910_95 | 对齐 | 用途 |
|----------|-----------|---------|--------|------|------|
| GM | `gm` | 数十 GB (HBM) | 数十 GB (HBM) | - | 外部全局存储 |
| L1 | `cbuf` | 512KB | 512KB | 32B | Cube 输入缓存 |
| UB | `ub` | 192KB | 256KB（编译器可用 248KB，预留 8KB） | 32B | Vector 工作区 |
| L0A | `ca` | 64KB | 64KB | 512B | 矩阵 A 输入缓存 |
| L0B | `cb` | 64KB | 64KB | 512B | 矩阵 B 输入缓存 |
| L0C | `cc` | 128KB | 256KB | 512B | 矩阵乘结果缓存 |
| BT Buffer | - | 1KB | 1KB | 64B | Bias 数据缓存（L1→BT 经 MTE1） |
| FP Buffer | - | 7KB | 7KB | 128B | FixPipe 中间缓冲区 |

## 对齐约束

| 算子类型 | 尾轴对齐要求 |
|----------|-------------|
| VV 类（纯 Vector） | 尾轴大小能被 32B 整除 |
| CV 类（Cube+Vector） | 尾轴大小能被 512B 整除 |

尾轴不足时编译器自动补齐，导致性能恶化（如 shape (2048, 3) 尾轴 3 补齐到 32B）。可用「借轴转置」技巧让长短轴都对齐。

## 流水线与数据通路

| 流水线 | 枚举值 | 数据流 |
|--------|--------|--------|
| PIPE_S | 0 | 标量流水线 |
| PIPE_V | 1 | UB -> UB 向量计算 |
| PIPE_M | 2 | L0A/L0B -> L0C 矩阵计算 |
| PIPE_MTE1 | 3 | L1 -> L0A/L0B/BT（单向） |
| PIPE_MTE2 | 4 | GM -> L1/UB（搬入；其他目的缓冲区按平台核对） |
| PIPE_MTE3 | 5 | UB -> GM（单向；910_95 另支持 UB -> L1） |
| PIPE_FIX | 10 | L0C -> GM/L1/UB（L0C->UB 仅 910_95） |

Cube 路径（`tl.dot`）：

```text
GM --[MTE2]--> L1 --[MTE1]--> L0A/L0B --[M]--> L0C --[FIX]--> GM/L1/UB
```

Vector 路径（`tl.load`/`tl.store` + 逐元素计算）：

```text
GM --[MTE2]--> UB --[V]--> UB --[MTE3]--> GM
```

CV 混合路径：**910_95 支持 L0C -> UB 直通**（`hivm.fixpipe` 输出直接进 UB 供 Vector 使用）；非 910_95 必须经 GM 中转（`L0C -> GM -> UB`），多一次 GM 往返，是 CV 融合算子的关键平台差异。

## Triton API 的硬件映射

| Triton API | 硬件映射 |
|------------|---------|
| `tl.load` | MTE2：GM -> UB（`hivm.load`，支持 padding/隐式转置） |
| `tl.store` | MTE3：UB -> GM（`hivm.store`，支持原子操作） |
| `tl.dot` | MTE2(GM->L1) + MTE1(L1->L0A/L0B) + M(Cube) + FIX(L0C->GM/L1/UB) |

## MultiBuffer 与存算并行

| 配置 | UB 可用容量 | 说明 |
|------|------------|------|
| multibuffer=True（A2/A3 默认） | 910B：96KB；910_95：128KB | 双缓冲，存算并行 |
| multibuffer=False（910_95 默认） | 910B：192KB；910_95：248KB | 单缓冲，存算串行 |

开启后 UB 可用容量减半。数据量小、UB 紧张时可关闭换取空间，建议 autotune 对比：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 2048, 'multibuffer': False}),
    ],
    key=['N'],
)
```

## NPU 适配要点

1. **UB 容量有限**：910B 192KB（双缓冲后 96KB）；910_95 256KB（双缓冲后 128KB）。控制单次搬运量。
2. **UB OVERFLOW 处理**：典型报错 `ub overflow, requires 3072256 bits while 1572864 bits available!`，解法是减小 `BLOCK_SIZE` 或引入 `BLOCK_SIZE_SUB` 子块分块。
3. **Cube/Vector 存储分离**：Cube 用 L1/L0A/L0B/L0C，Vector 用 UB；CV 融合需 FixPipe（910_95）或 GM 中转。
4. **910_95 默认关 MultiBuffer**，需显式 `multibuffer=True` 才开启存算并行。
5. **显式内存管理**（Buffer 模型扩展，定义于 `triton.backends.ascend.language.cann.extension.core`）：
   - `ascend_address_space`：`GM/L1/UB/L0A/L0B/L0C` 地址空间常量
   - `copy(src, dst)`：UB -> UB 或 UB -> L1（910_95）
   - `fixpipe(src, dst, dma_mode=FixpipeDMAMode.NZ2ND, dual_dst_mode=...)`：L0C -> UB（仅 910_95）
   - `sync_block_set/wait/all`：Cube-Vector 同步（event_id 0~15）
   - `sub_vec_id()` / `sub_vec_num()`：每个 AI Core 含 2 个 Vector Core


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py](https://github.com/triton-lang/triton-ascend/tree/b4db1e9e45c14f708fa5cbd05d50efa49c023a94/third_party/ascend/language/cann/extension/core.py)
- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/b4db1e9e45c14f708fa5cbd05d50efa49c023a94/docs/zh/programming_guide.md)
- [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
