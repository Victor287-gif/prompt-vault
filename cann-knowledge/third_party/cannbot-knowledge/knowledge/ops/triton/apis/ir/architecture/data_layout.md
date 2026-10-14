---
type: API
title: 数据布局详解（ND/nZ/zN/Fractal、DataLayoutAttr、nd2nz/nz2nd 转换）
description: Cube 单元要求的 Fractal 布局体系：DataLayout 枚举（ND/nZ/zN/DOTA_ND 等）、DataLayoutAttr
  参数（transpose/fractalSizes）、nd2nz/nz2nd/l12ub 布局转换操作与 fixpipe dma_mode，以及 910B 与
  950 的布局数据流差异。
tags: [task.lookup, topic.architecture, topic.data_movement.memory_access, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']

---

# 数据布局详解

Cube 单元对输入布局有硬性要求：矩阵 A/B 必须以 Fractal（分形）格式存于 L0A/L0B，结果也以 Fractal 存于 L0C；而用户数据在 GM 中是 ND（行优先）格式。因此 GM↔L1 搬运必须做 ND↔NZ 布局转换——这是 NPU 编程中核心且独特的概念。HIVM IR 用 `DataLayout` 枚举 + `DataLayoutAttr` 参数化属性描述布局，用 `nd2nz`/`nz2nd` 操作显式表达转换。布局错误会导致计算结果错误或硬件异常。

## DataLayout 枚举

| 值（数值） | IR 标识符 | 物理含义 |
|------------|-----------|----------|
| DOTA_ND(1) | `dotA_ND` | 矩阵 A 的 ND 布局标记（Cube 左操作数） |
| DOTB_ND(2) | `dotB_ND` | 矩阵 B 的 ND 布局标记（Cube 右操作数） |
| DOTC_ND(3) | `dotC_ND` | 矩阵 C 的 ND 布局标记（乘法结果） |
| nZ(4) | `nZ` | 列优先 Fractal：块内 N 维（列方向）低地址连续；矩阵 B 进 L0B 所需格式 |
| zN(5) | `zN` | 行优先 Fractal：块内 Z 维（行方向）低地址连续；矩阵 A 进 L0A 所需格式 |
| ND(6) | `ND` | 标准行优先，GM 中用户数据的自然格式（末维连续） |
| Fractal(7) | `Fractal` | 通用分形布局，由 `fractalSizes=[blockM, blockN]` 描述块大小 |

`isNDLayout()` 对 DOTA_ND/DOTB_ND/DOTC_ND/ND 均返回 true。

## DataLayoutAttr 参数化属性

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `data_layout` | DataLayout 枚举 | 是 | 布局类型 |
| `transpose` | BoolAttr | DOTA_ND/DOTB_ND 时必须提供 | 操作数是否转置 |
| `fractalSizes` | DenseI64ArrayAttr | 仅 Fractal 时 | 分形块 [行, 列] 大小，典型 16×16 |

```mlir
#hivm.data_layout<ND>
#hivm.data_layout<zN>
#hivm.data_layout<dotA_ND, transpose = true>
#hivm.data_layout<Fractal, fractalSizes = [16, 16]>
#hivm.data_layout<dotA_ND, transpose = true, fractalSizes = [16, 32]>
```

## 布局与使用场景 / 操作对应

| 布局 | 场景 | 存储位置 |
|------|------|----------|
| ND | GM 中用户数据 | GM |
| zN | Cube 矩阵 A | L1、L0A |
| nZ | Cube 矩阵 B | L1、L0B |
| Fractal | Cube 结果 C | L0C |
| DOTA/B/C_ND | mmadL1 布局接口标记 | — |

| 操作 | 输入 → 输出布局 | Pipeline |
|------|------------------|----------|
| `hivm.nd2nz` | ND（GM）→ zN/nZ（L1） | MTE2 |
| `hivm.nz2nd` | zN/nZ（L1）→ ND（GM） | MTE3 |
| `hivm.l12ub` | NZ（L1）→ ND（UB） | MTE1 |
| `hivm.mmadL1` | zN(A) + nZ(B) → Fractal(C) | M（经 MTE1 加载） |
| `hivm.fixpipe` | Fractal(L0C) → ND/NZ/DN | FIX |

`nd2nz` 可选属性：`dst_continuous`、`init_out_buffer`、`pad_value`。`mmadL1` 实现 `OpLayoutInterface`，提供 `getOperandA/B/C/BiasLayout()` 等查询。fixpipe 的 `dma_mode` 三态：NZ2ND(0)、NZ2DN(1，仅 950)、NZ2NZ(2，默认 `normal`)。

## 数据流对比

910B：
```
GM(ND) ─[MTE2+nd2nz]→ L1(NZ) ─[MTE1]→ L0A(zN)/L0B(nZ) ─[M]→ L0C(Fractal) ─[FIX NZ2ND]→ GM(ND)
```

950 关键差异：L0C 扩到 256KB；支持 L0C→UB 直通（fixpipe 直接输出到 UB）；新增 NZ2DN 模式；支持紧耦合缓冲区与 Dual Dst（ROW_SPLIT/COLUMN_SPLIT）把 Cube 结果切给两个 Vector 单元。

## 易混点

- `nd2nz`（MTE2，GM→L1 方向 ND 转 NZ）与 fixpipe `NZ2ND`（FIX，L0C 输出方向 NZ 转 ND）方向相反、Pipeline 不同。
- DOTA_ND 与 ND 的区别：DOTA_ND 是矩阵乘法中 A 操作数的专用标记，携带必选的 `transpose` 属性。
- `fractalSizes` 仅 Fractal 布局需要；zN/nZ/ND 等标准布局不设。

Cube 通路的 Pipeline 细节见 [pipeline_execution_model.md](pipeline_execution_model.md)，地址空间见 [memory_hierarchy.md](memory_hierarchy.md)。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- [HIVMAttrs.td#L84-L101](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L84-L101)
- [HIVMAttrs.td#L103-L165](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L103-L165)
- [HIVMDMAOps.td#L328-L368](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L328-L368)
- [HIVMDMAOps.td#L370-L386](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L370-L386)
- [HIVMDMAOps.td#L388-L404](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L388-L404)
- [HIVMMacroOps.td#L152-L172](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L152-L172)
- [HIVMAttrs.td#L847-L859](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L847-L859)
- [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
