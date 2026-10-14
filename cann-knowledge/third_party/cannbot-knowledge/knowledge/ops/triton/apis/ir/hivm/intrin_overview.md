---
type: API
title: HIVM 内建指令操作总览（Intrin Operations）
description: HIVM 内建指令（hivm.GET.BLOCK.IDX / SET.FLAG.IMM / SET.CROSS.CORE / SET.INTRA.BLOCK
  等）：继承 LLVM_IntrOpBase 的最低层操作，六类指令清单、SET.CROSS.CORE config 编码、INTRA.BLOCK 的 ID 映射与计数器语义、高层操作到内建指令映射表。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization, topic.architecture]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM 内建指令操作总览

内建指令操作（Intrin Operations）是 HIVM 最底层操作，直接对应 NPU 硬件指令，继承 `LLVM_IntrOpBase`，在 Lowering 最终阶段由高层 HIVM 操作转换而来。用户不直接编写；读最终 IR 时据此理解高层语义到硬件指令的映射。

## 指令分类

### 1. Block 索引

| 指令 | 高层操作 | 说明 |
|------|---------|------|
| `hivm.GET.BLOCK.IDX` | `hir.get_block_idx` | 当前 Block 索引 |
| `hivm.GET.BLOCK.NUM` | `hir.get_block_num` | Block 总数 |
| `hivm.GET.SUBBLOCKID` | `hir.get_sub_block_idx` | 当前 Sub-Block 索引 |
| `hivm.GET.SUBBLOCKDIM` | `hir.get_sub_block_num` | Sub-Block 总数 |

### 2. Pipe 同步

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.FLAG.IMM` / `hivm.WAIT.FLAG.IMM` | set_pipe, wait_pipe, event_id（均 I64Attr） | Event Flag 设置/等待，立即数模式 |
| `hivm.SET.FLAG.REG` / `hivm.WAIT.FLAG.REG` | set_pipe, wait_pipe（I64Attr）+ event_id（I\<64\> 寄存器） | 寄存器模式 |
| `hivm.BARRIER` | pipe（I64Attr） | Pipeline 屏障 |

IMM 模式把参数编码进指令（编译期常量、效率高）；REG 模式从寄存器读（运行时动态值）。

### 3. FFTS 同步

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.FFTS.BASE.ADDR` | addr（I\<64\>） | 设置 FFTS 基地址寄存器 |
| `hivm.SET.CROSS.CORE` | pipe（I64Attr）, config（I\<64\>） | FFTS 跨核同步发送 |
| `hivm.WAIT.FLAG.DEV.REG` | flag_id（I\<64\>） | FFTS Block/Sub-Block 等待（寄存器） |
| `hivm.WAIT.FLAG.DEV.PIPE.IMM` | pipe, flag_id（均 I64Attr） | 等待（全立即数） |
| `hivm.WAIT.FLAG.DEV.PIPE.REG` | pipe（I64Attr）, flag_id（I\<64\>） | 等待（Pipe 立即数 + Flag 寄存器） |

**SET.CROSS.CORE config 编码**：

```
config = 0x0001 | ((mode & 0x0f) << 4) | ((flagID & 0x0f) << 8)
mode: 0=跨 Block，1=跨 Sub-Block，2=Block 内
flagID: 8-bit，每个 flag ID 有一个计数器
```

### 4. Intra-Block 同步（Cube ↔ Vector，Mix-Mode Block）

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.INTRA.BLOCK.mode` / `hivm.WAIT.INTRA.BLOCK.mode` | pipe（I64Attr）, sync_id（I\<64\>） | 寄存器模式 |
| `hivm.SET.INTRA.BLOCKI.mode` / `hivm.WAIT.INTRA.BLOCKI.mode` | pipe, sync_id（均 I64Attr） | 立即数模式 |

**ID 映射**（1 CUBECORE + 2 VECCORE 的 Mix Block）：

| SET 来源 | 目标 | ID 映射 |
|----------|------|---------|
| VECCORE0 ID 0-15 | CUBECORE | ID 0-15 |
| VECCORE1 ID 0-15 | CUBECORE | ID 16-31 |
| CUBECORE ID 0-15 | VECCORE0 | ID 0-15 |
| CUBECORE ID 16-31 | VECCORE1 | ID 0-15 |

**计数器语义**：每个 sync ID 一个 4-bit 计数器；SET 递增对方 Core 对应 ID 的计数器（不阻塞当前 Pipeline）；WAIT 在计数器为 0 时阻塞，否则递减。

### 5. 控制指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.SET.MASK.NORM` | 无 | Mask 设为正常模式 |
| `hivm.GET.CTRL` / `hivm.SET.CTRL` | 无 / config（I\<64\>） | 控制寄存器读/写 |
| `hivm.SBITSET0` / `hivm.SBITSET1` | x, idx（I\<64\>） | 状态位置 0 / 置 1 |

### 6. 缓存指令

| 指令 | 参数 | 说明 |
|------|------|------|
| `hivm.DCCI.DST` | ptr(LLVMPointer), entire, dst | 清除/无效化 GM 数据缓存 |
| `hivm.DCCI.DST.UB` | 同上 | 清除/无效化 UB 数据缓存 |

## 高层操作 → 内建指令映射

| 高层操作 | 内建指令 |
|---------|---------|
| `hir.set_flag` / `hir.wait_flag`（静态 Event ID） | `SET.FLAG.IMM` / `WAIT.FLAG.IMM` |
| `hir.set_flag` / `hir.wait_flag`（动态 Event ID） | `SET.FLAG.REG` / `WAIT.FLAG.REG` |
| `hir.pipe_barrier` | `BARRIER` |
| `hir.set_ffts_base_addr` | `SET.FFTS.BASE.ADDR` |
| `hir.sync_block_set` | `SET.CROSS.CORE` / `SET.INTRA.BLOCKI.mode` |
| `hir.sync_block_wait` | `WAIT.FLAG.DEV.REG` / `WAIT.INTRA.BLOCKI.mode` |
| `hir.set_mask_norm` | `SET.MASK.NORM` |
| `hir.set_ctrl` | `SET.CTRL` |
| `hir.dcci` | `DCCI.DST` / `DCCI.DST.UB` |

高层同步操作语义见 [sync_pipe_sync.md](sync_pipe_sync.md) 与 [sync_block_sync.md](sync_block_sync.md)。

相关：[sync_overview.md](sync_overview.md)、[sync_pipe_sync.md](sync_pipe_sync.md)、[sync_block_sync.md](sync_block_sync.md)、[overview.md](overview.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMIntrinOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMIntrinOps.td)
