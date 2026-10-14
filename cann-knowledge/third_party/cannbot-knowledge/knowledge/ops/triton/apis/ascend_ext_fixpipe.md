---
type: API
title: fixpipe 操作（L0C → UB 搬运与融合后处理）
description: fixpipe 将 Cube 计算结果从 L0C 搬运到 UB（仅 910_95），支持 NZ↔ND 布局转换、双目的输出、预量化与预 ReLU
  融合。
tags: [task.lookup, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.data_movement.layout_transform, topic.compute.quantization, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/builder.py
  role: implementation
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3, '950']

---
# fixpipe 操作（L0C → UB 搬运与融合后处理）

fixpipe 是昇腾 NPU 的关键硬件流水线，将 Cube 计算结果从 L0C 搬到其他存储层级，同时可选执行格式转换、量化和 ReLU 后处理。910_95 系列支持 L0C→UB 直达，实现 Cube 计算到 Vector 后处理的零拷贝流水线；A2/A3 系列仅支持 L0C→GM/L1。

## 数据通路

```
Cube 计算 (PIPE_M) → L0C → fixpipe (PIPE_FIX) → UB → Vector 后处理 (PIPE_V)
```

- **L0C**：Cube 输出缓冲区，数据以 NZ（Narrow Z，昇腾特有分块排布）格式存储
- **UB**：Unified Buffer，Vector 的输入/输出缓冲区
- **fixpipe**：连接 L0C 与 UB 的硬件流水线，支持格式转换和融合后处理

## 模式枚举

### FixpipeDMAMode — DMA 传输模式

| 枚举值 | 说明 | 源格式 → 目标格式 |
|--------|------|-------------------|
| `NZ2ND` | NZ 到 ND 转换（默认，最常用） | NZ → ND 密集格式 |
| `NZ2DN` | NZ 到 DN 转换 | NZ → DN |
| `NZ2NZ` | 无格式转换 | NZ → NZ（后续操作可接受 NZ 时避免转换开销） |

### FixpipeDualDstMode — 双目的输出模式

| 枚举值 | 说明 |
|--------|------|
| `NO_DUAL` | 单目的输出（默认） |
| `COLUMN_SPLIT` | 列分割双输出 |
| `ROW_SPLIT` | 行分割双输出 |

### FixpipePreQuantMode — 预量化模式

`NO_QUANT`（默认）/ `F322BF16`（fp32→bf16）/ `F322F16`（fp32→fp16）/ `S322I8`（int32→int8）。

### FixpipePreReluMode — 预 ReLU 模式

`NO_RELU`（默认）/ `NORMAL_RELU` / `LEAKY_RELU` / `P_RELU`。

> 注意：当前 Python 层 `fixpipe` 函数固定使用 `NO_QUANT` 和 `NO_RELU`，高级功能通过内部语义层调用，未来版本可能开放。

## API 参考

```python
@builtin
def fixpipe(
    src: tl.tensor,                                  # 源张量，必须位于 L0C
    dst: bl.buffer,                                  # 目标缓冲区，必须位于 UB
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    _builder=None,
) -> None
```

**对齐约束（违反会导致运行时错误）：**

| 数据位宽 | 条件 | 对齐要求 |
|----------|------|----------|
| 32 位（fp32/int32） | 最后一维 | 对齐到 8 |
| 32 位（非 NZ2ND 模式） | 最后一维 | 对齐到 16 |
| 32 位（COLUMN_SPLIT 模式） | 最后一维 | 对齐到 32 |
| 32 位（NZ2DN 模式） | 第一维 | 对齐到 8 |
| 16 位（fp16/int16/bf16） | 最后一维 | 对齐到 16 |
| 16 位（NZ2DN 模式） | 第一维 | 对齐到 16 |

**平台限制：仅 Ascend 910_95 系列支持**，其他平台调用抛出 `RuntimeError`。

## 典型使用场景

1. **矩阵乘法后处理**：`tl.dot` 结果在 L0C，必须经 fixpipe 搬到 UB 才能做加 bias、激活等向量操作。
2. **量化融合**：搬运同时完成 fp32→bf16/fp16/int8 量化，避免额外转换开销。
3. **ReLU 融合**：搬运同时完成 ReLU 激活，实现计算与搬运的流水线重叠。

## 代码示例

### 基本 fixpipe 使用（matmul 结果落 UB 再 store）

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

@triton.jit
def matmul_fixpipe_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub_buf, dma_mode=al.FixpipeDMAMode.NZ2ND)

    acc_tensor = bl.to_tensor(ub_buf)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc_tensor)
```

### Cube-Vector 协同中的 fixpipe

```python
@triton.jit
def cube_vector_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    with al.scope(core_mode="cube"):
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc = tl.dot(a, b, acc)
        ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
        al.fixpipe(acc, ub_buf)
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        acc_tensor = bl.to_tensor(ub_buf)
        result = tl.sigmoid(acc_tensor)
        tl.store(c_ptr + ..., result)
```

## NPU 适配要点

1. **src 必须是 tensor、dst 必须是 buffer**：src 为 L0C 上的 `tl.tensor`，dst 为 UB 上的 `bl.buffer`；dst 地址空间必须是 `ascend_address_space.UB`，否则抛 `TypeError`。
2. **与直接 load/store 的区别**：fixpipe 是硬件级搬运，L0C 直读直写 UB 并支持格式转换与融合后处理；load/store 需经 GM 中转，效率低。
3. **dual_dst_mode 用途**：将一个 Cube 结果拆为两个输出写入不同 UB 缓冲区（按列/按行），某些融合算子中可减少搬运次数。


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L273-L333](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L273-L333)
- [core.py#L247-L270](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/core.py#L247-L270)
- [semantic.py#L132-L148](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/language/cann/extension/semantic.py#L132-L148)
