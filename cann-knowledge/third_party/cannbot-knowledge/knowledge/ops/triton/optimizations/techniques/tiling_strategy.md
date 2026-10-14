---
type: Optimization
title: NPU 分块策略（Tiling）
description: Triton-Ascend Tiling 分块完整规范：UB 容量约束、32B/512B 对齐、存算并行条件、一级/二级分块计算方法与 UB
  空间估算。
tags: [task.optimization, topic.tiling, topic.implementation.memory_management, topic.alignment, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# NPU 分块策略（Tiling）

## 触发条件

- 编译报 `ub overflow, requires xxx bits while yyy bits available`
- 需要确定 BLOCK_SIZE / BLOCK_SIZE_SUB
- MTE2 占比过高（>50%）怀疑 Tiling 过小
- 尾轴不对齐导致性能下降

## UB 容量约束

| 芯片型号 | UB 大小 | 开 multiBuffer 后可用 | 对应 bits |
|----------|---------|---------------------|-----------|
| Ascend910B (A2) | 192 KB | 96 KB | 1572864 |
| Ascend910_95 | 256 KB | 128 KB | 2097152 |
| Ascend950 | 256 KB | 128 KB | 2097152 |

UB 空间需容纳所有片上同时存在的张量（输入 + 输出 + 中间结果 + multiBuffer 副本）：

```
所需 UB 空间 = (输入张量数 + 输出张量数 + 中间张量数) × BLOCK_SIZE × dtype_bytes × multiBuffer副本数

例：fp16 add_kernel，2 输入 + 1 输出，multiBuffer=2
所需 = 3 × BLOCK_SIZE × 2 × 2 = 12 × BLOCK_SIZE bytes
A2 上限 196608 bytes → 最大 BLOCK_SIZE = 16384
```

## 对齐要求

| 算子类型 | 尾轴对齐 | 各 dtype 最小元素倍数 |
|----------|---------|----------------------|
| VV（纯 Vector） | 32B | fp16/bf16: 16；fp32: 8；int8: 32 |
| CV（含 Cube） | 512B | fp16: 256；fp32: 128 |

不满足时硬件自动补齐 → 额外内存 + 无效搬运 + 性能恶化。

**借轴转置**（适用 `tensor.numel() % 256B == 0`，如 `(2048, 3)` bf16 尾轴 6B 不对齐）：

```python
# 按 1D load，避免自动补齐
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride
                     + doffs * 3 + tl.arange(0, 2048 * 3))
# 长轴裂出对齐轴借给短轴
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
```

## 存算并行实现条件

编译器默认 `multiBuffer=True`，但需同时满足：

1. **for 循环 Tiling**：多次迭代才能形成"搬入+计算"流水
2. **无数据依赖**：当前迭代不依赖下一迭代搬入的数据
3. **UB 空间充足**：multiBuffer 使可用空间减半

multiBuffer 失效常见原因：

| 原因 | 解决 |
|------|------|
| 搬运与计算存在依赖 | `care_padding=False` 减少同步 |
| 无 Tiling 切分单次完成 | 添加 for 循环 |
| UB 空间不足 | 减小 BLOCK_SIZE_SUB |

## 分块计算

一级分块（决定每核总量与 coreDim）：

```
NUM_BLOCKS = ceil(N / BLOCK_SIZE);  coreDim = NUM_BLOCKS
约束：coreDim <= 65535；推荐对齐物理核数；BLOCK_SIZE 为 2 的幂次
```

二级分块（决定单次迭代 UB 用量）：

```
num_sub_blocks = ceil(BLOCK_SIZE / BLOCK_SIZE_SUB)
约束：单次迭代 UB 用量 <= UB可用 / multiBuffer副本数；满足对齐；2 的幂次
```

```python
def estimate_ub_usage(num_tensors, block_size, dtype_bytes, multibuffer=True):
    num_copies = 2 if multibuffer else 1
    return num_tensors * block_size * dtype_bytes * num_copies
```

## 二级分块标准模式

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                       BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, overwrite_vals, mask=mask)

MAIN_BLOCK_SIZE = 32768  # 保 coreDim 合规
SUB_BLOCK_SIZE = 1024    # 控 UB 用量
```

## 要点

1. UB 容量是硬约束；开 multiBuffer 可用空间减半
2. 32B/512B 对齐必须满足
3. 无 for 循环则无流水，multiBuffer 无法使能
4. BLOCK_SIZE 候选值必须 2 的幂次
5. multiBuffer 后性能反降 → UB 紧张致分块过小，减 BLOCK_SIZE_SUB 或关 multiBuffer
6. 判断 Tiling 合理性：`aiv_mte2_ratio > 50%` 且 vec 低 → 分块可能过小


## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [utils.py#L28-L61](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py#L28-L61)
- [programming_guide.md](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/programming_guide.md)
- [tile_generator.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/tile_generator.py)
- [utils.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py)
