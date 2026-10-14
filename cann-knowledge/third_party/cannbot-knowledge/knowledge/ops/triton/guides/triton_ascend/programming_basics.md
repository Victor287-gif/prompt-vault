---
type: Guide
title: 编程基础：核心概念与标准 Kernel 结构
description: Triton Ascend 采用与 Triton GPU 类似的执行模型：。
tags: [task.implementation, topic.tiling, topic.implementation.masking, topic.compilation, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/README.md
  role: supporting
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# 编程基础：核心概念与标准 Kernel 结构

## 1. 编程基础

### 1.1 核心概念

Triton Ascend 采用与 Triton GPU 类似的执行模型：

- **Kernel（内核）**：用 `@triton.jit` 装饰的函数，在 NPU 上并行执行
- **Grid（网格）**：启动的并行程序总数，由 `kernel[grid](args)` 指定
- **Block（块）**：每个程序处理的数据块，由 `tl.program_id(0)` 区分

#### 内存层次
- **全局内存**: 主内存，所有程序可访问，延迟高
- **共享内存**: 块内共享，延迟低，容量有限
- **寄存器**: 每个线程私有，最快访问

### 1.2 标准 Kernel 结构（五步模式）

```python
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    # Step 1: 计算当前核的起始偏移（核间切分）
    xoffset = tl.program_id(0) * XBLOCK

    # Step 2: 核内循环（核内切分）
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        # Step 3: 构造数据索引
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

        # Step 4: 边界处理（mask）
        xmask = x_index < xnumel

        # Step 5: 加载 → 计算 → 存储
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)
```

### 1.3 内核启动方式
```python
def launch_kernel(input_tensor, output_tensor):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)

    kernel[grid](
        output_tensor, input_tensor, input_tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

### 1.4 边界处理（Mask）

使用 mask 防止越界访问，确保只处理合法范围内的数据：

```python
xmask = x_index < xnumel
x = tl.load(in_ptr0 + x_index, mask=xmask, other=0.0)
tl.store(out_ptr0 + x_index, ret, mask=xmask)
```

### 1.5 编程模式

#### 1.5.1 向量操作模式
适用于元素级运算：加法、乘法、激活函数等。

```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)
```

#### 1.5.2 归约模式
适用于求和、最大值、最小值等聚合操作。

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # 块内归约
    block_sum = tl.sum(data, axis=0)

    # 原子操作写回全局内存
    if pid == 0:  # 只有第一个块写入结果
        tl.atomic_add(output_ptr, block_sum)
```

#### 1.5.3 矩阵乘法模式
使用块指针高效处理 2D 数据。

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # 获取程序 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 主循环：沿 K 轴迭代
    for k in range(0, K, BLOCK_SIZE_K):
        # 创建块指针
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )

        # 加载数据块
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # 矩阵乘加
        accumulator += tl.dot(a, b)

    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
```

### 1.6 Autotune 使用

Triton 支持 `autotune` 自动调优，但 **Ascend NPU 不支持 `num_warps`、`num_ctas`、`num_stages` 等 CUDA 专用调优参数**。在 Ascend 上主要调优：

- `BLOCK_SIZE` / `XBLOCK` / `XBLOCK_SUB` 等分块参数
- `grid` 维度（核数）

```python
@triton.autotune(
     configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}),
    ],
    key=['M', 'N', 'K'],  # 当这些参数变化时触发重新autotune
)
@triton.jit
ddef matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, # configs中的参数必须声明为constexpr
    BLOCK_SIZE_N: tl.constexpr, # configs中的参数必须声明为constexpr
    BLOCK_SIZE_K: tl.constexpr, # configs中的参数必须声明为constexpr
):
    # kernel实现
    pass

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),) # grid设置是tuple

    # 关键：调用时不要传递configs中的参数（BLOCK_SIZE_M等）
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # 不要写：BLOCK_SIZE_M=128  # 错误！autotune会自动传入
    )
    return c
```
---
