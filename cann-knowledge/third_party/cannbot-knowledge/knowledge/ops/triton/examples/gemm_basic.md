---
type: Example
title: GEMM 基础代码骨架
description: 带 AutoTune 的标准 GEMM kernel 模板（ND layout, trans_b=True）。适用 M≥256, N≥256,
  K≥1024, fp16/bf16。
tags: [task.implementation, topic.compute.composite, topic.tiling, topic.data_movement.memory_access]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/03-matrix-multiplication.py
  role: example
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---
# GEMM 基础代码骨架

> 经验来源：exp_matmul_001 (40% boost), skill_tiling_diagonal (M,N≥512 时)

## 关键设计点

1. **AutoTune 搜索空间**：`BLOCK_M ∈ {128,256}`, `BLOCK_N ∈ {128,256}`, `BLOCK_K=256`, `GROUP_SIZE ∈ {8,16,32}`
2. **列优先 PID 分组**：GROUP_SIZE 控制相邻 PID 的 Core 分配，共享 L2 中 A 矩阵数据
3. **2D Block Pointer**：`tl.make_block_ptr` + `order=(1,0)` 行主序访问，编译器生成更优地址指令
4. **L0C 寄存器复用**：`acc` 保留在 L0C 中，K 轴迭代完成后一次性写回
5. **fp32 累加**：`acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)`，最终 `acc.to(tl.float16)`

## 核心结构

```python
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # 列优先 PID → (pid_m, pid_n) 映射
    # tl.make_block_ptr → L0C acc → K 轴循环 tl.dot → tl.store
```

完整骨架见源文件 `skeletons/gemm_basic.py`。

## 触发条件

- 计算图含 `torch.matmul`/`torch.mm`/`torch.bmm`/`F.linear`
- M≥128, N≥128, K≥256
- fp16/bf16, ND layout, trans_b=True

