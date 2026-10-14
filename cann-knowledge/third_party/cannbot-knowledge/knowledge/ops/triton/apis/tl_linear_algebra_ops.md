---
type: API
title: 线性代数 API（tl.dot / tl.dot_scaled）与 Cube 管线
description: Triton-Ascend 矩阵乘 API：tl.dot、tl.dot_scaled 签名、NPU 精度模式映射（tf32→hf32）、dtype
  约束、ND2NZ 布局转换与分块策略。
tags: [task.lookup, topic.compute.composite, topic.programming_model.execution_unit.cube, topic.data_movement.layout_transform, topic.precision]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/tree/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Linear_Algebra_Ops/dot.md
  role: primary
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']

---
# 线性代数 API（tl.dot / tl.dot_scaled）与 Cube 管线

`tl.dot` 在 NPU 上映射为 Cube 单元（矩阵计算引擎，对应 GPU Tensor Core）的矩阵乘指令，累加精度固定 fp32/int32。

## tl.dot

```python
triton.language.dot(input, other, acc=None, input_precision=None, allow_tf32=None,
                    max_num_imprecise_acc=None, out_dtype=float32) -> tensor
```

| 参数 | 说明 |
|------|------|
| `input` / `other` | 2D 或 3D（首维 batch）张量，dtype 必须相同；支持 int8/fp16/bf16/fp32；K 维须匹配 |
| `acc` | 累加器，非 None 时结果累加到其上；**不支持 fp16 累加**（Cube 累加器固定 fp32/int32） |
| `input_precision` | NPU 可选 `"ieee"`（默认）、`"hf32"`；GPU 另有 `"tf32"`/`"tf32x3"` |
| `allow_tf32` | **已弃用**，True 时映射为 `input_precision="hf32"` |
| `max_num_imprecise_acc` | **NPU 不支持**，传入被忽略（置 0），所有累加用完整精度 |
| `out_dtype` | 默认 float32；NPU 支持 float32 / int32（int8 输入时）/ float16；**bfloat16 不支持**，需 fp32/fp16 后手动 cast |

**GPU→NPU 精度模式映射**：

| GPU | NPU | 说明 |
|-----|-----|------|
| `"tf32"` | `"hf32"` | NPU 特有半精度 fp32 格式，精度介于 fp16 与 fp32 之间 |
| `"tf32x3"` | 不支持 | - |
| `"ieee"` | `"ieee"` | IEEE fp32 |
| `allow_tf32=True` | `"hf32"` | 自动映射 |

`input_precision="hf32"` 且输入非 fp32 时精度设置被忽略，回退 `"ieee"`。

## tl.dot_scaled

微缩放格式（microscaling）矩阵乘。

```python
triton.language.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
                           acc=None, out_dtype=float32, lhs_k_pack=True, rhs_k_pack=True)
```

- `lhs_format` / `rhs_format`：`"e4m3"` / `"e5m2"` / `"e2m3"` / `"e3m2"` / `"e2m1"` / `"bf16"` / `"fp16"`
- `lhs_scale` / `rhs_scale`：ue8m0 float8 缩放因子（表示为 int8 张量；GPU 为 uint8）
- `out_dtype` 仅支持 float32；lhs/rhs 必须同 dtype
- **平台限制**：910B 上 lhs/rhs 仅支持 bf16/fp16；910_95 额外支持 uint8/e5m2/e4m3

## NPU 适配要点

1. **ND2NZ 转换开销**：Cube 要求 NZ 数据排布，Triton kernel 数据为 ND（行优先），编译器自动插入 ND2NZ（输入）与 NZ2ND（输出）转换，消耗额外 UB 与周期。优化方向：增大分块摊薄转换开销；增大 BLOCK_K 减少重复转换次数。
2. **分块策略**：权衡 UB 容量与 Cube 利用率。典型起点：fp16 用 BLOCK_M=128 / BLOCK_N=128 / BLOCK_K=32；fp32 需更多 UB 可用 64/64/64。
3. **out_dtype=bf16 不支持**：`result.to(tl.bfloat16)` 手动转换。

## 代码示例

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

int8 量化矩阵乘：累加器 `tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)`，`tl.dot(a, b, acc, out_dtype=tl.int32)`。

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [core.py#L1518-L1558](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1518-L1558)
- [core.py#L1561-L1581](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/core.py#L1561-L1581)
- [semantic.py#L1559-L1626](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1559-L1626)
- [semantic.py#L1659-L1730](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/python/triton/language/semantic.py#L1659-L1730)
