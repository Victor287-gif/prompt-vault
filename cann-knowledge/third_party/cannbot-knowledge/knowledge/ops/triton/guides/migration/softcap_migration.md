---
type: Guide
title: Softcap GPU→NPU 迁移对照
description: Softcap（softcap*tanh(x/softcap)，fwd/bwd 逐元素算子）GPU→NPU 迁移 Diff：BLOCK_NUM
  多块循环、autotune 搜索空间重构、enable_vf_fusion 与 libdevice 路径变更。
tags: [task.migration, topic.attention, topic.quality.compatibility.version_migration, topic.compute.elementwise.arithmetic, topic.programming_model.execution_unit.vector, topic.tiling]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Math_Ops/softmax.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/tutorials/06-fused-attention.py
  role: example
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---
# Softcap GPU→NPU 迁移对照

基于 `softcap.py`（GPU）与 `softcap_npu.py`（NPU）对比。适用：注意力分数软截断 `softcap * tanh(x / softcap)` 等逐元素算子（fwd/bwd 双子 kernel）迁移。

## Diff 1：导入差异

```python
# GPU
from maybe_triton_jit import maybe_triton_jit
from triton.language.extra import libdevice
from .utils import is_hopper, is_ampere
# NPU
import torch_npu
# from maybe_triton_jit import maybe_triton_jit       # 注释，改用 @triton.autotune
from triton.language.extra.cann import libdevice      # CANN 后端路径
from utils import is_hopper, is_ampere
```

`libdevice.tanh` 两路径 API 一致，底层分别调用 CUDA / CANN 实现。

## Diff 2：Autotune 装饰器与配置

```python
# GPU：@maybe_triton_jit(configs=get_fwd_config(), key=[])
# 各架构固定 1 个 Config：triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=4)
# NPU：
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM, "multibuffer": MB, "BLOCK_NUM": BN})
        for BM in [18725, 16384, 8192, 4096, 2048, 1024, 512]
        for MB in [True, False]
        for BN in [64, 32, 16, 8, 4, 2, 1]
    ],
    key=["n_elements"]
)
```

| 维度 | GPU | NPU |
|------|-----|-----|
| Config 数量 | 1（按架构选） | 7×2×7 = 98 |
| `num_stages`/`num_warps` | 有 | **必须移除**（NPU 不支持，保留会编译报错） |
| NPU 专属参数 | 无 | `multibuffer`（双缓冲）、`BLOCK_NUM`（多块循环） |
| key | `[]` | `["n_elements"]`（按数据量选配置） |

## Diff 3：核内结构 —— BLOCK_NUM 多块循环（核心差异）

```python
# GPU：无循环，单块
pid = tl.program_id(axis=0)
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
tl.store(y_ptr + offsets, y, mask=mask)
# NPU：BLOCK_NUM 多块循环
pid = tl.program_id(axis=0)
for i in range(BLOCK_NUM):
    block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)
```

每个 program 处理 `BLOCK_SIZE * BLOCK_NUM` 个元素，减少 program 数量、降低调度开销；`BLOCK_NUM=1` 时两者等价。反向 kernel（`dx = dy * (1 - y*y)`）做完全对称的迁移，计算逻辑不变。

## Diff 4：编译参数（NPU 特有）

```python
softcap_fwd_kernel[grid](x, y, numel, softcap,
    debug=True,               # 仅 fwd；生产环境应移除（增加编译时间、影响性能）
    enable_vf_fusion=True,    # VF 融合：cast→除→tanh→乘→cast 融合为更少指令，减少 GM↔UB 搬运
)
```

fwd 加了 `debug=True, enable_vf_fusion=True`；bwd 仅 `enable_vf_fusion=True`。

## Diff 5：Grid 计算

```python
# GPU
grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
# NPU
grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"] * META["BLOCK_NUM"]),)
```

## Diff 6：测试代码

`device="cuda"` → `"npu"`；加 `torch_npu.profiler`（30 次迭代）与 `torch.npu.synchronize()`；验证从 if/else 打印改为 `assert torch.allclose(...)`（适合自动化）。

## 910_95 特别注意（避坑重点）

1. **搜索空间过大**：98 个配置首次 autotune 耗时极长。用 `TRITON_PRINT_AUTOTUNING=1` 观察最优配置分布后收窄；控制在 20 个以内；可考虑固定 `BLOCK_NUM` 为 1 或 2 减少组合。
2. **BLOCK_SIZE 非对齐值**：`18725` 非 2 的幂次，不满足 32B 对齐（FP16 元素数须为 16 的倍数），导致性能下降。建议 `[512, 1024, 2048, 4096, 8192, 16384]`。
3. **multibuffer 对 UB 的影响**：910_95 默认 `multibuffer=False`；选 `True` 时 UB 可用从 256KB 减半至 128KB。bwd kernel 需同时缓冲 `dy` 和 `x`，`BLOCK_SIZE=16384` FP16 双缓冲需 128KB，余量很小。
4. **enable_vf_fusion 的 UB 开销**：VF 融合增加中间缓冲，与 `multibuffer=True` 同时开启可能 UB 溢出。建议把 `enable_vf_fusion` 也加入 autotune 搜索维度，或双缓冲时关闭。
5. **`num_stages`/`num_warps` 必须移除**：保留会导致 NPU 编译错误。
6. **`debug=True` 仅限调试阶段**，生产环境移除。