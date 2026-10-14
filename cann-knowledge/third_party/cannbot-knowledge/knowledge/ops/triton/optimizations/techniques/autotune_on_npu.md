---
type: Optimization
title: NPU Autotune 配置与 AutoTilingTuner
description: NPU 上 @triton.autotune 与 GPU 的核心差异（Config 仅含 Tiling 参数）、候选配置空间、key/hints/reset_to_zero
  用法，以及 AutoTilingTuner 自动分块调优机制。
tags: [task.optimization, topic.tiling, topic.implementation.memory_management, topic.alignment]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/triton/autotune.md
  role: primary
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/triton/jit.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, '950']
---
# NPU Autotune 配置与 AutoTilingTuner

触发场景：GPU kernel 迁移到 NPU 需重配 autotune；固定 Tiling 需适配多 shape；已有配置性能不理想需调整候选范围或 key。

## GPU → NPU 核心差异（迁移第一要点）

| 维度 | GPU Autotune | NPU Autotune |
|------|-------------|--------------|
| Config 内容 | Tiling 参数 + `num_stages` + `num_warps` | **仅 Tiling 参数** |
| `num_stages` | 软件流水线 1~4 | **不使用**，NPU 硬件自动管理流水线 |
| `num_warps` | 4/8/16/32 | **不使用**，SIMD 模式固定 `num_warps=1` |
| 候选值约束 | 灵活 | **必须为 2 的幂次**（128, 256, 512, 1024...） |
| Grid 更新 | `lambda meta:` | 同样支持 `lambda meta:` |
| 缓存机制 | 按 key 值缓存 | 按 key 值缓存，相同 key 复用 |

GPU 写法（**NPU 上错误**）：

```python
triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4)
```

NPU 正确写法：

```python
triton.Config({"BLOCK_M": 128, "BLOCK_N": 64})
```

## 候选配置空间

| 参数名模式 | 含义 | 典型候选值 |
|-----------|------|-----------|
| `BLOCK_SIZE` | 一维分块 | `[128, 256, 512, 1024, 2048]` |
| `BLOCK_M` / `BLOCK_SIZE_M` | M 维分块 | `[32, 64, 128, 256]` |
| `BLOCK_N` / `BLOCK_SIZE_N` | N 维分块 | `[32, 64, 128]`（受 UB 限制） |
| `BLOCK_K` / `BLOCK_SIZE_K` | K 维（归约轴）分块 | `[32, 64, 128, 256]` |
| `GROUP_SIZE_M` | M 维分组 | 8（通常固定不搜索） |
| `SPLIT_K` | K 维拆分因子 | `[8, 16]` |
| Flash Attention BLOCK_M/N | attention 场景 | `[32, 64, 128]`（UB 压力大） |

原则：候选值 2 的幂次；小值让小 shape 分核数匹配物理核数，大值让大 shape 充分利用 UB；总配置数控制在 **20 个以内**；最优值总落在边界时扩展范围。

多维组合用列表推导生成笛卡尔积，用 `filter` 剪枝不合理组合：

```python
def keep(config):
    return config.kwargs["BLOCK_M"] % config.kwargs["BLOCK_N"] == 0

@triton.autotune(
    configs=list(filter(keep, [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN})
        for BM in [32, 64, 128] for BN in [32, 64, 128]
    ])),
    key=["M", "N"],
)
```

## key 参数

`key` 指定调优维度：值变化触发重新调优，相同值复用缓存。选择**代表数据规模的变量**，不要选 Tiling 参数名（`key=["BLOCK_SIZE"]` 错误）或指针参数。

| 场景 | 推荐 key |
|------|---------|
| 一维向量 | `["n_elements"]` |
| 矩阵乘法 | `["M", "N"]` 或 `["M", "N", "K"]` |
| Flash Attention | `["QK_DIM", "V_DIM"]` |
| 变长序列 | `["max_seqlen_q", "max_seqlen_k"]` |

两种形式：

```python
key=["M", "N", "K"]              # 列表：按顺序映射到轴 x, y, z
key={'x': 'M', 'rx': 'K'},       # 字典：显式映射，必须同时给 hints
hints={
    'split_params': {'x': 'BLOCK_M'},     # 分核轴：与 tl.program_id 关联
    'tiling_params': {'rx': 'BLOCK_K'},   # 分块轴：与 tl.arange 关联
    'low_dim_axes': ['y'],                # 低维轴：影响对齐与分块
    'reduction_axes': ['x'],              # 归约轴（不带 'r' 前缀）
}
```

轴名约定：普通轴 `x, y, z, w, v, t`；归约轴加前缀 `r`（`rx, ry, ...`）。

## Grid 动态更新与装饰器顺序

Grid 依赖 Tiling 参数时必须用 `lambda META:` 动态计算；Grid 由物理核数固定时不需要：

```python
grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
kernel[grid](x, w, y, M, N, K)
```

装饰器顺序固定：`@libentry()` → `@triton.autotune` → `@triton.jit`（autotune 必须在 jit 上方）。

## reset_to_zero

kernel 用 `tl.atomic_add` 累加输出时，autotune 每个 Config 都运行一次会导致结果被多次累加。用 `reset_to_zero` 在每次评估前将指定张量清零：

```python
@triton.autotune(configs=[...], key=["QK_DIM", "V_DIM"], reset_to_zero=["dq_ptr"])
```

`tl.store` 覆盖写场景不需要。

## AutoTilingTuner（自动分块调优）

当 `configs=[]`（空列表）或未提供时自动启用：根据算子 AST 分析与硬件参数自动生成候选分块配置。混合模式（用户提供部分 configs）下会额外自动生成配置。

内部 Parser 从 AST 提取轴信息：

| Parser | 识别模式 |
|--------|---------|
| SplitAxesParser | `tl.program_id` 及 grid-stride for 循环 → split_params |
| TilingAxesParser | `tl.arange` / for 循环 `range(0, XBLOCK, XBLOCK_SUB)` → tiling_params |
| ReductionAxesParser | `tl.sum/tl.max/tl.min/tl.argmax` 等归约 → reduction_axes |
| LowDimsAxesParser | 最低维 `tl.arange` 参与切片 → low_dim_axes |
| PtrNumsParser | 参与 load/store 地址计算的指针数 → UB 估算 |

tile_generator 递减策略：分核轴从最大值减半直到 program 数 ≤ 物理核数；分块轴按 2 的幂递减；低维轴同步递减保持比例；非 SIMT 模式按 32B 对齐向上取整。关键阈值：

```python
max_numel_threshold = ub_size_in_kbytes * 1024 // dtype_bytes // num_buffers
# 例：192 * 1024 / 2 / 3 = 32768 elements（fp16, 3 buffers）
```

每个基础配置自动生成 `multibuffer=True/False` 两个变体参与调优。

## 910_95 特别注意

- 910_95 UB 为 **248KB**（256KB 预留 8KB；910B 为 192KB），候选值上限可适当增大：fp16 单 buffer 最大元素数 910_95 ≈ 126976，910B ≈ 98304
- Tiling 值须满足 32B 对齐（fp16 元素数为 16 的倍数；2 的幂次天然满足）
- 910_95 支持 SIMT VF 模式与 L0C→UB fixpipe 直通，CV 融合 kernel 的候选值可更激进
- 估算公式：`最大 Tiling 元素数 ≈ UB_SIZE / (dtype字节数 × buffer数)`

## 调试与验证

```bash
export TRITON_PRINT_AUTOTUNING=1          # 打印解析过程与最优配置
export TRITON_AUTOTUNE_PARALLEL_COMPILE=1 # 并行编译加速调优（默认开启）
```

常见问题：调优时间过长 → 减少配置数 / 并行编译 / `prune_configs_by` 预剪枝；自动生成配置数为 0 → 检查 key 映射并开 `TRITON_PRINT_AUTOTUNING` 看解析日志；强制重调 → 重启进程或清 Triton 缓存（结果按 key 进程内缓存）。


## 相关源

- 主源：docs_triton_ascend/05-Performance-Optimization/03-autotune-guide.md
- 合并来源：docs_for_triton_agent/10-autotune-on-npu.md（GPU→NPU 差异表、配置空间、reset_to_zero、Grid 更新）

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [autotuner.py#L45-L124](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/autotuner.py#L45-L124)
- [utils.py#L98-L105](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py#L98-L105)
- [autoparser.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/autoparser.py)
- [tile_generator.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/tile_generator.py)
- [autotuner.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/autotuner.py)
- [utils.py](https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/third_party/ascend/backend/runtime/utils.py)
