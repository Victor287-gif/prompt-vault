---
type: Guide
title: NMS 算子（Non-Maximum Suppression） 实现手册
description: '算子类别: sorttopk + indexinggather 复合（顺序算法 + 每轮成对 IoU 计算）。'
tags: [task.implementation, topic.compute.composite, topic.implementation.masking, topic.tiling, topic.compute.ordering, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]
---
# NMS 算子（Non-Maximum Suppression） 实现手册

> 类别约束见 [object_detection_constraints.md](../constraints/object_detection_constraints.md)，
> 全局约束见 [global_constraints.md](../constraints/global_constraints.md)。

## §2 NMS 算子（Non-Maximum Suppression）

**算子类别**: `sort-topk` + `indexing-gather` 复合（顺序算法 + 每轮成对 IoU 计算）
**典型特征**: 输入 boxes `[N,4]` + scores `[N]`，输出定长 `selected_indices [max_output_size]` + `num_selected`。包含过滤 → argsort → 顺序遍历 → 每轮 IoU 抑制四个阶段
**性能基准**: 31/31 verify pass，几何平均加速比 **1.0187x** vs torch（达标 1.0x 正常归档阈值，达标 0.6x 用户目标）
**峰值内存**: 13.92 MB vs torch 14.16 MB（略优）

### 版本演进

| 版本 | geomean | kernel 延迟 | 关键技术 | 状态 |
|------|---------|------------|---------|------|
| v1 | 0.9857x | 13.21 ms | 分离 kernel + 逐轮 .item() 同步 | 已被 v2 替换 |
| v2 | 1.0187x | 4.38 ms | fused kernel + device counter + chunked loop | 当前归档 |

### §2.0 首次生成必读：为什么必须把主要框架写对

NMS 是一个**顺序算法**——每轮的候选范围 `[i+1, N)` 依赖上一轮的抑制结果。这意味着不能简单地把整个 N 轮循环搬进单个 kernel（kernel 内无法跨轮传递 suppressed 状态的动态变化），但逐轮 `.item()` 同步又是主要性能瓶颈。**首次生成如果把框架写偏（例如全 kernel 化、逐轮 host 同步、把 NMS 当 sort-topk 优化），后续迭代很难通过局部修bug把性能救回来**。

本章按 Layer 1→3 组织：
- **L1 是硬性约束**，首次生成必须全部满足；
- **L2 是 host chunked loop 骨架 + fused kernel 骨架**，必须一次写对；
- **L3 是关键技巧**，贴出消除同步开销的核心代码。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 必须复用 IOU 的 1D Grid + 交织循环模式
（见 §1 D5）
- **必须**在 IoU 抑制循环中使用 `pid = tl.program_id(0)` 配合 `for block_idx in range(pid, num_blocks, num_cores)` 的交织循环模式。
- **Why:** NMS 每轮的 IoU 计算本质上是当前 box 与候选 box 集合的成对计算，与 §3 IOU L1.2 同构；交织循环天然负载均衡，且 `num_blocks` 随 `N - cur_idx - 1` 递减，无需额外处理。
- **How to apply:** 直接套用 §3 IOU L1.2 骨架，把 `gtboxes` 替换为单个 `cur_box` 标量 broadcast。

#### L1.2 kernel 内必须用 tl.cdiv
（见 §1 D2）

#### L1.3 NPU 原生 NMS 与标准算法不兼容时必须替换 Golden
（见 §1 D3）
- **具体到 NMS**：`npu_nms_v4` 在 `pad_to_max_output_size=False` 时返回变长输出，与定长 `max_output_size` 输出不兼容；其内部排序/抑制规则不透明。

#### L1.4 suppressed 数组必须用 int8 + OR 合并
- **必须**用 `torch.zeros(N, dtype=torch.int8)` 表示 suppressed 状态，kernel 内用 `existing | suppress_mask.to(tl.int8)` 合并。
- **Why:** int8 节省显存（vs int32 节省 4x）；OR 操作是幂等的，支持多次 kernel 调用累加抑制标记，免原子操作。
- **How to apply:**
  ```python
  suppressed = torch.zeros(N, dtype=torch.int8, device=boxes.device)
  # kernel 内:
  existing = tl.load(suppressed_ptr + offs, mask=mask, other=0)
  new_supp = existing | suppress_mask.to(tl.int8)
  tl.store(suppressed_ptr + offs, new_supp, mask=mask)
  ```

#### L1.5 主循环必须在 Host 侧 Python 中执行，禁止全 kernel 化
- **必须**用 Python `while i < N` 主循环（v2 用 chunked loop），禁止试图把整个 N 轮循环搬进单个 kernel。
- **Why:** NMS 是**顺序算法**——每轮的候选范围 `[i+1, N)` 依赖上一轮的抑制结果；chunked loop 以 CHUNK=256 为粒度批量 launch，在 chunk 边界同步 suppressed 状态，是顺序算法在 Triton 上的最优落地形式。
- **How to apply:** 见 L2.1 Host 侧骨架。**禁止方向**: 把整个 N 轮循环写进单个 `@triton.jit`（kernel 内无法跨轮传递 suppressed 状态的动态变化）。

#### L1.6 grid_size 必须动态限制为实际核数
（见 §1 D4）

#### L1.7 cur_box 必须标量加载 + broadcast，禁止向量重复加载
- **必须**当前 box 的 4 个坐标 `(cur_x1, cur_y1, cur_x2, cur_y2)` 和 `cur_area` 用标量 `tl.load` 加载 1 次，与候选向量 `[BLOCK_N]` broadcast 计算 IoU。
- **Why:** 每轮 cur_box 是固定值，若每个候选 box 都重新加载 cur_box 会浪费访存带宽。
- **How to apply:** 见 L2.2 kernel 骨架。

#### L1.8 选中逻辑必须 fused 进 kernel 的 pid==0 分支（v2 关键约束）
- **必须**把 `selected_indices` 写入和 `num_sel` 自增逻辑放在 kernel 的 `if pid == 0` 分支内执行，禁止在 host 侧逐轮写入。
- **Why:** host 侧逐轮 `selected_indices[num_selected] = ...` 需要 `.item()` 同步读 `num_selected` 和 `suppressed[i]`，N 轮累计同步开销巨大（v1 的主要瓶颈）。fused 进 kernel 后，`num_sel_tensor` 常驻 device，kernel 内 `tl.store(num_sel_ptr, count+1)` 自增，只在 CHUNK 边界同步。
- **How to apply:**
  ```python
  # kernel 内:
  if pid == 0:
      count = tl.load(num_sel_ptr)
      should_select = not_supp & (count < max_output_size)
      orig = tl.load(orig_idx_ptr + cur_idx, mask=should_select, other=0)
      tl.store(selected_ptr + count, orig, mask=should_select)
      tl.store(num_sel_ptr, count + 1, mask=should_select)
  ```
  **注意**: `pid==0` 分支保证只有 1 个 core 写入，避免竞争；`mask=should_select` 保证未选中时不写入。

#### L1.9 必须用 chunked loop 消除逐轮 .item() 同步（v2 关键约束）
- **必须**用 `while i < N` + `CHUNK=256` 的分块循环，每 CHUNK 轮才 `.item()` 同步一次，禁止每轮同步。
- **Why:** v1 每轮 `.item()` 同步 N 次，是主要瓶颈；v2 同步次数降到 N/256，geomean 从 0.99x 提升到 1.02x。
- **How to apply:** 见 L2.1 Host 侧骨架。**CHUNK 选择**: 256 是经验值，平衡同步开销与提前终止粒度（CHUNK 过大则 max_output_size 达标后不能立即 break，浪费计算；CHUNK 过小则同步开销仍高）。

#### L1.10 所有 kernel 调用加 multibuffer=True
（见 §1 D6）

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧 chunked loop 骨架（v2）

```python
import torch_npu
import triton.runtime.driver as driver

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
        self.VEC_CORE_NUM = properties["num_vectorcore"]
        self.AI_CORE_NUM = properties["num_aicore"]

    def forward(self, boxes, scores, max_output_size, iou_threshold,
                scores_threshold, pad_to_max_output_size=False):
        # 1. 过滤 + 升精度
        boxes_f32 = boxes.float()
        scores_f32 = scores.float()
        score_mask = scores_f32 > scores_threshold
        filtered_boxes = boxes_f32[score_mask]
        filtered_scores = scores_f32[score_mask]
        original_indices = torch.where(score_mask)[0]

        # 2. 定长输出缓冲 + device-side counter (L1.8)
        selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)
        N = filtered_boxes.shape[0]
        if N == 0:
            return selected_indices, torch.tensor(0, dtype=torch.int32, device=boxes.device)

        # 3. stable argsort 降序
        sorted_indices = torch.argsort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_indices].contiguous()
        sorted_original_indices = original_indices[sorted_indices].to(torch.int32).contiguous()

        # 4. 预计算 areas
        areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]) * \
                (sorted_boxes[:, 3] - sorted_boxes[:, 1])

        # 5. suppressed + num_sel_tensor (device 常驻)
        suppressed = torch.zeros(N, dtype=torch.int8, device=boxes.device)
        num_sel_tensor = torch.zeros(1, dtype=torch.int32, device=boxes.device)

        # 6. chunked loop (L1.9)
        BLOCK_N = 256
        CHUNK = 256
        i = 0
        while i < N:
            end = min(i + CHUNK, N)
            for j in range(i, end):
                remaining = N - j - 1
                num_blocks = (remaining + BLOCK_N - 1) // BLOCK_N if remaining > 0 else 1
                grid_size = min(max(num_blocks, 1), self.VEC_CORE_NUM)    # L1.6
                nms_fused_kernel[(grid_size,)](
                    sorted_boxes, areas, suppressed,
                    selected_indices, num_sel_tensor,
                    sorted_original_indices,
                    float(iou_threshold), j, N, max_output_size,
                    BLOCK_N=BLOCK_N, multibuffer=True,
                )
            # CHUNK 边界同步
            num_selected = num_sel_tensor.item()
            if num_selected >= max_output_size:
                break
            i = end

        num_selected = num_sel_tensor.item()
        return selected_indices, torch.tensor(num_selected, dtype=torch.int32, device=boxes.device)
```

#### L2.2 nms_fused_kernel 骨架（v2）

```python
@triton.jit
def nms_fused_kernel(boxes_ptr, areas_ptr, suppressed_ptr,
                     selected_ptr, num_sel_ptr, orig_idx_ptr,
                     iou_threshold, cur_idx, N, max_output_size,
                     BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)

    # 1. kernel 内 suppressed 检查 (L1.8 关键: 消除 host 侧 .item())
    is_supp = tl.load(suppressed_ptr + cur_idx)
    not_supp = is_supp == 0

    # 2. 选中逻辑 fused 进 pid==0 分支 (L1.8)
    if pid == 0:
        count = tl.load(num_sel_ptr)
        should_select = not_supp & (count < max_output_size)
        orig = tl.load(orig_idx_ptr + cur_idx, mask=should_select, other=0)
        tl.store(selected_ptr + count, orig, mask=should_select)
        tl.store(num_sel_ptr, count + 1, mask=should_select)

    # 3. IoU 抑制 (仅当 not_supp 时执行)
    if not_supp:
        remaining = N - cur_idx - 1
        if remaining > 0:
            # 标量加载 cur_box (L1.7)
            cur_x1 = tl.load(boxes_ptr + cur_idx * 4 + 0)
            cur_y1 = tl.load(boxes_ptr + cur_idx * 4 + 1)
            cur_x2 = tl.load(boxes_ptr + cur_idx * 4 + 2)
            cur_y2 = tl.load(boxes_ptr + cur_idx * 4 + 3)
            cur_area = tl.load(areas_ptr + cur_idx)

            num_blocks = tl.cdiv(remaining, BLOCK_N)    # L1.2
            num_cores = tl.num_programs(0)

            # 交织循环 (L1.1)
            for block_idx in range(pid, num_blocks, num_cores):
                j_start = cur_idx + 1 + block_idx * BLOCK_N
                offs = j_start + tl.arange(0, BLOCK_N)
                mask = offs < N

                # 候选向量加载
                cand_x1 = tl.load(boxes_ptr + offs * 4 + 0, mask=mask, other=0.0)
                # ... cand_y1/y2/x2/area 同理 ...

                # IoU 计算 (cur_box 标量 broadcast)
                x1_inter = tl.maximum(cur_x1, cand_x1)
                # ... inter/union/iou 同理 ...
                iou = inter / tl.maximum(union, 1e-6)

                # OR 合并 (L1.4)
                suppress_mask = (iou >= iou_threshold) & mask
                existing = tl.load(suppressed_ptr + offs, mask=mask, other=0)
                new_supp = existing | suppress_mask.to(tl.int8)
                tl.store(suppressed_ptr + offs, new_supp, mask=mask)
```

### §2.3 Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

#### L3.1 Fused kernel pid==0 分支消除 host 侧写入（v2 核心）

**问题**: v1 在 host 侧逐轮 `selected_indices[num_selected] = sorted_original_indices[i]` + `num_selected += 1`，需要 `.item()` 同步读 suppressed[i] 和 num_selected，N 轮累计同步开销巨大。

**解决**: 把选中逻辑放进 kernel 的 `if pid == 0` 分支。`num_sel_tensor` 常驻 device，kernel 内 `tl.load(num_sel_ptr)` 读当前计数，`tl.store(num_sel_ptr, count+1, mask=should_select)` 自增。host 侧完全不需要在 chunk 内读计数。

```python
if pid == 0:
    count = tl.load(num_sel_ptr)
    should_select = not_supp & (count < max_output_size)
    tl.store(selected_ptr + count, orig, mask=should_select)
    tl.store(num_sel_ptr, count + 1, mask=should_select)
```

**关键点**: `pid==0` 保证只有 1 个 core 写入，避免多 core 竞争；`mask=should_select` 保证未选中或超 max_output_size 时不写入。

**可替代方向**: 若需进一步减少同步，可让多个 core 分担选中逻辑（如按 cur_idx 分区），但需处理写入竞争（原子操作或分片写入）。

#### L3.2 Device-side counter 消除逐轮 .item() 同步（v2 核心）

**问题**: v1 的 `num_selected` 是 host 侧 Python int，每轮 `.item()` 同步；`suppressed[i].item()` 也是每轮同步。

**解决**:
- `num_sel_tensor = torch.zeros(1, dtype=torch.int32, device=boxes.device)` 常驻 device
- kernel 内 `tl.store(num_sel_ptr, count+1)` 自增
- host 侧只在 CHUNK 边界 `num_sel_tensor.item()` 同步一次
- `suppressed[i]` 的检查推进 kernel 内：`is_supp = tl.load(suppressed_ptr + cur_idx)`

**收益**: 同步次数从 N 次降到 N/256 次（v1 → v2 的核心提升）。

**可替代方向**: 若 CHUNK 内的提前终止很重要，可用 event/fence 机制让 host 异步查询 device 状态，但 Triton 当前不支持。

#### L3.3 Chunked loop 的 CHUNK 选择

**问题**: CHUNK 过大则 max_output_size 达标后不能立即 break，浪费计算；CHUNK 过小则同步开销仍高。

**解决**: `CHUNK = 256` 是经验值。
- 同步开销: N/256 次 `.item()`，相比 v1 的 N 次减少 256x
- 提前终止粒度: 最多多算 255 轮，对大 N 可接受

**可替代方向**:
- 若 max_output_size 通常远小于 N（如 N=32768, max=1000），可动态调整 CHUNK（前期大、后期小）
- 若 N 固定已知，可离线调优 CHUNK

#### L3.4 int8 suppressed + OR 合并替代 int32 + 原子写（沿用 v1）

**问题**: suppressed 数组每轮被多 core 并发读写，需要某种并发安全机制。

**解决**: int8 + OR 操作。OR 是幂等的（`x | 1 = 1`，`x | 0 = x`），即使两个 core 同时写同一位置，结果也是正确的；无需原子操作。int8 比 int32 节省 4x 显存和访存带宽。

```python
existing = tl.load(suppressed_ptr + offs, mask=mask, other=0)
new_supp = existing | suppress_mask.to(tl.int8)
tl.store(suppressed_ptr + offs, new_supp, mask=mask)
```

**可替代方向**: 若需更细粒度状态（如 "candidate" / "selected" / "suppressed" 三态），可用 int8 的不同 bit 位编码。

#### L3.5 cur_box 标量加载消除重复访存（沿用 v1）

**问题**: 每轮 cur_box 是固定值，若每个候选 box 都 `tl.load(cur_box)` 会重复访存 BLOCK_N 次。

**解决**: cur_box 的 4 个坐标在 kernel 入口标量加载 1 次，与候选向量 `[BLOCK_N]` broadcast 计算。Triton 的标量-向量运算会自动 broadcast。

**可替代方向**: 若 cur_box 也按 BLOCK 分块（如多轮合并 kernel），需重新设计 cur_box 的加载策略。

#### L3.6 stable argsort 保证排序稳定性（沿用 v1）

**问题**: scores 相同时，argsort 的顺序不确定，导致 verify 失败。

**解决**: `torch.argsort(filtered_scores, descending=True, stable=True)`。stable=True 保证相同 score 按原始索引顺序排列，与 PyTorch golden 行为一致。

**可替代方向**: 若性能瓶颈在 argsort（aclnnSort 4.58ms），可考虑 NPU 专用 sort 算子或分桶排序。

#### L3.7 multibuffer=True 启用多缓冲（沿用 v1）

**解决**: kernel launch 时传 `multibuffer=True`，让 Ascend runtime 重叠 host 侧 launch 开销与 device 侧 kernel 执行。

**可替代方向**: 若 kernel 极短（小 N 末尾轮次），multibuffer 的缓冲区管理开销可能超过收益，可动态决定是否启用。

#### L3.8 BLOCK_N = 256 的选择依据（沿用 v1）

- **256**: 单块覆盖 256 个候选 box，平衡 SRAM 占用与并行度
- **128**: 块更小，并行度更高，但每块的 cur_box broadcast 开销摊薄变差
- **512/1024**: 块更大，SRAM 压力增大，小 N 时浪费

**可替代方向**: 若 N 固定且已知（如检测框固定 1000），可调大 BLOCK_N 减少块数。

### §2.4 NMS 性能基准（几何平均 1.0187x，突破 1.0x 归档阈值）

| Shape 区间 | 典型加速比 | 说明 |
|-----------|-----------|------|
| 小 shape [50-512, 4] | 0.88x ~ 1.21x | chunked loop 同步开销在小 N 上仍可见 |
| 中 shape [1024-8192, 4] | 0.81x ~ 1.14x | 基本持平，部分 case profiler fail |
| 大 shape [10000-32768, 4] | 1.00x ~ 1.55x | 大 N chunked loop 收益最大，Triton 反超 |
| 全量 31 cases | 1.0187x | 几何平均，达标 1.0x 归档阈值 |

### implementation 侧 operator 耗时拆解（v2）

| operator | latency (ms) | 占比 | vs v1 | 说明 |
|----------|-------------|------|-------|------|
| `nms_fused_kernel` | 4.38 | 4% | 13.21→4.38 (3x) | fused + device counter 收益 |
| `aclnnNonzero*` | 81.78 | 80%+ | 基本持平 | host 侧过滤，主要瓶颈 |
| `aclnnIndex*` | 21.40 | 21% | 基本持平 | filtered_boxes[sorted_indices] |
| `aclnnSort` | 6.10 | 6% | 基本持平 | argsort |

**关键结论**:
1. v2 把 kernel 本体从 13% 降到 4%，host 侧 `nonzero + index + sort` 升至 80%+，是下一版优化方向
2. fused kernel + device counter + chunked loop 是顺序算法在 Triton 上消除同步开销的通用模式
3. v2 落地了 v1 报告中标注的 "batched kernel 消除 .item() 同步" 未来方向

---
