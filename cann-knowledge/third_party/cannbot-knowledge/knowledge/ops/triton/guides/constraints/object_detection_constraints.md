---
type: Guide
title: object_detection 类算子设计约束
description: object_detection 类算子的适用范围、类别专属设计约束与常见陷阱；跨类别通用条款见 global_constraints.md。
tags: [task.design, topic.compute.composite, topic.tiling, topic.compute.ordering, topic.implementation.pipeline]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
created_at: '2026-08-12T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2]
---

# object_detection 类算子设计约束

> 跨类别通用约束见 [global_constraints.md](global_constraints.md)（`GC-1`~`GC-15`），
> 本卡只列本类别特有条款。

## 目标检测类算子优化经验

本文档合并了目标检测中两类核心算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复）
- **§2 NMS**（Non-Maximum Suppression，顺序算法 + 每轮 IoU 抑制）
- **§3 IOU**（Intersection over Union，成对计算）
- **§4 各算子常见陷阱**

---

## 本类别的算子级实现手册

> 本卡只列类别级共性约束。以下单算子手册含该算子专属的 Layer 1 设计约束、
> 算法骨架与性能基准，**命中对应算子时必须一并加载**。

- [`operators/object_detection__iou.md`](../operator_playbooks/object_detection__iou.md) — IOU 算子（Intersection over Union）
- [`operators/object_detection__nms.md`](../operator_playbooks/object_detection__nms.md) — NMS 算子（Non-Maximum Suppression）

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| NMS | `sort-topk` + `indexing-gather` 复合 | 顺序算法：过滤 → argsort → 顺序遍历 → 每轮成对 IoU 抑制 | fused kernel + device-side counter + chunked loop 消除 .item() 同步 |
| IOU | `elementwise` / `pairwise-compute` | 成对计算：输入两个 [N,4] 和 [M,4] 的 bbox，输出为 [M,N] 的 IoU/IoF 矩阵 | 2D Tiling + Broadcast 替代离散访存 |

> ⚠️ **关键区分**：NMS 属 **顺序算法类**（核心瓶颈是顺序 IoU 抑制循环中的 host-device 同步开销），IOU 属 **成对计算类**（核心瓶颈是离散逐元素访存无法利用 NPU 向量单元）。两类优化哲学相反，生成时**禁止混用经验**：
> - 生成 NMS 时，**不要**套用 IOU 的纯 2D Tiling 思路（NMS 每轮只对单个 cur_box 做抑制）
> - 生成 IOU 时，**不要**套用 NMS 的 chunked loop / device counter（IOU 无顺序依赖，单 kernel 即可完成）
> - **通用复用点**：NMS 的 IoU 抑制阶段可直接复用 IOU 的 1D Grid + 交织循环骨架（见 §2 L1.1 与 §3 L1.2）

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 6 条约束在两个算子中均适用，各算子章节不再重复。

### D1 动态读取 Vector/Cube Core 数量，禁止硬编码
- **必须**动态读取实际核数，禁止硬编码 `num_cores=8` 或 `num_cores=48`。
- **正确做法**（一次拿 vector + cube，权威值，无需设备 init）：
  ```python
  import torch_npu
  import triton.runtime.driver as driver

  device = torch_npu.npu.current_device()
  properties = driver.active.utils.get_device_properties(device)
  vectorcore_num = properties["num_vectorcore"]   # elementwise / reduction 用它做 grid 钳制
  aicore_num = properties["num_aicore"]           # cube / matmul 用它
  ```
- **已弃用**：旧式 `npu_config` 取值方式（仅 vector、硬编码 40 不准、设备未 init 会抛 `RuntimeError`）。
- **Why:** 硬编码会浪费多核并行能力；不同 NPU 型号核数不同（ascend910b1 = 40）。

### D2 禁止在 kernel 内使用 triton.cdiv，必须用 tl.cdiv
- **禁止**在 `@triton.jit` kernel 内使用 `triton.cdiv`，**必须**用 `tl.cdiv`。
- **Why:** `triton.cdiv` 在 JIT 函数中会触发 `ValueError: Did you forget to add @triton.jit ?`。
- **How to apply:** kernel 内 `num_blocks = tl.cdiv(remaining, BLOCK_N)`；host 侧用 `triton.cdiv` 或 `(n + BLOCK - 1) // BLOCK`。

### D3 NPU 原生算子与标准算法不兼容时必须替换 Golden
- **必须**当 task 文件存在多个 `Model` 类（如 `torch_npu.npu_nms_v4` / `torch_npu.npu_iou` 与纯 PyTorch 数学公式实现）时，以**纯 PyTorch 数学公式实现**作为 verify/benchmark 的 golden。
- **禁止**试图逆向工程 NPU 原生算子的内部排序/抑制/计算规则。
- **Why:** NPU 原生算子（`npu_nms_v4`、`npu_iou`）内部实现不透明（fp16 中间计算、不同 eps 值、变长输出等尝试均失败），且与标准算法存在系统性数值差异（约 1e-3 量级，差异模式：NPU 结果总是略小于标准算法）。
- **How to apply:** 将 task 文件的 `Model.forward` 改为注释中的数学公式实现，Triton 实现按标准算法编写即可天然匹配。

### D4 Grid 总数必须动态限制为实际核数
- **必须**`grid_size = min(max(num_blocks, 1), VEC_CORE_NUM)`，禁止直接 `grid = (num_blocks,)`。
- **Why:** 当 `num_blocks < VEC_CORE_NUM` 时（小 N 或末尾轮次），多余 core 会空跑；当 `num_blocks` 极大时可能超 grid 上限。
- **How to apply:**
  ```python
  grid_size = min(max(num_blocks, 1), VEC_CORE_NUM)
  kernel[(grid_size,)](..., multibuffer=True)
  ```

### D5 必须使用 1D Grid + 交织循环模式
- **必须**使用 `pid = tl.program_id(0)` 配合 `for block_idx in range(pid, num_blocks, num_cores)` 的交织循环模式。
- **Why:** 输出元素数不确定（可能大于或小于核数），1D 交织循环天然负载均衡；无需额外处理 `num_blocks` 随 `N - cur_idx - 1` 递减的情况。
- **How to apply:**
  ```python
  pid = tl.program_id(0)
  num_blocks = tl.cdiv(remaining, BLOCK_N)
  num_cores = tl.num_programs(0)
  for block_idx in range(pid, num_blocks, num_cores):
      ...
  ```

### D6 所有 kernel 调用加 multibuffer=True
- **必须**所有 kernel 调用附加 `multibuffer=True`。
- **Why:** IoU/抑制 kernel 为访存密集型或计算密集型，`multibuffer=True` 让 Ascend runtime 重叠 host 侧 launch 开销与 device 侧 kernel 执行。
- **How to apply:** 见各算子 L2 骨架中 kernel 调用末尾的 `multibuffer=True`。

---

## §4 常见陷阱与避免方法

### §4.1 NMS 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 试图全 kernel 化 NMS 主循环 | 把整个 N 轮 Python 循环搬进单个 Triton kernel，期望消除所有同步 | NMS 是顺序算法，每轮依赖上一轮抑制状态，kernel 内无法跨轮传递动态状态；必须保留 Python 主循环 (L1.5)，用 chunked loop (L1.9) 折中 |
| 用 npu_nms_v4 作为 golden | `npu_nms_v4` 返回变长输出且内部规则不透明，verify 反复失败 | 替换为纯 PyTorch 数学公式实现作为 golden (L1.3 / D3) |
| 用 int32 表示 suppressed | int32 浪费 4x 显存，且 .item() 同步读 int32 比 int8 慢 | int8 + OR 合并 (L1.4 / L3.4) |
| kernel 内用 triton.cdiv | `triton.cdiv` 在 JIT 函数中触发 `ValueError: Did you forget to add @triton.jit ?` | 用 `tl.cdiv` (L1.2 / D2) |
| grid_size 不限制为 VEC_CORE_NUM | 小 N 或末尾轮次 num_blocks < VEC_CORE_NUM 时，多余 core 空跑 | `grid_size = min(max(num_blocks, 1), VEC_CORE_NUM)` (L1.6 / D4) |
| 误把 NMS 当成 sort-topk 类算子优化 | 见框架 6 类分类，误以为 NMS 属 sort-topk，尝试并行排序优化 | NMS 的核心瓶颈是顺序 IoU 抑制循环，不是排序；排序只是预处理。优化重点应在 IoU kernel + 减少同步，复用 §3 IOU 经验而非 sort-topk 经验 |
| cur_box 每个候选重复加载 | 在 block 内对每个候选 box 都 `tl.load(cur_box)`，浪费访存带宽 | cur_box 标量加载 1 次 + broadcast (L1.7 / L3.5) |
| profiler 失败误判为代码错误 | case 7/8/9/23/25/26 报 `RuntimeError: 无法从 profiler 提取有效时延数据`，误以为代码 bug | 这是 profiling 基础设施问题（并发 profiling 或 SQLite TASK 表缺失），verify 阶段全部通过；重新单独跑这些 case 通常能拿到正常数据 |
| host 侧逐轮 .item() 同步（v1 的教训） | v1 每轮 `suppressed[i].item()` + `num_selected` 读写，N 轮同步开销导致 geomean 卡在 0.99x | v2 用 fused kernel (L1.8) + device-side counter (L3.2) + chunked loop (L1.9) 把同步次数从 N 降到 N/256，突破 1.0x |
| CHUNK 过大导致提前终止失效 | CHUNK 设置过大（如 CHUNK=N），max_output_size 达标后不能立即 break，浪费计算 | CHUNK=256 是经验值 (L3.3)；若 max_output_size 通常较小，可在 CHUNK 边界检查 `num_selected >= max_output_size` 提前退出 while 循环 |

### §4.2 IOU 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| NPU 原生算子与标准算法的系统性数值差异 | `torch_npu.npu_iou` 与标准 PyTorch 数学公式实现存在约 1e-3 量级的系统性差异（NPU 结果总是略小于标准算法，如 0.1418 vs 0.1429） | 不要试图逆向工程 NPU 原生算子的行为；将 task 文件的 `Model` 替换为注释中提供的 PyTorch 数学公式实现 (L1.4 / D3) |
| triton.cdiv 在 kernel 中使用 | 在 `@triton.jit` kernel 中使用 `triton.cdiv` 导致编译错误 | kernel 内使用 `tl.cdiv`，host 侧使用 `triton.cdiv` (L1.3 / D2) |
| 离散逐元素访存模式 | 每个线程计算一个输出元素，离散加载 8 个标量，无法利用 NPU 向量单元 | 采用 2D Tiling + Broadcast 模式，向量加载块数据后通过 broadcast 并行计算 (L1.5 / L3.1) |
| 2D Tiling 的 broadcast 兼容性 | 尝试 2D tiling 时可能遇到 `Cannot make_shape_compatible` 错误 | 确保 broadcast 维度匹配（`[None, :]` 和 `[:, None]` 的组合），并正确计算 mask |
| Grid 过大导致调度开销 | grid_size 超过实际核数时，核间调度开销显著 | 限制 grid_size 为 `min(num_blocks, vectorcore_num)`，并启用 `multibuffer=True` (L1.6 / D4) |
