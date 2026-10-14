---
type: Guide
title: GPU Triton → Ascend 910_95 迁移工作流（最小变更）
description: GPU Triton kernel 迁移到 Ascend 910_95 NPU 的三阶段工作流：Phase 1 最小变更跑通、Phase
  2 单点迭代优化、Phase 3 910_95 专属优化，附精度验证模板。
tags: [task.migration, topic.compilation, topic.architecture, topic.quality.compatibility.version_migration]
status: stable
sources:
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/index.md
  role: supporting
- resource: https://gitcode.com/Ascend/triton-ascend/blob/c22984baa55e1506eadd7c72955d9bf669b1c959/docs/zh/triton_api/Programming_Model/program_id.md
  role: supporting
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: ['950']
---

# 功能说明

GPU Triton kernel → Ascend 910_95 NPU 的迁移与迭代优化工作流，核心原则：**最小变更、每步验证、一轮只改一处**。目标硬件 `dav-c310`（Reg-based，1 Cube + 2 Vector）；UB/L0C/L1 容量与对齐约束见 [NPU 硬件总览](../../apis/ir/architecture/npu_hardware_overview.md)。

## Phase 1：跑通（仅最小变更）

按序只做 5 处修改，本阶段**不加任何优化**（即使迁移向导里有完整示例也不一次套用）：

1. **设备替换**：`cuda` → `npu`，在任何 triton import 之前 `import torch_npu`。
2. **删除 GPU 专属参数**：`num_warps`、`num_stages`、`cache_modifier`、`eviction_policy`。
3. **数据类型替换**：fp64 → fp32，uint 系列 → int 系列。
4. **Grid 调整**：对齐物理核数，优先 1D。
5. **加精度验证**：`if __name__ == "__main__"` 块对比 Triton 输出与 PyTorch 参考（模板见下）。

随后运行数据采集（正确性验证 + IR 提取 + 基线性能）。此阶段精度失败 = 迁移本身有 bug，先修再继续。

## Phase 2：迭代优化（一轮一处）

1. 分析 IR 与 msprof 数据，定位最大瓶颈（分析方法见 [IR 分析工作流](ir_analysis_workflow.md)）。
2. 从迁移向导文档中选**一个**针对该瓶颈的优化。
3. 应用该优化。
4. 重新验证精度 + 收集 IR + profile。
5. 精度失败或性能回退 → 回退该改动，分析后可修正再验，否则换下一个候选。
6. 有提升 → 处理下一个瓶颈。

停止条件：关键指标全部健康 / 无优化再有提升 / 修正尝试后精度仍无法保持。

## Phase 3：910_95 专属优化

Phase 2 收敛后，再尝试 910_95 独占优化（L0C→UB 直通、FP8、SIMT 模式等），同样遵守一轮一处与失败分析规则。

## 精度验证模板

每个迁移脚本必须带 `if __name__ == "__main__"` 验证块：

```python
if __name__ == "__main__":
    dtype = torch.float16
    x = torch.randn((1024, 1024), dtype=dtype, device="npu").requires_grad_()

    # --- Triton kernel ---
    y = YourTritonFunction.apply(x, ...)
    dy = torch.randn_like(y)
    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None

    # --- PyTorch reference ---
    ref = your_pytorch_reference(x, ...)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None

    # --- Compare ---
    atol, rtol = 1e-3, 1e-3
    if torch.allclose(y, ref, atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol):
        print("✅ [Bwd]Triton and Torch match")
    else:
        print("❌ [Bwd]Triton and Torch differ")
```

关键规则：参考实现必须是纯 PyTorch（无 Triton 调用）；`.backward()` 后立即 clone 梯度并把 `x.grad` 置 `None`；fp16/bf16 用 `atol=rtol=1e-3`，fp32 用 `1e-4`；`✅`/`❌` 输出格式是给数据采集器解析的，不可改。

## 知识查阅优先级

1. **迁移向导**（问题导向、代码驱动）：已收录为 [gpu_migration_overview](../workflows/gpu_migration_overview.md)、[migration_code_patterns](../workflows/migration_code_patterns.md) 及 [flash_attention_migration](../migration/flash_attention_migration.md) 等 GPU→NPU 迁移卡（workflows/ 与 ops/ 分区）。
2. **Triton-Ascend 开发文档**（API 参数、编译流程细节）：已收录为 [../api/](../../apis/index.md)、[../techniques/](../../optimizations/techniques/index.md)、[../workflows/](../workflows/index.md) 等分区卡片，以及本分区 [compilation_pipeline/](../../apis/ir/compilation_pipeline/index.md)。
3. **AscendNPU-IR 编译器文档**（IR 语义、pass 定义）：已收录为本分区 [architecture/](../../apis/ir/architecture/index.md)、[hivm/](../../apis/ir/hivm/index.md) 等卡片，分析提取的 IR 时查阅。