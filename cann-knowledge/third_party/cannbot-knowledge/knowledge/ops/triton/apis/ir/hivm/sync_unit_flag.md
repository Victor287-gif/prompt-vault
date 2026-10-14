---
type: API
title: HIVM UnitFlag 同步机制（unit_flag_cond / unit_flag_mode）
description: HIVM UnitFlag 条件同步语义：DISABLED/RESERVED/ENABLED_WITHOUT_UPDATE/ENABLED_WITH_UPDATE
  四模式，嵌入宏操作处理循环"至少执行一次"依赖，UnitFlagEnabledInterface 方法与库调用值编码。
tags: [task.lookup, topic.compilation, topic.implementation.synchronization, topic.programming_model.execution_unit.cube]
status: stable
sources:
- resource: https://gitcode.com/Ascend/AscendNPU-IR/blob/47a0229060e37f92a49cfb82d81c756628e6c7ae/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td
  role: implementation
created_at: '2026-08-14T00:00:00Z'
updated_at: '2026-09-08T09:05:00Z'
platforms: [a2, a3]

---

# HIVM UnitFlag 同步机制

UnitFlag 是嵌入在宏操作（如 [mmadL1](macro_mmad_l1.md)）内的条件同步机制，解决循环"至少执行一次"的依赖问题：Split-K 场景中 L0C 需首迭代清零、后续累加，但若循环可能**零次执行**，set_flag/wait_flag 配对会被打破导致同步错误。`unit_flag_cond` 为 true 时，即使循环未执行同步也能正确完成。

## 四种模式

| 模式 | 值 | 二进制 | 说明 |
|------|---|--------|------|
| DISABLED | 0 | 0b00 | 禁用（默认），常规参与同步分析 |
| RESERVED | 1 | 0b01 | 保留未用，**不传递给库调用** |
| ENABLED_WITHOUT_UPDATE | 2 | 0b10 | 启用但不更新标志计数器——操作可能不执行但需保持同步配对 |
| ENABLED_WITH_UPDATE | 3 | 0b11 | 启用并更新标志计数器——循环至少执行一次且条件为 true |

## 在宏操作中的使用

```mlir
scf.for %arg0 = %start to %end step %step {
  %init_condition = arith.cmpi eq, %arg0, %start : index
  hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, ... : ...)
                  outs(%mc : memref<256x256xf32>)
                  unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
                  unit_flag_cond(%loop_alive_cond)
}
```

- `unit_flag_cond`：Variadic\<I1\>，通常 0 或 1 个条件值，语义为"循环是否至少执行一次"。
- `unit_flag_mode`：UnitFlagArrayAttr，**数组长度应与输出 Tensor 数量一致**。

## UnitFlagEnabledInterface

实现该接口的操作可用以下方法（接口完整清单见 [attr_interfaces.md](attr_interfaces.md)）：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getUnitFlagModes()` / `setUnitFlagModes(...)` | `optional<SmallVector<UnitFlagAttr>>` / void | 读/写 UnitFlag 模式数组 |
| `getUnitFlagConditions()` / `setUnitFlagConditions(...)` | `optional<SmallVector<Value>>` / void | 读/写条件值 |
| `getUnitFlagModeLibValue(PatternRewriter&)` | Value | 传给库调用的编码值（仅 0b00/0b10/0b11） |

接口带验证方法 `verifyUnitFlagEnabledInterface`，检查 `unit_flag_mode` 与 `unit_flag_cond` 声明正确性。

## 与 InjectSync 的关系

InjectSync Pass 开启 `enable-unit-flag=true` 时会自动在宏操作上设置 UnitFlag 模式与条件（见 [sync_sync_injection.md](sync_sync_injection.md)）。若使用 `assume-alive-loops=true`（假设所有循环至少执行一次），则可避免插入 UnitFlag——假设不成立时会导致同步错误，属于调试选项。

相关：[sync_overview.md](sync_overview.md)、[sync_sync_injection.md](sync_sync_injection.md)、[macro_mmad_l1.md](macro_mmad_l1.md)、[attr_enumerations.md](attr_enumerations.md)

## 参考

官方源码参考（链接已锁定 2026-08-17 上游 HEAD；行号锚点继承自源文档，上游演进后可能漂移）：

- [HIVMAttrs.td#L512-L534](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L534)
- [HIVMInterfaces.td#L680-L761](https://gitcode.com/Ascend/AscendNPU-IR/tree/d2a6661365ef65ca738cb66299449b2740c857eb/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L680-L761)
