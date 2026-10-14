---
type: Concept
title: "Ascend910B (DaVinci V220) — Target Platform"
description: "Ascend910B（DaVinci V220 单 die，A2）目标平台规格：SIMD-only 编程模型、B1–B4 SOC 型号、24/20 AICore、UB 192KB/AIV、L0C 128KB、CMake 构建参数与待硬件实测确认项（前瞻性规格，尚未上机验证）。"
tags: [task.lookup, topic.architecture, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/hardware/target-ascend910b.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-16T07:31:58Z"
platforms: [a2]
aliases: [hw/ascend910b]
---

# Ascend910B (DaVinci V220) — Target Platform

> 原 OKF v1 卡号：**hw/ascend910b**（原 OKF v1 kind: guide；confidence: single_run）。

> 本卡规格未经真机核验（forward-looking spec），保持 draft 待硬件验证。

> **NOT YET VERIFIED ON HARDWARE.** This file is a forward-looking spec
> derived from CANN source + public references. When an A2 server becomes
> available, run smoke tests (build + `acl.get_soc_name()` + a SIMD `Add`
> kernel) and update `verified: true` + `date_verified`.
>
> A2 = single die DaVinci V220. CANN treats it as `ascend910b_list` family,
> compiles with `BUILD_MODE=c220`, arch dir `arch22`. Same arch as A3 but
> single-die (so half the AIV count of A3).
>
> **SIMD-only.** Same SIMT-restriction list as A3 — see [target_ascend910c](target_ascend910c.md).

## SOC version reference

| Variant | SOC string | Marketed as |
|---------|-----------|-------------|
| B1 | `Ascend910B1` | 910B1 |
| B2 | `Ascend910B2` / `Ascend910B2C` | 910B2 |
| B3 | `Ascend910B3` | 910B3 |
| B4 | `Ascend910B4` / `Ascend910B4-1` | 910B4 |

`acl.get_soc_name()` returns the active SOC string at runtime — query before
choosing build flags. Most public A2 deployments are B3.

## Compute (per chip)

| Parameter | B2 | B3/B4 | Source |
|-----------|----|----|--------|
| AICore total | 24 | 20 | HAMi guide, public refs |
| AIV per AICore | 2 | 2 | CANN `GetAicAivTaskRation()` |
| **AIV total** | **48** | **40** | derived |
| AIC total (Cube) | 24 | 20 | derived |
| **Programming model** | SIMD only | SIMD only | DavidV100 manual §25.1.1 |
| `threadIdx` / `blockDim` SIMT | NOT available | NOT available | arch22 |
| `Simt::` namespace | NOT available | NOT available | arch22 |

What works / fails on A2 = same as A3 (see [target_ascend910c](target_ascend910c.md)).

## Memory (per chip)

| Parameter | B2 | B3/B4 | Source |
|-----------|----|----|--------|
| HBM type | HBM2e | HBM3e (B3/B4) | public refs |
| **HBM capacity** | 64 GB | 32 GB (B4) / 64 GB | HAMi guide |
| **HBM bandwidth** | ~400 GB/s | ~800 GB/s | public refs |
| **UB / AIV** | **192 KB** | **192 KB** | CANN `hardware.h` |
| L1 / AIC | 512 KB | 512 KB | CANN `arch.h` |
| L0A / AIC | 64 KB | 64 KB | CANN `arch.h` |
| L0B / AIC | 64 KB | 64 KB | CANN `arch.h` |
| **L0C / AIC** | **128 KB** | **128 KB** | CANN `arch.h` (`__NPU_ARCH__==3510`→256, else→128 — **was "3101", corrected 2026-06-18 vs arch.h:29**) |
| **BT (bias table) / AIC** | **1 KB (1024)** | **1 KB (1024)** | cannbot npu-arch §2.3 DAV_2201 `bt_size` (A5/950PR=4KB — differs) |
| **Structured sparsity 4:2** | **supported** (`sparsity=1`) | **supported** | cannbot npu-arch §2.3 DAV_2201 (A5/950PR `sparsity=0` — NOT supported; don't assume 4:2 on A5) |
| L2 cache | TBD | TBD | needs probe |

## Atomics

Same TBD list as A3 — see [target_ascend910c](target_ascend910c.md) § Atomics. A2 and A3 share the
V220 architecture, so atomic capability should be identical, but verify
on first A2 port.

## Cross-platform porting summary

| What | A5 (950PR) | A3 (910C) | A2 (910B) |
|------|-----------|-----------|-----------|
| Programming model | SIMT + SIMD | SIMD only | SIMD only |
| `__NPU_ARCH__` | 3510 | 2201 | 2201 |
| arch dir | arch35 | arch22 | arch22 |
| AIV count | 56 | 80~96 | 40~48 |
| UB per AIV | 256 KB | 192 KB | 192 KB |
| L0C per AIC | 256 KB | 128 KB | 128 KB |
| HBM bandwidth | 1.6 TB/s | ~1.6 TB/s (dual-die) | 0.4–0.8 TB/s |

## Build flags (CMake)

```cmake
set(SOC_VERSION "Ascend910B3" CACHE STRING "...")  # or B1/B2/B4 to match your chip
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann" CACHE STRING "...")
include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
```

## Open verification tasks (when A2 server becomes available)

1. Confirm SOC string via `acl.get_soc_name()` and update this doc's
   `soc_version` frontmatter
2. Confirm AIV count via `GetCoreNumAiv()` (40, 48, or other)
3. Smoke test: build the SIMD `Add` test kernel with the right SOC and run it
4. Probe: FP16/BF16 atomicAdd support (same probe as A3)
5. Probe: HBM bandwidth measurement (compare against the public 400 GB/s figure)
6. Update `verified: true` and `date_verified` in frontmatter

## References

- 源仓 `docs/design/PLUGIN_PARADIGM_NOTES.md#ascend-chip-comparison`（插件仓内部文档，未迁移） — A2/A3/A5 comparison
- CANN source (`code_channel_infer.py`, `arch.h`, `hardware.h`)
- HAMi Ascend 910B virtualization guide (public)
- HiAscend AscendC manual 9.0.0 (Chinese version)
- Hi1980B&C V100 Davinci Cloud 芯片 用户指南（本地 docx 文档，未迁移） —
  primary hardware doc covering both Hi1980B (A2) and Hi1980C (A3)

## 相关卡片

- [Ascend910C (DaVinci V220, dual-die) — Target Platform](target_ascend910c.md)：A3 双 die 同族平台（SIMT 限制清单与 Atomics 待测项见该卡）。
- [Ascend950PR (David V100) -- Target Platform](target_ascend950pr.md)：A5 目标平台规格。
- [Ascend 910B2/910B3 硬件规格](ascend_910b.md)：同芯片的 Triton 视角硬件速查（Grid 语义、编译选项），与本卡来源不同、互为补充。
