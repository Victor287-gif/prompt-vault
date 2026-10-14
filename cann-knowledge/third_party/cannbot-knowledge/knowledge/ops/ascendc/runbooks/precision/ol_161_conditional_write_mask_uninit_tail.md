---
type: Runbook
title: "Conditional-write kernels: bit-exact verify must mask the uninit tail the kernel skips"
description: "A kernel writing only the first realLen rows leaves an uninitialised GM tail that differs across runs; slice both reference and output to [:real_len] before bit-exact compare or get false FAILs."
tags: [task.debugging, topic.implementation.masking, topic.memory, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-161-conditional-write-mask-uninit-tail.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-161"]
---

# Conditional-write kernels: bit-exact verify must mask the uninit tail the kernel skips

> 原 OKF v1 卡号：**OL-161**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - capture-replay verify reports large per-element diffs at extreme magnitudes (±1e10, ±1e-30) on a subset of cases, non-deterministic across re-runs, for an op that writes only the first realLen rows

## 现象 / 触发

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=conditional_write, dynamic_length, masked_output`.
Unverified on Ascend910_V220 (A3 captures confirmed to contain garbage tails; the masking rule is
symmetric but only the A5 verification side was exercised).

When an op's contract says "compute only the first `realLen` rows; rows beyond are not part of the
output", the kernel typically **does not write** those tail rows (no `Duplicate(0)`, no
`InitGlobalMemory`). The bytes the verifier sees in the tail are whatever lived in the GM allocation
beforehand — uninitialised memory. Both the A3 capture and the A5 re-run produce DIFFERENT garbage in
the tail (depending on prior buffer state, allocator behavior, parallel kernel residue). Comparing
the full output tensor reports large per-element diffs (e.g. `-1.79e+10`, `-5.98e-34`) that have
NOTHING to do with kernel correctness.

**Detection symptoms**:
- A small subset of cases shows per-element diffs at extreme magnitudes (`±1e10`, `±1e-30`) while the
  rest of the batch is bit-exact.
- The diff is non-deterministic across re-runs of the same case.
- Re-running the SAME case on the SAME hardware twice produces different captured outputs in the
  affected region (smoking gun — kernel output is deterministic, the garbage tail is not).

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

In port-mode / capture-replay verification, the runner MUST read whatever shape-parameter controls
the live region (`group_index`, `seq_lens`, `valid_mask`, `realBatchSize`, `dynamic_length`, ...),
compute the deterministic prefix, and slice BOTH the captured reference and the re-run output to
`[:real_len]` along the relevant axis before bit-exact comparison. Failing to do so produces false
FAILs that are pure verification noise.

**Concrete anchor** (clipped_swiglu `edge_runner.py`):
```python
# A3 capture: group_index limits how many batch rows the kernel touches.
# Rows beyond group_index[0] in 'out_a3' are uninitialised; do NOT compare them.
real_batch = int(case["scalar_inputs"]["group_index"][0])
ref  = a3_outputs[case_idx][:real_batch]   # slice along batch dim
cand = npu_out.cpu()[:real_batch]
assert torch.equal(ref, cand)              # bit-exact on the real region only
```

**Trigger classifier** — apply the slicing rule when ANY of:
- The op signature includes a length / index / mask scalar that bounds the active work
  (`group_index`, `seq_len`, `valid_count`, `actual_seq_qlen`, `dynamic_len`).
- The source kernel has an early-exit `if (idx >= realLen) return;` shape, OR a
  `for (i = 0; i < realLen; ++i)` loop that never touches `[realLen, full_len)`.
- Capture-replay verification is in use (A3-capture vs A5-run, source-capture, etc.).
