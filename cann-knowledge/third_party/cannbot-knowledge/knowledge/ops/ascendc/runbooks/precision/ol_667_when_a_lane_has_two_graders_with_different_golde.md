---
type: Runbook
title: "When a lane has TWO graders with DIFFERENT goldens, measure golden-vs-golden distance per failing case BEFORE kernel iteration — a both-pass intersection smaller than the case set is a structural block requiring an owner decision, not a kernel bug"
description: "一条 lane 上两个 golden 不同的评分器可能给出矛盾接受域，没有 kernel 能同时满足——判别法是把 golden 自己当候选跑一遍（native-as-candidate）"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-667"]
---

# When a lane has TWO graders with DIFFERENT goldens, measure golden-vs-golden distance per failing case BEFORE kernel iteration — a both-pass intersection smaller than the case set is a structural block requiring an owner decision, not a kernel bug

`applies_to: soc=all (observed on Ascend310P3); cann=all (observed 9.1.0); bisheng=n/a; op_class=all (any op graded by ≥2 independent acceptance scripts against different goldens); workflow=verification`
`verified_on: soc=Ascend310P3; cann=9.1.0 (instance_norm_grad 310p, 2026-09-02, benchmark-acceptance lane — measured both-pass intersection 7/41)`

**Principle**: two graders whose goldens differ (e.g. a dtype-native CPU golden on one lane vs an fp64-truth golden on another) impose CONTRADICTORY acceptance regions wherever the golden-vs-golden distance exceeds the sum of their tolerances. No kernel output can satisfy both there; iterating the kernel toward one grader strictly degrades it under the other. The cheap decisive probe is **native-as-candidate**: run the reference golden ITSELF through the other grader's comparator. If the golden fails, the case set is structurally ungradeable and the correct escalation is an owner-level decision (input distribution, single grading standard, or documented-evidence acceptance) — never more kernel iterations.

**Trigger classifier — verify the TRUTH SOURCE before declaring a structural grader conflict (2nd-instance correction, 2026-09-05)**: the native-as-candidate probe is only decisive when the "vendor-self" leg runs the REAL golden-generating program. A worker-modified `model.py` (not byte-identical to the sha-pinned vendor def) silently re-targets both the kernel and the probe's vendor leg — an apparent golden-vs-kernel contradiction can be an artifact of the modified truth source, not a structural conflict. Check `sha256(model.py) == dataset_provenance.op_def_sha256` (or an explicit diff-justification artifact) BEFORE escalating a both-pass-intersection block to an owner decision; the corrected 2nd-instance evidence below is the catch instance.

**Concrete anchor**: instance_norm_grad 310p — the owner lane grades vs a CPU dtype-native golden (PB-68 class) while the O5 two-tier grader grades vs fp64 CPU truth (normal-region mismatch = instant FAIL). Benchmark inputs carry raw `randn` mean/std (std min ≈ 5e-3), so `pow(std,-3)` (gain ~1e6..1e8) amplifies any noise: the fp16 CPU-native golden cascades inf→NaN while fp64-truncated truth has ±inf and no NaN (18/19 fp16 cases ungradeable by both at once), and fp32 row-stat accumulation-order noise (~1.7e-6 rel — carried by ANY faithful fp32 impl, torch CPU included) amplifies to element rel errors up to 9.9e-2. Measured: the CPU-native golden itself fails the fp64-truth comparator on 12/14 failing fp32 cases; both-pass intersection 7/41.

**Evidence**: instance_norm_grad 310p (2026-09-02): `instance_norm_grad/knowledge_update.md` finding 1 + `instance_norm_grad/failures_ledger.md` precision iter 3; per-case dual-grader evidence table recorded with the op's Phase D artifacts (PROGRESS.md).

**Evidence (2nd instance — CORRECTED 2026-09-05 kw-2; the originally-recorded "structural fp16 golden conflict" is WITHDRAWN)**: 28_MultimodalRopePositionComputationWithGridBasedIndexing 310p (benchmark, DEVICE-RUN device 3) initially read as this entry's 2nd instance — 5 pass_b fp16 "conflicts", "vendor-self fails its own golden", shipped 55/60. The post-worker audit (F1 truth-source swap) found the "vendor" was a WORKER-MODIFIED model.py (sha e0d6c71d) that ADDED dtype-specialized branches the vendored def (sha f195e397, pinned in `dataset_provenance.op_def_sha256`) does not have — the true def accumulates unified-fp32 with ONE final round (in-code marker comment "统一 fp32 累加，消除 fp16/bf16 分支差异"). With model.py restored byte-verbatim and the kernel re-derived from the true contract (exact fp32 weights, per-op fp32 chain, single final RNE round): pass_a 50/50 PASS_T1 with fp32/fp16/bf16 bit-exact (ours_mere=0) on all 6 low-dtype cases AND pass_b 60/60 — the fp64-promoted edge golden was NEVER in conflict with the true vendor semantics. `28_MultimodalRopePositionComputationWithGridBasedIndexing/knowledge_update.md` findings 1-2 + `28_MultimodalRopePositionComputationWithGridBasedIndexing/verification.json` truth_source. Contract-provenance rule: OL-691 (revised).

**Other instances (predicted)**: any op running under parallel acceptance harnesses (owner benchmark scripts + O5 two-tier, vendor-native leg + CPU-truth leg, cross-SoC grading) whose goldens are not bit-compatible — pow/reciprocal/division chains with near-zero denominators are the highest-risk class (P-CAT-6). Generalizes OL-89/OL-90's single-grader truth-regime reasoning to the multi-grader case.

**Cross-ref**: PB-68 (dtype-native truth regime), [`OL-89`](ol_89_when_the_active_reference_is_non_deterministic_loo.md) / [`OL-90`](ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md) (Path-A CPU-truth topology), P-P61 (structural-block escalation class), PB-74 (the cast-saturation half of the fp16 divergence), P-CAT-6 (the input-distribution hazard that created the divergence), [`OL-668`](ol_668_a_composite_grader_verdict_ands_independent_sub.md) (reading which sub-check binds). backend=all.
