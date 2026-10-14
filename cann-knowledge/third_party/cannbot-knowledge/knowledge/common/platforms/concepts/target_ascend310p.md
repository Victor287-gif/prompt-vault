---
type: Concept
title: "Ascend310P (dav-m200 / 200x) -- Target Platform"
description: "Ascend310P（dav-m200/200x）目标平台参考：交叉编译限定的证据契约，含 SoC 身份、能力清单与证据 JSON 说明"
tags: [task.lookup, topic.architecture]
status: draft
sources: []
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["hw/ascend310p"]
---

# Ascend310P (dav-m200 / 200x) -- Target Platform

<!-- ASCEND310P_CANONICAL_CLAIMS_BEGIN -->
<!-- Authority: Owner-ref-history-selected pre-candidate hardware blob plus schema semantics; JSON is a checked mirror. -->
| Machine-owned claim | Canonical value |
|---|---|
| `target` | `Ascend310P` |
| `evidence_scope` | `HISTORICAL_CROSS_COMPILE_ONLY` |
| `target_availability` | `UNAVAILABLE` |
| `compile_evidence_device_execution` | `FORBIDDEN` |
| `identity.allowed_direction` | `Ascend310P -> DAV_2002 -> dav-m200` |
| `identity.reverse_is_unique` | `false` |
| `identity.shared_dav_2002_products` | `Ascend310, Ascend310P` |
| `identity.status` | `UNVERIFIED` |
| `device.runtime` | `NOT_RUN` |
| `device.precision` | `NOT_RUN` |
| `device.performance` | `NOT_MEASURED` |
| `device.production_device_validation` | `UNAVAILABLE` |
| `prerequisite.kimi_adapter` | `UNAVAILABLE` |
| `prerequisite.cannbot_lock` | `UNAVAILABLE` |
| `prerequisite.ascend310p_preflight` | `UNAVAILABLE` |
| `prerequisite.public_toolchain_fingerprint` | `UNAVAILABLE` |
| `established_observation_ids` | `[]` |
<!-- ASCEND310P_CANONICAL_CLAIMS_END -->

This page is a provenance-bounded register of earlier source reads and cross-compile trials. Its
machine-readable checked mirror is
`ascend310p_evidence.json`（源仓 kb/okf/runbooks/hardware/ 内，未迁移）. The verifier derives authority from the
unique pre-candidate hardware blob selected from the owner remote-tracking lineage, then applies
schema semantics independent of parent position or candidate manifest bytes and
rejects JSON drift or duplicate keys, unknown machine fields, and contradictory positive claims
in prose, headings, comments, or fenced text.

Nothing has been executed on a 310P device. Runtime and precision are `NOT_RUN`; performance is
`NOT_MEASURED`; production-device validation and a public toolchain fingerprint are `UNAVAILABLE`.
Every observation below is `UNPINNED/UNVERIFIED` and is excluded from established facts.

## Identity contract

The only allowed direction is **Ascend310P -> DAV_2002 -> dav-m200**. That source-derived mapping is
itself `UNVERIFIED` until the source revision and blob are pinned. The reverse direction is
forbidden as a unique identity claim: `DAV_2002`/`200x` does not uniquely identify Ascend310P,
because the observed source snapshot maps both Ascend310 and Ascend310P to `DAV_2002`.

Ascend310B is separate: its documented candidate identity is `DAV_3002` / `dav-m300`, not this
page's `DAV_2002` / `dav-m200`. No measurement or behavior transfers between these targets.

## Current capability state

The following values come from `ascend310p_evidence.json`, not from an installation probe:

| State key | Repository-owned value |
|---|---|
| device runtime | `NOT_RUN` |
| device precision | `NOT_RUN` |
| device performance | `NOT_MEASURED` |
| production-device validation | `UNAVAILABLE` |
| Kimi adapter | `UNAVAILABLE` |
| Cannbot lock | `UNAVAILABLE` |
| Ascend310P preflight | `UNAVAILABLE` |
| public toolchain fingerprint | `UNAVAILABLE` (`PUBLIC_TOOLCHAIN_FINGERPRINT_UNAVAILABLE`) |

## SOC mapping source snapshot

Evidence: id=E310P-SOC-MAP | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=NOT_APPLICABLE | compiler_sha256=NOT_APPLICABLE | container_image_digest=NOT_APPLICABLE | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=NOT_APPLICABLE

An unpinned CANN source snapshot contained a `SocVersion -> NpuArch` table with an
`ASCEND310P -> DAV_2002` row and, in one copy, an `ASCEND310 -> DAV_2002` row. The actual CANN
version, source revision, and blob hash were not retained. This supports only the directional
candidate recorded above; it is not an established product identity.

| Historical source row | Candidate value | Evidence state |
|---|---|---|
| `ASCEND310P` | `DAV_2002` | `E310P-SOC-MAP`; CANN `UNKNOWN`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| `ASCEND310` | `DAV_2002` | `E310P-SOC-MAP`; CANN `UNKNOWN`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| example SOC string | `Ascend310P3` | `E310P-SOC-MAP`; CANN `UNKNOWN`; source revision/blob `UNPINNED`; `UNVERIFIED` |

## Vendor 200x architecture snapshot

Evidence: id=E310P-VENDOR-ARCH | status=UNVERIFIED | cann_version=NOT_APPLICABLE | compiler_version=NOT_APPLICABLE | compiler_sha256=NOT_APPLICABLE | container_image_digest=NOT_APPLICABLE | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=NOT_APPLICABLE

An unpinned vendor-page snapshot described 200x as Cube and Vector units on one core sharing a
Scalar unit, with no cross-core synchronization. The page revision and blob hash are unavailable,
and the page did not uniquely name Ascend310P. Treat the topology as a research lead only.

| Historical field | Snapshot value | Evidence state |
|---|---|---|
| compute topology | Cube and Vector on one core, shared Scalar unit | `E310P-VENDOR-ARCH`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| cross-core synchronization | described as unsupported | `E310P-VENDOR-ARCH`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| intra-core synchronization | EventID plus paired `SetFlag`/`WaitFlag` | `E310P-VENDOR-ARCH`; source revision/blob `UNPINNED`; `UNVERIFIED` |

These source-read candidates must not be converted into runtime behavior claims. In particular,
the presence of Cube and Vector macros does not prove separate AIC/AIV device cores.

## CANN 9.1.0-beta.1 macro-dump observation

Evidence: id=E310P-MACRO-91B1 | status=UNVERIFIED | cann_version=9.1.0-beta.1 | compiler_version=UNKNOWN | compiler_sha256=UNPINNED | container_image_digest=UNPINNED | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=UNPINNED

An earlier `-dM -E` cross-compile note named CANN `9.1.0-beta.1`, but did not retain the compiler
version/hash, immutable image digest, source revision/blob, argv receipt, or macro-dump artifact
hash. Its candidate values remain historical and unverified:

| Token | Historical macro result | Evidence state |
|---|---|---|
| `dav-m200` | `__CCE_AICORE__ 200`, `__DAV_M200__ 1`, `__DAV_CUBE__ 1`, `__DAV_VEC__ 1` | `E310P-MACRO-91B1`; CANN `9.1.0-beta.1`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `dav-m300` | `__CCE_AICORE__ 300`, `__DAV_M300__ 1` | `E310P-MACRO-91B1`; CANN `9.1.0-beta.1`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `dav-m310` | `__CCE_AICORE__ 310`, `__DAV_M310__ 1` | `E310P-MACRO-91B1`; CANN `9.1.0-beta.1`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `dav-c310` | `__CCE_AICORE__ 310`, `__DAV_C310_CUBE__ 1` | `E310P-MACRO-91B1`; CANN `9.1.0-beta.1`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `dav-c220` | `__CCE_AICORE__ 220`, `__DAV_C220_CUBE__ 1` | `E310P-MACRO-91B1`; CANN `9.1.0-beta.1`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |

The same note recorded that `__mix__(1,2)` cross-compiles for `dav-m200` and that the macro expands
to AIC-only while discarding the ratio. This is only a compiler-expansion candidate. It says
nothing about execution on a physical device, portability, correctness, precision, or performance.

## CANN 9.0.0 AIC header-macro observation

Evidence: id=E310P-AIC-MACRO-900 | status=UNVERIFIED | cann_version=9.0.0 | compiler_version=UNKNOWN | compiler_sha256=UNPINNED | container_image_digest=UNPINNED | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=NOT_APPLICABLE

A separate earlier note named CANN `9.0.0` and reported the header expansion
`ASCEND_IS_AIC -> constexpr(g_coreType == AscendC::AIC)` after including `kernel_operator.h`.
The compiler/header provenance was not pinned. This compile-time expansion must not be merged with
the CANN `9.1.0-beta.1` macro dump or treated as device execution evidence.

## bfloat16 source snapshot

Evidence: id=E310P-BF16-SOURCE | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=NOT_APPLICABLE | compiler_sha256=NOT_APPLICABLE | container_image_digest=NOT_APPLICABLE | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=NOT_APPLICABLE

Unpinned CANN source notes recorded a `DAV_2002` guard rejecting `DT_BF16`, an
`ASCEND310P_AICORE_DTYPE_SUPPORT_LIST` containing float and float16, and a negative unit-test case.
The CANN version, repository revision, and source blobs were not retained. These notes are useful
search anchors, but they are not established dtype facts and cannot justify a product behavior
claim.

## bfloat16 trial cross-compile

Evidence: id=E310P-BF16-COMPILE | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=UNKNOWN | compiler_sha256=UNPINNED | container_image_digest=UNPINNED | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=UNPINNED

The trial's CANN version is not recoverable from the retained note, so it is deliberately
`UNKNOWN`, not inferred from either frontmatter version. The historical result was:

| Trial arm | Historical compiler result | Evidence state |
|---|---|---|
| `dav-m200`, fp16 | rc=0 | `E310P-BF16-COMPILE`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `dav-m200`, bf16 | rc=1, unknown `bfloat16_t` | `E310P-BF16-COMPILE`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `dav-NOSUCH` | rc=1, unsupported architecture | `E310P-BF16-COMPILE`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |

These are cross-compile outcomes only. They do not establish runtime dtype behavior or precision.

## Memory and alignment source snapshot

Evidence: id=E310P-MEMORY-SOURCE | status=UNVERIFIED | cann_version=NOT_APPLICABLE | compiler_version=NOT_APPLICABLE | compiler_sha256=NOT_APPLICABLE | container_image_digest=NOT_APPLICABLE | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=NOT_APPLICABLE

An unpinned source snapshot recorded the following candidate values. No source revision/blob was
retained, so none belongs in established facts:

| Unit | Historical alignment/format | Evidence state |
|---|---|---|
| Unified Buffer | 32 B | `E310P-MEMORY-SOURCE`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| L1 Buffer | 32 B, `FRACTAL_NZ` | `E310P-MEMORY-SOURCE`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| L0A Buffer | 512 B, `FRACTAL_ZZ` | `E310P-MEMORY-SOURCE`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| L0B Buffer | 512 B, `FRACTAL_ZN` | `E310P-MEMORY-SOURCE`; source revision/blob `UNPINNED`; `UNVERIFIED` |
| L0C Buffer | 64 B, `FRACTAL_NZ` | `E310P-MEMORY-SOURCE`; source revision/blob `UNPINNED`; `UNVERIFIED` |

## Two-operator trial cross-compile

Evidence: id=E310P-TWO-OP-COMPILE | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=UNKNOWN | compiler_sha256=UNPINNED | container_image_digest=UNPINNED | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=UNPINNED

Historical notes described edits needed to cross-compile `group_norm_silu` and `celu`: class-scope
constants, dependent template names, entry-function stack accumulation, an API candidate named
`Divs`, and half arithmetic. The actual CANN version and every immutable provenance field are
unknown or unpinned. These are research leads, not durable porting rules.

| Historical trial | Retained result | Evidence state |
|---|---|---|
| `group_norm_silu` | 13 branch objects reportedly emitted | `E310P-TWO-OP-COMPILE`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| `celu` | half and float objects reportedly emitted | `E310P-TWO-OP-COMPILE`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |

No resulting object is accepted evidence. Tiling layout/ABI, runtime, correctness, and precision
remain unverified.

## Stack-size trial cross-compile

Evidence: id=E310P-STACK-TRIAL | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=UNKNOWN | compiler_sha256=UNPINNED | container_image_digest=UNPINNED | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=UNPINNED

An earlier note reported a 16,000-byte compiler stack-frame limit and branch-count experiments.
Because the toolchain version, source, compiler, image, argv, and output objects are unpinned, this
number is not an established hardware or compiler limit.

| Historical branch trial | Retained compiler note | Evidence state |
|---|---|---|
| 1 / 2 / 4 / 6 branches | rc=0 | `E310P-STACK-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| 8 branches | stack frame 20,448 | `E310P-STACK-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| 10 branches | stack frame 25,440 | `E310P-STACK-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| 13 branches | stack frame 32,560 | `E310P-STACK-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |

## 24-operator batch trial cross-compile

Evidence: id=E310P-BATCH24-TRIAL | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=UNKNOWN | compiler_sha256=UNPINNED | container_image_digest=UNPINNED | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=UNPINNED

The retained tally did not identify a CANN version or immutable inputs/outputs. It remains a
historical observation only:

| Retained category | Count | Evidence state |
|---|---:|---|
| unknown type name | 15 | `E310P-BATCH24-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| cross-compile rc=0 | 4 | `E310P-BATCH24-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| global constant diagnostic | 2 | `E310P-BATCH24-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| Local Memory array diagnostic | 2 | `E310P-BATCH24-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |
| half-operation diagnostic | 1 | `E310P-BATCH24-TRIAL`; CANN `UNKNOWN`; compiler/image/source/artifact `UNPINNED`; `UNVERIFIED` |

An rc=0 row means only that an unpinned cross-compile reportedly emitted an object. It does not
authorize device, correctness, portability, or completion claims.

## Upstream host-branch source snapshot

Evidence: id=E310P-UPSTREAM-HOST | status=UNVERIFIED | cann_version=UNKNOWN | compiler_version=NOT_APPLICABLE | compiler_sha256=NOT_APPLICABLE | container_image_digest=NOT_APPLICABLE | source_revision=UNPINNED | source_blob_sha256=UNPINNED | artifact_sha256=NOT_APPLICABLE

An unpinned source search noted Ascend310P host branches in `group_norm_silu`, `grid_sample`,
`multi_scale_deformable_attn_function`, and `gather_elements_v2`. Without source revision/blob
hashes, those names are only future search anchors.

## Historical command shape

`PUBLIC_TOOLCHAIN_FINGERPRINT_UNAVAILABLE`: this is not an installation or reproduction path.
The old argv shape is retained only to describe what a future fingerprinted receipt must replace:

```text
<fingerprinted-bisheng> -std=c++17 --cce-aicore-arch=dav-m200 --cce-aicore-only -xcce \
  -I<fingerprinted-cann-root>/compiler/tikcpp/tikcfw \
  -I<fingerprinted-cann-root>/compiler/tikcpp/tikcfw/impl \
  -I<fingerprinted-cann-root>/compiler/tikcpp/tikcfw/interface \
  -I<op>/op_kernel \
  -c <op>.cpp -o <op>.o
```

Before any external replay, a repository-owned receipt must bind the immutable container image
digest, OS/CPU architecture, exact CANN package/version, compiler version and SHA-256, source
revision and blob SHA-256, complete argv, inputs, outputs, and controls. Until then, stop with
`PUBLIC_TOOLCHAIN_FINGERPRINT_UNAVAILABLE` before compiling.

## No source

There is no pinned source or device receipt here for core count, UB capacity, pipeline behavior,
SIMD register layout, sort/reduce primitives, numerical behavior, atomics, runtime bugs, or any
performance value. Do not infer these from Ascend950PR, Ascend910C, Ascend310, or Ascend310B.
