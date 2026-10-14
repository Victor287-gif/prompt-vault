# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Knowledge lifecycle checks."""

from __future__ import annotations

from pathlib import Path

from ..knowledge import Contract, Issue


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.metadata is None or document.path_info is None:
        return []
    status = document.metadata.get("status")
    replaced_by = document.metadata.get("replaced_by")
    if replaced_by is None:
        return []
    if "status" not in document.metadata:
        return []
    if status != "deprecated":
        return [Issue("status.replaced_by", "replaced_by requires deprecated status")]
    if (
        not isinstance(replaced_by, str)
        or replaced_by.startswith(("/", "knowledge/"))
        or replaced_by.endswith(".md")
        or ".." in Path(replaced_by).parts
    ):
        return [
            Issue(
                "status.replaced_by",
                "replaced_by must be a bundle-relative Concept ID without .md",
            )
        ]

    relative = Path("knowledge") / f"{replaced_by}.md"
    target = contract.root / relative
    try:
        info = contract.derive_card_path(relative)
        if info.doc_id != replaced_by:
            raise ValueError("replacement must use the canonical Concept ID")
        resolved = target.resolve()
        resolved.relative_to((contract.root / "knowledge").resolve())
        contract.derive_card_path(resolved.relative_to(contract.root.resolve()))
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            Issue(
                "status.replaced_by",
                f"replacement must identify a non-index Concept inside knowledge/: {exc}",
            )
        ]

    issues: list[Issue] = []
    if replaced_by == document.path_info.doc_id or resolved == document.path.resolve():
        issues.append(
            Issue("status.replaced_by", "replaced_by cannot reference the current card")
        )
    if not target.is_file():
        issues.append(
            Issue(
                "status.replaced_by",
                f"replacement card does not exist: {replaced_by}",
            )
        )
    return issues
