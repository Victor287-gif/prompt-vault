# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Frontmatter Schema and path Profile checks."""

from __future__ import annotations

from ..knowledge import Contract, Issue


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.parse_error is not None:
        return [Issue("frontmatter.parse", document.parse_error)]
    if document.path.name == "index.md":
        return []
    if document.metadata is None:
        return [Issue("frontmatter.missing", "knowledge card requires frontmatter")]
    if document.path_info is None:
        return []

    metadata = document.metadata
    issues: list[Issue] = []
    for error in sorted(
        contract.schema_validator.iter_errors(metadata),
        key=lambda item: list(item.path),
    ):
        location = ".".join(str(part) for part in error.path) or "frontmatter"
        issues.append(Issue("frontmatter.schema", f"{location}: {error.message}"))

    card_type = document.path_info.card_type
    profile = contract.profiles[card_type]
    if not isinstance(profile, dict):
        return issues + [Issue("frontmatter.profile", f"invalid profile: {card_type}")]
    required = contract.common_fields | set(profile.get("required", []))
    allowed = required | set(profile.get("optional", []))
    for field in sorted(required - set(metadata)):
        issues.append(Issue("frontmatter.required", f"missing required field: {field}"))
    for field in sorted(set(metadata) - allowed):
        issues.append(
            Issue(
                "frontmatter.extra",
                f"field is not allowed by {card_type} profile: {field}",
            )
        )
    for field in ("title", "description"):
        value = metadata.get(field)
        if isinstance(value, str) and not value.strip():
            issues.append(
                Issue("frontmatter.text", f"{field} must be a non-empty string")
            )
    expected_type = contract.okf_type(card_type)
    if "type" in metadata and metadata["type"] != expected_type:
        issues.append(
            Issue(
                "frontmatter.type",
                f"type must be {expected_type} for {card_type}",
            )
        )
    return issues
