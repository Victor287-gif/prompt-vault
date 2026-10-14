# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Registered tag checks."""

from __future__ import annotations

from ..knowledge import Contract, Issue


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.metadata is None or document.path_info is None:
        return []
    tags = document.metadata.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        return []

    issues: list[Issue] = []
    dimensions: set[str] = set()
    for tag in tags:
        definition = contract.tags.get(tag)
        if not isinstance(definition, dict):
            issues.append(Issue("tag.unknown", f"unregistered tag: {tag}"))
        else:
            dimensions.add(str(definition.get("dimension")))
    for required in ("task", "topic"):
        if required not in dimensions:
            issues.append(
                Issue(f"tag.{required}", f"at least one {required} tag is required")
            )
    return issues
