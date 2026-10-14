# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Knowledge timestamp checks."""

from __future__ import annotations

import re
from datetime import datetime

from ..knowledge import Contract, Issue


RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def check(contract: Contract, document, context) -> list[Issue]:
    del contract, context
    if document.metadata is None or document.path_info is None:
        return []
    parsed: dict[str, datetime] = {}
    issues: list[Issue] = []
    for field in ("created_at", "updated_at"):
        if field not in document.metadata:
            continue
        value = document.metadata.get(field)
        if not isinstance(value, str) or not RFC3339_UTC_RE.fullmatch(value):
            issues.append(
                Issue(
                    "frontmatter.time",
                    f"{field} must use UTC RFC3339 seconds ending in Z",
                )
            )
            continue
        try:
            parsed[field] = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            issues.append(
                Issue("frontmatter.time", f"{field} is not a valid UTC calendar time")
            )
    if len(parsed) == 2 and parsed["updated_at"] < parsed["created_at"]:
        issues.append(
            Issue("frontmatter.time_order", "updated_at is earlier than created_at")
        )
    return issues
