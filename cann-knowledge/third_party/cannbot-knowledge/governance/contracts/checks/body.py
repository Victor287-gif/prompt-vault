# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Knowledge body and description hygiene checks."""

from __future__ import annotations

import re

from ..knowledge import Contract, Issue
from .links import mask_code


CONFLICT_RE = re.compile(r"(?m)^(?:<{7}|={7}|>{7})(?: .*)?$")
H1_RE = re.compile(r"(?m)^#\s+\S")
COLLAPSED_TABLE_RE = re.compile(r"(?m)\|[ \t]+\|[ \t]*:?-{3,}")
WEAK_DESCRIPTION_RE = re.compile(
    r"^(?:本文件为文档发布时所需的索引文件。|> \[![^]]+]|表\s*\d+|以下内容针对如下型号生效)"
)


def check(contract: Contract, document, context) -> list[Issue]:
    issues = check_card(contract, document, context)
    if (
        document.parse_error is not None
        or document.path.name == "index.md"
        or document.metadata is None
    ):
        return issues
    if CONFLICT_RE.search(document.body):
        issues.append(
            Issue("body.conflict_marker", "body contains a Git conflict marker")
        )
    prose = mask_code(document.body, inline=False)
    h1_count = len(H1_RE.findall(prose))
    if h1_count == 0 or (
        h1_count > 1
        and contract.fixed_archive_mirror(document.metadata) is None
    ):
        issues.append(
            Issue(
                "body.h1_count",
                "knowledge card body must contain a level-1 heading; only fixed-archive API/Guide mirrors may preserve multiple source headings",
            )
        )
    if COLLAPSED_TABLE_RE.search(prose):
        issues.append(
            Issue(
                "body.collapsed_table",
                "Markdown table rows appear collapsed onto one physical line",
            )
        )
    description = document.metadata.get("description")
    if isinstance(description, str) and WEAK_DESCRIPTION_RE.match(description.strip()):
        issues.append(
            Issue(
                "frontmatter.description_placeholder",
                "description is publishing markup or a table label, not a standalone summary",
            )
        )
    return issues


def check_card(contract: Contract, document, context) -> list[Issue]:
    del contract, context
    if (
        document.parse_error is not None
        or document.path.name == "index.md"
        or document.metadata is None
        or document.path_info is None
        or document.body.strip()
    ):
        return []
    return [Issue("body.empty", "knowledge card body is empty")]
