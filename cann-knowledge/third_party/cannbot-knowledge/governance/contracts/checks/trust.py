# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Verification record checks."""

from __future__ import annotations

import re

from ..knowledge import Contract, Issue


ACTOR_RE = re.compile(r"^(?:human:[^\s/:]+|process:[^\s/:]+|[^\s/:]+/[^\s/]+)$")


def check(contract: Contract, document, context) -> list[Issue]:
    del contract, context
    if document.metadata is None or document.path_info is None:
        return []
    value = document.metadata.get("verified")
    if value is None:
        return []
    events = value if isinstance(value, list) else [value]
    issues: list[Issue] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        actor = event.get("by")
        if not isinstance(actor, str) or not ACTOR_RE.fullmatch(actor):
            issues.append(
                Issue(
                    "verified.actor",
                    f"verified[{index}].by must use human:<id>, process:<id>, or <producer>/<version>",
                )
            )
    return issues
