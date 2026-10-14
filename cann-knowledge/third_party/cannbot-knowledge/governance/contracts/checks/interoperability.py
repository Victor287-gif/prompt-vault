# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Cross-technology mapping checks."""

from __future__ import annotations

from ..knowledge import Contract, Issue


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.metadata is None or document.path_info is None:
        return []
    if document.path_info.card_type != "interoperability":
        return []
    if not {"source_technology", "target_technology"} <= document.metadata.keys():
        return []
    source = document.metadata.get("source_technology")
    target = document.metadata.get("target_technology")
    if not isinstance(source, str) or not isinstance(target, str):
        return []
    issues: list[Issue] = []
    if source not in contract.technologies or target not in contract.technologies:
        issues.append(
            Issue(
                "technology.value",
                "source and target technology must be registered",
            )
        )
    if source == target:
        issues.append(
            Issue("technology.same", "source and target technology must differ")
        )
    return issues
