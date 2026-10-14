# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Knowledge card path checks."""

from __future__ import annotations

import re
from pathlib import Path

from ..knowledge import Contract, Issue, PathInfo


ID_RE = re.compile(r"^[a-z0-9_]+$")
FREELY_NESTED_PROFILES = {"apis", "concepts", "examples", "guides", "optimizations"}


def check(contract: Contract, document, context) -> list[Issue]:
    del contract, context
    if (
        document.parse_error is not None
        or document.path.name == "index.md"
        or document.path_error is None
    ):
        return []
    return [Issue("path.invalid", document.path_error)]


def derive_card_path(contract: Contract, relative: Path) -> PathInfo:
    if (
        relative.is_absolute()
        or relative.suffix != ".md"
        or relative.name == "index.md"
    ):
        raise ValueError("card path must be a non-index Markdown path under knowledge/")
    parts = relative.parts
    if not parts or parts[0] != "knowledge" or ".." in parts:
        raise ValueError("card path must stay under knowledge/")
    if not ID_RE.fullmatch(relative.stem):
        raise ValueError("card filename must use lowercase snake_case")
    if len(parts) < 4:
        raise ValueError("card path is missing a knowledge type")
    domain = parts[1]
    if domain not in contract.domains:
        raise ValueError(f"unregistered knowledge domain: {domain}")

    type_indexes = [
        index for index, part in enumerate(parts[:-1]) if part in contract.types
    ]
    if len(type_indexes) != 1:
        raise ValueError("card path must contain exactly one registered knowledge type")
    type_index = type_indexes[0]
    card_type = parts[type_index]
    route: str | None
    if type_index == 2:
        route = None
        if card_type not in set(contract.domains[domain].get("direct_types", [])):
            raise ValueError(f"domain {domain} does not allow direct type: {card_type}")
    elif type_index == 3:
        route = parts[2]
        if route not in contract.routes:
            raise ValueError(f"unregistered technology or scope: {route}")
        if domain not in contract.route_domains(route):
            raise ValueError(f"route {route} is not allowed under domain {domain}")
    else:
        raise ValueError("path must be domain/type or domain/technology-or-scope/type")

    suffix = parts[type_index + 1 : -1]
    if card_type == "operators":
        if len(suffix) != 3 or suffix[0] not in contract.operator_call_modes:
            allowed = "|".join(sorted(contract.operator_call_modes))
            raise ValueError(
                "operator path must be "
                f"operators/<{allowed}>/<family>/<category>/<operator>.md"
            )
        if suffix[1] not in contract.operator_families:
            allowed = "|".join(sorted(contract.operator_families))
            raise ValueError(
                f"unregistered operator family: {suffix[1]}; expected <{allowed}>"
            )
        if suffix[2] not in contract.operator_categories[suffix[1]]:
            raise ValueError(
                f"unregistered operator category for {suffix[1]}: {suffix[2]}"
            )
    elif card_type == "runbooks":
        if len(suffix) != 1 or suffix[0] not in contract.runbook_areas:
            allowed = "|".join(sorted(contract.runbook_areas))
            raise ValueError(f"runbook path must be runbooks/<{allowed}>/<runbook>.md")
    elif card_type in FREELY_NESTED_PROFILES:
        if any(not ID_RE.fullmatch(part) for part in suffix):
            raise ValueError(
                f"{card_type} subdirectories must use lowercase snake_case"
            )
    elif suffix:
        raise ValueError(f"{card_type} cards cannot add unregistered subdirectories")
    if card_type == "interoperability" and route != "shared":
        raise ValueError("interoperability cards must be placed under a shared scope")
    bundle_relative = Path(*relative.parts[1:])
    return PathInfo(
        domain, route, card_type, bundle_relative.with_suffix("").as_posix()
    )
