# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Static registry and dispatcher for deterministic knowledge checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..knowledge import Contract, Issue, PathInfo
from . import (
    body,
    images,
    index_entry_description,
    index_sync,
    interoperability,
    lifecycle,
    links,
    local_sources,
    logs,
    navigation,
    path,
    platforms,
    profile,
    source_navigation,
    sources,
    tags,
    time,
    trust,
)


@dataclass(frozen=True)
class Document:
    path: Path
    relative: Path
    text: str
    metadata: dict[str, Any] | None
    body: str
    parse_error: str | None
    path_info: PathInfo | None
    path_error: str | None

    @classmethod
    def create(
        cls,
        contract: Contract,
        path: Path,
        relative: Path,
        text: str,
        metadata: dict[str, Any] | None,
        body_text: str,
        parse_error: str | None = None,
    ) -> Document:
        path_info: PathInfo | None = None
        path_error: str | None = None
        if path.name != "index.md":
            try:
                path_info = contract.derive_card_path(relative)
            except ValueError as exc:
                path_error = str(exc)
        return cls(
            path,
            relative,
            text,
            metadata,
            body_text,
            parse_error,
            path_info,
            path_error,
        )


@dataclass
class RunContext:
    strict_navigation: bool = False
    checked_local_roots: set[str] = field(default_factory=set)


DocumentCheck = Callable[[Contract, Document, RunContext], list[Issue]]
RepositoryCheck = Callable[[Contract], list[tuple[Path, Issue]]]
Tier = Literal["P0", "P1", "P2"]


@dataclass(frozen=True)
class Check:
    id: str
    tier: Tier
    document: DocumentCheck | None = None
    repository: RepositoryCheck | None = None
    card: DocumentCheck | None = None
    requires_clean: bool = False


CHECKS: tuple[Check, ...] = (
    Check("path", "P0", document=path.check, card=path.check),
    Check("profile", "P0", document=profile.check, card=profile.check),
    Check("body", "P0", document=body.check, card=body.check_card),
    Check("images", "P0", document=images.check, card=images.check),
    Check("tags", "P0", document=tags.check, card=tags.check),
    Check("sources", "P0", document=sources.check, card=sources.check),
    Check("time", "P0", document=time.check, card=time.check),
    Check("trust", "P0", document=trust.check, card=trust.check),
    Check("platforms", "P0", document=platforms.check, card=platforms.check),
    Check("lifecycle", "P0", document=lifecycle.check, card=lifecycle.check),
    Check(
        "interoperability",
        "P0",
        document=interoperability.check,
        card=interoperability.check,
    ),
    Check("source_navigation", "P0", document=source_navigation.check),
    Check(
        "navigation",
        "P0",
        document=navigation.check,
        repository=navigation.check_repository,
    ),
    Check(
        "index_entry_description",
        "P1",
        document=index_entry_description.check,
    ),
    Check(
        "local_sources",
        "P0",
        document=local_sources.check,
        repository=local_sources.check_repository,
    ),
    Check("links", "P0", document=links.check),
    Check("logs", "P0", repository=logs.check_repository),
    Check(
        "index_sync",
        "P0",
        repository=index_sync.check_repository,
        requires_clean=True,
    ),
)


def validate_card(
    contract: Contract,
    relative: Path,
    metadata: dict[str, Any],
    body_text: str,
) -> list[Issue]:
    document = Document.create(
        contract,
        contract.root / relative,
        relative,
        body_text,
        metadata,
        body_text,
    )
    context = RunContext()
    issues: list[Issue] = []
    for check in CHECKS:
        if check.card is not None:
            issues.extend(check.card(contract, document, context))
    return deduplicate(issues)


def deduplicate(issues: list[Issue]) -> list[Issue]:
    seen: set[tuple[str, str, int | None, str | None, Path | None]] = set()
    result: list[Issue] = []
    for issue in issues:
        key = (
            issue.code,
            issue.message,
            issue.line,
            issue.details,
            issue.path,
        )
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result
