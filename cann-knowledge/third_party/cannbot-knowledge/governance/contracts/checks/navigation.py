# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Index path, structure, and direct-child navigation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..knowledge import Contract, Issue
from .links import links_in_text
from .path import FREELY_NESTED_PROFILES, ID_RE


OKF_VERSION = "0.2"


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.path.name != "index.md" or document.parse_error is not None:
        return []
    return [
        *validate_index_frontmatter(document.relative, document.metadata),
        *validate_index_path(contract, document.relative),
    ]


def validate_index_frontmatter(
    relative: Path,
    metadata: dict[str, Any] | None,
) -> list[Issue]:
    if relative == Path("knowledge/index.md"):
        if metadata != {"okf_version": OKF_VERSION}:
            return [
                Issue(
                    "index.frontmatter",
                    f"bundle root index.md requires only okf_version: {OKF_VERSION}",
                )
            ]
        return []
    if metadata is not None:
        return [
            Issue(
                "index.frontmatter",
                "only the bundle root index.md may have frontmatter",
            )
        ]
    return []


def validate_index_path(contract: Contract, relative: Path) -> list[Issue]:
    if (
        relative.name != "index.md"
        or not relative.parts
        or relative.parts[0] != "knowledge"
    ):
        return [Issue("index.path", "index.md must stay under knowledge/")]
    parts = relative.parts[:-1]
    if parts == ("knowledge",):
        return []
    domain = parts[1] if len(parts) > 1 else ""
    if domain not in contract.domains:
        return [Issue("index.domain", f"unregistered knowledge domain: {domain}")]
    if len(parts) == 2:
        return []

    route: str | None = None
    card_type: str | None = None
    tail: tuple[str, ...] = ()
    if parts[2] in contract.types:
        card_type = parts[2]
        if card_type not in set(contract.domains[domain].get("direct_types", [])):
            return [
                Issue(
                    "index.type",
                    f"domain {domain} does not allow direct type: {card_type}",
                )
            ]
        tail = parts[3:]
    else:
        route = parts[2]
        if route not in contract.routes:
            return [Issue("index.route", f"unregistered technology or scope: {route}")]
        if domain not in contract.route_domains(route):
            return [
                Issue(
                    "index.route",
                    f"route {route} is not allowed under domain {domain}",
                )
            ]
        if len(parts) == 3:
            return []
        card_type = parts[3]
        if card_type not in contract.types:
            return [Issue("index.type", f"unregistered knowledge type: {card_type}")]
        tail = parts[4:]

    if card_type == "operators":
        if len(tail) > 3:
            return [
                Issue(
                    "index.operator",
                    "operator index is deeper than call-mode/family/category",
                )
            ]
        if tail and tail[0] not in contract.operator_call_modes:
            return [
                Issue(
                    "index.operator",
                    f"unregistered operator call mode: {tail[0]}",
                )
            ]
        if len(tail) >= 2 and tail[1] not in contract.operator_families:
            return [
                Issue(
                    "index.operator",
                    f"unregistered operator family: {tail[1]}",
                )
            ]
        if (
            len(tail) == 3
            and tail[1] in contract.operator_families
            and tail[2] not in contract.operator_categories[tail[1]]
        ):
            return [
                Issue(
                    "index.operator",
                    f"unregistered operator category for {tail[1]}: {tail[2]}",
                )
            ]
    elif card_type == "runbooks":
        if len(tail) > 1 or (tail and tail[0] not in contract.runbook_areas):
            return [
                Issue(
                    "index.runbook",
                    "runbook index must stop at a registered area",
                )
            ]
    elif card_type in FREELY_NESTED_PROFILES:
        if any(not ID_RE.fullmatch(part) for part in tail):
            return [
                Issue(
                    "index.depth",
                    f"{card_type} subdirectories must use lowercase snake_case",
                )
            ]
    elif tail:
        return [
            Issue(
                "index.depth",
                f"{card_type} index has an unregistered subdirectory",
            )
        ]
    if card_type == "interoperability" and route != "shared":
        return [
            Issue(
                "index.route",
                "interoperability must be placed under a shared scope",
            )
        ]
    return []


def check_repository(contract: Contract) -> list[tuple[Path, Issue]]:
    return validate_index_tree(contract)


def validate_index_tree(contract: Contract) -> list[tuple[Path, Issue]]:
    knowledge_root = contract.root / "knowledge"
    results: list[tuple[Path, Issue]] = []
    structural = structural_directories(contract)
    for relative in sorted(structural):
        directory = contract.root / relative
        if not directory.is_dir():
            results.append(
                (
                    directory,
                    Issue(
                        "structure.missing",
                        "required domain or predeclared route directory is missing",
                    ),
                )
            )
    for directory in sorted(
        path for path in knowledge_root.rglob("*") if path.is_dir()
    ):
        index = directory / "index.md"
        if not index.is_file():
            results.append(
                (
                    directory,
                    Issue("index.missing", "knowledge directory requires index.md"),
                )
            )
            continue
        if directory.relative_to(
            contract.root
        ) not in structural and not has_knowledge_content(contract, directory):
            results.append(
                (
                    directory,
                    Issue("index.empty", "directory has no real knowledge content"),
                )
            )
        results.extend(
            (index, issue)
            for issue in validate_index_path(contract, index.relative_to(contract.root))
        )
        results.extend(
            (index, issue) for issue in validate_index_links(contract, index)
        )
    root_index = knowledge_root / "index.md"
    if not root_index.is_file():
        results.append(
            (knowledge_root, Issue("index.missing", "knowledge root requires index.md"))
        )
    else:
        results.extend(
            (root_index, issue)
            for issue in validate_index_path(
                contract,
                root_index.relative_to(contract.root),
            )
        )
        results.extend(
            (root_index, issue) for issue in validate_index_links(contract, root_index)
        )
    return results


def structural_directories(contract: Contract) -> set[Path]:
    result = {Path("knowledge") / domain for domain in contract.domains}
    precreated_domains = {
        domain
        for domain, definition in contract.domains.items()
        if definition.get("precreate_routes", False)
    }
    for route in sorted(contract.routes):
        for domain in contract.route_domains(route) & precreated_domains:
            result.add(Path("knowledge") / domain / route)
    return result


def has_knowledge_content(contract: Contract, directory: Path) -> bool:
    """Return whether a directory has a valid direct card or child index."""

    for child in directory.iterdir():
        if child.is_file() and child.suffix == ".md" and child.name != "index.md":
            try:
                contract.derive_card_path(child.relative_to(contract.root))
            except ValueError:
                continue
            return True
        if child.is_dir():
            child_index = child / "index.md"
            if (
                child_index.is_file()
                and not validate_index_path(
                    contract,
                    child_index.relative_to(contract.root),
                )
            ):
                return True
    return False


def validate_index_links(contract: Contract, index: Path) -> list[Issue]:
    text = index.read_text(encoding="utf-8")
    expected: set[Path] = set()
    for child in index.parent.iterdir():
        if child.name == "index.md":
            continue
        if child.is_dir() and (child / "index.md").is_file():
            expected.add((child / "index.md").resolve())
        elif child.is_file() and child.suffix == ".md":
            expected.add(child.resolve())

    actual: list[tuple[Path, str, int]] = []
    issues: list[Issue] = []
    for target_text, target, line in links_in_text(
        text,
        index,
        contract.root / "knowledge",
    ):
        actual.append((target, target_text, line))
        if target not in expected:
            issues.append(
                Issue(
                    "index.non_direct",
                    f"index links a non-direct child: {target_text}",
                    line=line,
                    details=target_text,
                )
            )
    actual_targets = {target for target, _, _ in actual}
    for target in sorted(expected - actual_targets):
        issues.append(
            Issue(
                "index.unlisted",
                f"direct child is not listed: {target.relative_to(index.parent)}",
                details=target.relative_to(index.parent).as_posix(),
            )
        )
    seen: set[Path] = set()
    for target, raw, line in actual:
        if target in seen:
            issues.append(
                Issue(
                    "index.duplicate",
                    "index contains duplicate local targets",
                    line=line,
                    details=raw,
                )
            )
            break
        seen.add(target)
    return issues
