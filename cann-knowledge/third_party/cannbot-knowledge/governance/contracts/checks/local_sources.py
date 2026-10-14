# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Restored local-source and fetch-script checks."""

from __future__ import annotations

import os
from pathlib import Path

from ..knowledge import Contract, Issue
from ..source_resources import git_source, validate_archive, validate_checkout


def check(contract: Contract, document, context) -> list[Issue]:
    if document.metadata is None or document.path.name == "index.md":
        return []
    sources = document.metadata.get("sources")
    if not isinstance(sources, list):
        return []
    issues: list[Issue] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        resource = source.get("resource")
        if not isinstance(resource, str) or resource.startswith(
            ("http://", "https://")
        ):
            continue
        relative = Path(resource)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            continue
        definition = contract.local_source_definition(resource)
        if not isinstance(definition, dict):
            continue
        if definition.get("kind") == "git":
            try:
                location = git_source(contract, resource)
                if location is None or not (contract.root / location.root).exists():
                    continue
                key = f"{location.root}/{location.revision}"
                if key not in context.checked_local_roots:
                    validate_checkout(contract.root, definition, location.revision)
                    context.checked_local_roots.add(key)
                target = location.local_path(contract.root)
                if not target.exists() or not (target.is_file() or target.is_dir()):
                    raise ValueError(f"local Git source does not exist: {resource}")
            except (OSError, ValueError) as exc:
                issues.append(Issue("source.git_checkout", str(exc)))
            continue
        source_root = contract.root / relative.parts[0]
        if not source_root.exists():
            continue
        if relative.parts[0] not in context.checked_local_roots:
            context.checked_local_roots.add(relative.parts[0])
            marker = source_root / ".complete"
            try:
                validate_archive(contract.root, definition)
            except (OSError, ValueError) as exc:
                issues.append(Issue("source.archive_marker", str(exc), path=marker))
        if not (contract.root / relative).is_file():
            issues.append(
                Issue(
                    "source.local_missing",
                    f"local source file does not exist: {resource}",
                )
            )
    return issues


def check_repository(contract: Contract) -> list[tuple[Path, Issue]]:
    results: list[tuple[Path, Issue]] = []
    for source_id, definition in contract.local_sources.items():
        fetch_script = contract._safe_repository_path(str(definition["fetch_script"]))
        if not fetch_script.is_file():
            results.append(
                (
                    contract.root / "governance/schemas/registries.yaml",
                    Issue(
                        "source.fetch_script_missing",
                        f"local source {source_id} fetch script does not exist: "
                        f"{fetch_script.relative_to(contract.root)}",
                    ),
                )
            )
        elif not os.access(fetch_script, os.X_OK):
            results.append(
                (
                    fetch_script,
                    Issue(
                        "source.fetch_script_mode",
                        f"local source {source_id} fetch script must be executable",
                    ),
                )
            )
    return results
