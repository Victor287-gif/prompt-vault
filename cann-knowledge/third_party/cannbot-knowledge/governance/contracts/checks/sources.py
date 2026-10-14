# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Frontmatter source resource and Git revision checks."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..knowledge import Contract, Issue
from ..source_resources import git_source


COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
GIT_VIEW_HOSTS = {
    "bitbucket.org",
    "codeberg.org",
    "gitcode.com",
    "github.com",
    "gitee.com",
    "gitlab.com",
}
RAW_GIT_HOSTS = {"raw.gitcode.com", "raw.githubusercontent.com"}


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.metadata is None or document.path_info is None:
        return []
    metadata = document.metadata
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        return []

    issues: list[Issue] = []
    if (
        metadata.get("status") in {"stable", "deprecated"}
        and not sources
        and metadata.get("type") != "Runbook"
    ):
        issues.append(
            Issue(
                "source.required",
                "non-draft cards other than Runbook require at least one source",
            )
        )
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        resource = source.get("resource")
        if not isinstance(resource, str) or not resource.strip():
            issues.append(
                Issue("source.resource", f"sources[{index}].resource must be non-empty")
            )
        elif re.match(r"^https?://", resource):
            parsed = urlparse(resource)
            if (
                not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                issues.append(
                    Issue(
                        "source.resource",
                        f"sources[{index}].resource is not a canonical HTTP(S) URL",
                    )
                )
            elif git_work_item_url_is_stable(resource):
                continue
            elif not git_url_is_pinned(resource):
                issues.append(
                    Issue(
                        "source.pin",
                        f"sources[{index}] Git resource is not pinned to a 40-hex commit",
                    )
                )
        elif not is_safe_local_resource(contract, resource):
            issues.append(
                Issue(
                    "source.resource",
                    f"sources[{index}].resource must be an HTTP(S) URL or registered local source",
                )
            )
    return issues


def is_safe_local_resource(contract: Contract, resource: str) -> bool:
    if resource != resource.strip() or "\\" in resource:
        return False
    parsed = urlparse(resource)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return False
    relative = Path(resource)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return False
    if relative.as_posix() != resource or len(relative.parts) < 2:
        return False
    try:
        (contract.root / relative).resolve().relative_to(contract.root)
    except ValueError:
        return False
    definition = contract.local_source_definition(resource)
    if definition is None:
        return False
    if definition.get("kind") != "git":
        return True
    try:
        return git_source(contract, resource) is not None
    except ValueError:
        return False


def git_url_is_pinned(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if host in GIT_VIEW_HOSTS:
        for marker in ("blob", "tree", "raw"):
            if marker in parts:
                index = parts.index(marker)
                return index + 1 < len(parts) and bool(
                    COMMIT_RE.fullmatch(parts[index + 1])
                )
        if host == "bitbucket.org" and "src" in parts:
            index = parts.index("src")
            return index + 1 < len(parts) and bool(
                COMMIT_RE.fullmatch(parts[index + 1])
            )
        if host == "codeberg.org" and "src" in parts:
            index = parts.index("src")
            return (
                index + 2 < len(parts)
                and parts[index + 1] == "commit"
                and bool(COMMIT_RE.fullmatch(parts[index + 2]))
            )
        return False
    if host == "raw.gitcode.com" and "raw" in parts:
        index = parts.index("raw")
        return index + 1 < len(parts) and bool(COMMIT_RE.fullmatch(parts[index + 1]))
    if host == "raw.githubusercontent.com" and len(parts) >= 3:
        return bool(COMMIT_RE.fullmatch(parts[2]))
    return host not in RAW_GIT_HOSTS


def git_work_item_url_is_stable(url: str) -> bool:
    """Return whether URL names one repository PR or issue."""

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.endswith("/")
    ):
        return False
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 4 or not re.fullmatch(r"[1-9][0-9]*", parts[3]):
        return False
    if host == "gitcode.com":
        return parts[2] in {"pull", "merge_requests", "issues"}
    if host == "github.com":
        return parts[2] in {"pull", "issues"}
    return False


def git_raw_file_url_is_pinned(url: str) -> bool:
    """Return whether URL directly serves a Git file at a 40-hex commit."""

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    host = (parsed.hostname or "").lower()
    parts = _canonical_path_parts(parsed.path)
    if parts is None:
        return False
    if host == "raw.gitcode.com":
        return (
            len(parts) >= 5
            and parts[2] == "raw"
            and bool(COMMIT_RE.fullmatch(parts[3]))
        )
    if host == "raw.githubusercontent.com":
        return len(parts) >= 4 and bool(COMMIT_RE.fullmatch(parts[2]))
    return False


def _canonical_path_parts(path: str) -> list[str] | None:
    decoded = unquote(path)
    if not decoded.startswith("/") or "\\" in decoded:
        return None
    parts = decoded.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return parts
