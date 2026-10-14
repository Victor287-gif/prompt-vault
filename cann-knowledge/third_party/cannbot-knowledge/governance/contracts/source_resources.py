# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Registered archive and pinned Git source locations and read-only validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
from urllib.parse import quote, unquote, urlsplit


def validate_archive(root: Path, definition: dict) -> Path:
    """Check acquisition completion and origin, without pinning rolling content."""
    directory = root / definition["root"]
    marker = directory / ".complete"
    if (directory.resolve() != directory or not directory.is_dir()
            or marker.is_symlink() or not marker.is_file()):
        raise ValueError(f"restored documentation requires a regular .complete marker: {directory}")
    expected = f"archive_url={definition['archive_url']}"
    if expected not in marker.read_text(encoding="utf-8").splitlines():
        raise ValueError(f"restored documentation archive URL does not match Registry: {directory}")
    return directory


@dataclass(frozen=True)
class GitSource:
    root: str
    repository: str
    revision: str
    path: str

    @property
    def resource(self) -> str:
        return f"{self.root}/{self.revision}/{self.path}"

    @property
    def url(self) -> str:
        return f"{self.repository}/blob/{self.revision}/{quote(self.path, safe='/._-')}"

    def checkout(self, repository_root: Path) -> Path:
        return repository_root / self.root / self.revision

    def local_path(self, repository_root: Path) -> Path:
        checkout = self.checkout(repository_root)
        target = checkout / self.path
        if checkout.resolve() != checkout or not target.resolve().is_relative_to(checkout):
            raise ValueError(f"source escapes its pinned checkout: {self.resource}")
        return target


def safe_path(value: str) -> bool:
    path = Path(value)
    return bool(
        value and value == value.strip() and path.as_posix() == value
        and not path.is_absolute() and not any(part in {"..", ".git", "."} for part in path.parts)
        and not any(char in value for char in "\\?#\n\r\0")
    )


def git_source(contract, resource: str) -> GitSource | None:
    """Resolve a registered remote source or commit-qualified local resource.

    Unknown repositories return None. A registered location with an unsafe path
    or unregistered revision fails closed, including while sources are offline.
    """
    for definition in contract.local_sources.values():
        if definition.get("kind") != "git":
            continue
        root = definition["root"]
        repository = definition["repository"]
        if resource.startswith(repository + "/"):
            parsed = urlsplit(resource)
            suffix = parsed.path[len(urlsplit(repository).path) + 1:].removeprefix("-/")
            parts = suffix.split("/", 2)
            if len(parts) != 3 or parts[0] not in {"blob", "tree", "raw"} or parsed.query or parsed.fragment:
                raise ValueError(f"invalid pinned Git source: {resource}")
            _, revision, path = parts
            path = unquote(path)
        elif resource.startswith(root + "/"):
            parts = resource[len(root) + 1:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"local Git source requires commit and path: {resource}")
            revision, path = parts
        else:
            continue
        if revision not in definition["revisions"] or not safe_path(path):
            raise ValueError(f"unregistered revision or unsafe Git source path: {resource}")
        source = GitSource(root, repository, revision, path)
        source.local_path(contract.root)
        return source
    return None


def git_read(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-C", str(directory), *args],
        text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"Git source validation failed in {directory}: {(result.stderr or '').strip()}")
    return (result.stdout or "").strip()


def validate_checkout(root: Path, definition: dict, revision: str) -> Path:
    checkout = root / definition["root"] / revision
    if checkout.resolve() != checkout or not (checkout / ".git").exists():
        raise ValueError(f"missing pinned Git checkout: {checkout}")
    if git_read(checkout, "rev-parse", "HEAD") != revision:
        raise ValueError(f"Git source revision mismatch: {checkout}")
    if git_read(checkout, "remote", "get-url", "origin").removesuffix(".git") != definition["repository"]:
        raise ValueError(f"Git source origin mismatch: {checkout}")
    if git_read(checkout, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError(f"Git source has local changes; refusing to overwrite or trust it: {checkout}")
    # Sparse or assume-unchanged entries would hide content from the normal dirty check.
    if any(line[:1] != "H" for line in git_read(checkout, "ls-files", "-v").splitlines()):
        raise ValueError(f"Git source has hidden or sparse index entries: {checkout}")
    return checkout


def local_source_link(contract, path: Path, target: str) -> GitSource | None:
    """Allow only registered source-file links outside the knowledge Bundle."""
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or parsed.query or target.startswith("/"):
        return None
    if parsed.fragment and not re.fullmatch(r"L[1-9][0-9]*(?:-L[1-9][0-9]*)?", parsed.fragment):
        return None
    try:
        relative = Path(os.path.normpath(path.parent / unquote(parsed.path))).relative_to(contract.root)
    except ValueError:
        return None
    return git_source(contract, relative.as_posix())
