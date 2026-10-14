# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Platform scope checks."""

from __future__ import annotations

import re
from ..knowledge import Contract, Issue
from .links import mask_code


PRODUCT_SUPPORT_RE = re.compile(
    r"(?ms)^## [^\n]*\u4ea7\u54c1\u652f\u6301\u60c5\u51b5[^\n]*\n(.*?)(?=^## |\Z)"
)
HEADING_RE = re.compile(r"(?m)^#{1,6}[ \t]+(?P<title>[^\n]*)$")
DIRECT_CARD_SCOPE_RE = re.compile(
    r"^\s*(?:>\s*)?(?:[-*+]\s*)?"
    r"(?:\u76ee\u524d|\u5f53\u524d)?(?:\u4ec5|\u53ea)\u652f\u6301(?:\u5728)?"
    r"(?P<scope>[^\u3002\uff1b;\n]{1,180})"
)
SUBJECT_CARD_SCOPE_RE = re.compile(
    r"(?:\u672c|\u8be5)(?:\u6837\u4f8b|\u793a\u4f8b|\u6027\u80fd\u4f18\u5316\u624b\u6bb5|\u4f18\u5316\u624b\u6bb5|\u63a5\u53e3|API|\u7b97\u5b50|\u529f\u80fd)"
    r"(?:\u4e2d|\u4e0b)?(?:\u76ee\u524d|\u5f53\u524d)?(?:\u4ec5|\u53ea)"
    r"(?:\u652f\u6301(?:\u5728)?|\u9488\u5bf9|\u9762\u5411)"
    r"(?P<scope>[^\u3002\uff1b;\n]{1,180})"
)

MIRROR_PLATFORM_PATTERNS = {
    "950": re.compile(r"Ascend\s*950"),
    "a2": re.compile(r"Atlas\s*A2|AtlasA2|200I/500\s*A2|200I/500A2"),
    "a3": re.compile(r"Atlas\s*A3|AtlasA3"),
    "310p": re.compile(r"Atlas\s*\u63a8\u7406\u7cfb\u5217\u4ea7\u54c1"),
    "910": re.compile(r"Atlas\s*\u8bad\u7ec3\u7cfb\u5217\u4ea7\u54c1"),
    "kirin_x90": re.compile(r"Kirin\s*X90", re.IGNORECASE),
    "kirin_9030": re.compile(r"Kirin\s*9030", re.IGNORECASE),
}
MIRROR_PLATFORM_ORDER = (
    "950",
    "a2",
    "a3",
    "310p",
    "910",
    "kirin_x90",
    "kirin_9030",
)


def _supported_product_row(row: str) -> bool:
    if "\u4e0d\u652f\u6301" in row or "\u5b9e\u9645\u529f\u80fd\u4e0d\u751f\u6548" in row:
        return False
    if "\u652f\u6301" in row:
        return True
    # Some official product-support sections enumerate the supported capability
    # after a product-name colon instead of spelling out "supported" on every row.
    return re.search(r"[\uff1a:]\s*\S", row) is not None


def _explicit_narrative_platforms(body: str) -> list[str] | None:
    result: list[str] = []
    for line in body.splitlines():
        match = DIRECT_CARD_SCOPE_RE.search(line) or SUBJECT_CARD_SCOPE_RE.search(line)
        if match is None:
            continue
        scope = match.group("scope")
        # Mixed positive/negative sentences require a richer clause parser; keep
        # them unlocated instead of widening the card from an ambiguous sentence.
        if "\u4e0d\u652f\u6301" in scope or "\u4e0d\u751f\u6548" in scope:
            continue
        for platform in MIRROR_PLATFORM_ORDER:
            platform_match = MIRROR_PLATFORM_PATTERNS[platform].search(scope)
            # The product must be the object of the restrictive clause, rather
            # than a later mention in an architecture comparison or subfeature.
            if (
                platform_match is not None
                and platform_match.start() <= 24
                and platform not in result
            ):
                result.append(platform)
    return result or None


def fixed_archive_platforms(body: str) -> list[str] | None:
    """Return platforms explicitly supported by a mirrored source document.

    None means that the document has no parseable support table, capability list,
    or restrictive support statement.  A platform is retained when at least one
    product row in that governed family is supported.  The fixed-archive producer
    may omit that source section from the card after deriving Frontmatter, so this
    parser remains the source-side verification contract.  Compile-only rows whose
    function does not take effect are not treated as runtime support.
    """

    body = mask_code(body, inline=False)
    match = PRODUCT_SUPPORT_RE.search(body)
    if match is None:
        return _explicit_narrative_platforms(body)
    section = match.group(1)
    result: list[str] = []
    for platform in MIRROR_PLATFORM_ORDER:
        pattern = MIRROR_PLATFORM_PATTERNS[platform]
        rows = [line for line in section.splitlines() if pattern.search(line)]
        if any(_supported_product_row(row) for row in rows):
            result.append(platform)
    return result


def _contains_omitted_heading(body: str, fragments: list[str]) -> bool:
    body = mask_code(body, inline=False)
    return any(
        fragment in re.sub(r"<[^>]+>", "", match.group("title"))
        for match in HEADING_RE.finditer(body)
        for fragment in fragments
    )


def check(contract: Contract, document, context) -> list[Issue]:
    del context
    if document.metadata is None or document.path_info is None:
        return []
    if document.path_info.card_type == "glossaries":
        return []
    platforms = document.metadata.get("platforms")
    if not isinstance(platforms, list) or not all(
        isinstance(item, str) for item in platforms
    ):
        return []
    issues: list[Issue] = []
    if not platforms and document.metadata.get("status") in {"stable", "deprecated"}:
        issues.append(
            Issue("platform.unknown", "empty platforms requires status draft")
        )
    if "agnostic" in platforms and len(platforms) != 1:
        issues.append(
            Issue(
                "platform.agnostic",
                "agnostic cannot be combined with concrete platforms",
            )
        )
    for platform in platforms:
        if platform not in contract.platforms:
            issues.append(
                Issue("platform.unknown", f"unregistered platform: {platform}")
            )
    mirror_match = contract.fixed_archive_mirror(document.metadata)
    if mirror_match is not None:
        mirror, source_path = mirror_match
        omitted = [
            str(value) for value in mirror.get("omit_section_title_contains", [])
        ]
        if _contains_omitted_heading(document.body, omitted):
            issues.append(
                Issue(
                    "platform.redundant_body_scope",
                    "fixed-archive mirror must keep product support in Frontmatter only",
                )
            )
        if omitted and not source_path.is_file():
            return issues
        source_body = (
            source_path.read_text(encoding="utf-8")
            if omitted
            else document.body
        )
        expected = fixed_archive_platforms(source_body)
        if expected is None:
            if platforms or document.metadata.get("status") != "draft":
                issues.append(
                    Issue(
                        "platform.source_unknown",
                        "fixed-archive mirror without parseable explicit platform scope must be draft with empty platforms",
                        details=(
                            f"expected_status='draft' expected_platforms=[] "
                            f"actual_status={document.metadata.get('status')!r} "
                            f"actual_platforms={platforms!r}"
                        ),
                    )
                )
        elif platforms != expected:
            issues.append(
                Issue(
                    "platform.source_mismatch",
                    "fixed-archive mirror platforms do not match its explicit source scope",
                    details=f"expected={expected!r} actual={platforms!r}",
                )
            )
    return issues
