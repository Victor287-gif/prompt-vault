# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Pinned external image checks."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from ..knowledge import Contract, Issue
from .links import markdown_image_targets, mask_code
from .sources import git_raw_file_url_is_pinned


HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
OBSIDIAN_IMAGE_RE = re.compile(r"!\[\[[^\]\n]+\]\]")


def check(contract: Contract, document, context) -> list[Issue]:
    del contract, context
    if document.parse_error is not None:
        return []

    text = HTML_COMMENT_RE.sub(_blank, document.text)
    masked = mask_code(text)
    targets = markdown_image_targets(text)
    issues: list[Issue] = []
    for target, line, raw in html_image_targets(document.text):
        if target is None:
            issues.append(
                Issue(
                    "image.syntax",
                    "HTML image requires exactly one non-empty src URL",
                    line=line,
                    details=raw,
                )
            )
            continue
        targets.append((target, line))

    for match in OBSIDIAN_IMAGE_RE.finditer(masked):
        issues.append(
            Issue(
                "image.syntax",
                "Obsidian image embeds are not supported",
                line=_line(masked, match.start()),
                details=match.group(0),
            )
        )

    for target, line in sorted(targets, key=lambda item: item[1]):
        if not git_raw_file_url_is_pinned(target):
            issues.append(
                Issue(
                    "image.pin",
                    "image must use a direct HTTPS Git raw URL with a 40-hex commit and file path",
                    line=line,
                    details=target,
                )
            )
    return issues


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _blank(match: re.Match[str]) -> str:
    return "".join("\n" if char == "\n" else " " for char in match.group(0))


def html_image_targets(text: str) -> list[tuple[str | None, int, str]]:
    """Return actual HTML img src attributes outside code and comments."""

    without_comments = HTML_COMMENT_RE.sub(_blank, text)
    parser = _ImageHTMLParser()
    parser.feed(mask_code(without_comments))
    parser.close()
    return parser.targets


class _ImageHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.targets: list[tuple[str | None, int, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "img":
            return
        sources = [value for name, value in attrs if name.casefold() == "src"]
        target = sources[0] if len(sources) == 1 and sources[0] else None
        self.targets.append((target, self.getpos()[0], self.get_starttag_text()))
