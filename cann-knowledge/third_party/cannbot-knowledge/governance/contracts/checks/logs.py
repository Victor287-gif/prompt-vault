# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Repository governance log checks."""

from __future__ import annotations

import re
from pathlib import Path

from ..knowledge import Contract, Issue


def check_repository(contract: Contract) -> list[tuple[Path, Issue]]:
    log_root = contract.root / "logs"
    if not log_root.is_dir():
        return [(log_root, Issue("log.missing", "repository requires logs/"))]
    results: list[tuple[Path, Issue]] = []
    for path in sorted(log_root.glob("*.md")):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", path.name):
            results.append(
                (path, Issue("log.filename", "log filename must use YYYY-MM-DD.md"))
            )
            continue
        date = path.stem
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != f"# {date}":
            results.append((path, Issue("log.heading", f"first line must be # {date}")))
        times: list[str] = []
        entries = [index for index, line in enumerate(lines) if line.startswith("## ")]
        for position, index in enumerate(entries):
            line = lines[index]
            match = re.match(r"## \[(\d{2}:\d{2})\] [^|]+ \| .+", line)
            valid_time = False
            if match:
                hour, minute = (int(value) for value in match.group(1).split(":"))
                valid_time = hour < 24 and minute < 60
            if not valid_time:
                results.append(
                    (path, Issue("log.entry", f"invalid log entry heading: {line}"))
                )
            else:
                assert match is not None
                times.append(match.group(1))
            end = entries[position + 1] if position + 1 < len(entries) else len(lines)
            section_headings = [
                item for item in lines[index + 1 : end] if item.startswith("### ")
            ]
            if (
                section_headings.count("### Summary") != 1
                or section_headings.count("### Details") != 1
                or section_headings.index("### Summary")
                > section_headings.index("### Details")
            ):
                results.append(
                    (
                        path,
                        Issue(
                            "log.sections",
                            "each log entry requires Summary then Details",
                        ),
                    )
                )
        if not times:
            results.append(
                (path, Issue("log.entry", "log file requires at least one entry"))
            )
        elif times != sorted(times, reverse=True):
            results.append(
                (path, Issue("log.order", "log entries must be newest first"))
            )
    return results
