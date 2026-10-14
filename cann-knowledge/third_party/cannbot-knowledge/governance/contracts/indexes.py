# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Pure generation and comparison helpers for knowledge indexes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .knowledge import markdown_targets, split_frontmatter


DIRECT_ENTRY_RE = re.compile(
    r"^- \[(?:\\.|[^]\\])+\]\(([^)#]+)(?:#[^)]*)?\)(?:：.*)?$"
)
REFERENCE_ENTRY_RE = re.compile(
    r"^- \[((?:\\.|[^]\\])+)\]\[((?:\\.|[^]\\])*)\](?:：.*)?$"
)
REFERENCE_DEFINITION_LINE_RE = re.compile(r"^[ \t]{0,3}\[[^]\n]+\]:")
WEAK_SUMMARY_RE = re.compile(
    r"^(?:本文件为文档发布时所需的索引文件。|> \[![^]]+]|表\s*\d+|以下内容针对如下型号生效)"
)


@dataclass(frozen=True)
class IndexUpdate:
    path: Path
    content: str


def read_card_description(card_path: Path) -> tuple[str, str]:
    metadata, _ = split_frontmatter(card_path.read_text(encoding="utf-8"))
    if metadata is None:
        raise ValueError(f"knowledge card is missing frontmatter: {card_path}")
    title = str(metadata.get("title", "")).strip()
    description = " ".join(str(metadata.get("description", "")).split())
    if not title or not description:
        raise ValueError(f"knowledge card requires title and description: {card_path}")
    return title, description


def _summary_card(directory: Path) -> Path | None:
    """Return the directory's curated overview card when one exists."""
    candidates = (directory / f"{directory.name}.md", directory / "overview.md")
    for candidate in candidates:
        if candidate.name != "index.md" and candidate.is_file():
            return candidate
    return None


def summarize_subdir(directory: Path) -> str:
    """Lead with a curated overview card, then describe the collection itself."""
    direct_cards = sorted(
        path for path in directory.glob("*.md") if path.name != "index.md"
    )
    descendants = sorted(
        path for path in directory.rglob("*.md") if path.name != "index.md"
    )
    count = len(descendants)
    if count == 0:
        return f"{directory.name.replace('_', ' ')} 导航。"
    if count == 1 and direct_cards:
        return read_card_description(direct_cards[0])[1]
    child_dirs = sorted(
        {
            path.relative_to(directory).parts[0]
            for path in descendants
            if path.parent != directory
        }
    )
    if direct_cards:
        summary_card = _summary_card(directory)
        example_paths = [
            path for path in direct_cards if path != summary_card
        ][:3]
        examples = [read_card_description(path)[0] for path in example_paths]
        if summary_card is not None:
            description = read_card_description(summary_card)[1]
            if examples:
                more = " 等" if count > len(examples) + 1 else ""
                return (
                    f"{description}（共 {count} 张知识卡："
                    f"{'、'.join(examples)}{more}）"
                )
            spread = f"，分布在 {len(child_dirs)} 个子目录" if child_dirs else ""
            return f"{description}（共 {count} 张知识卡{spread}）"
        more = " 等" if count > len(examples) else ""
        return f"共 {count} 张知识卡（如 {'、'.join(examples)}{more}）。"
    shown = "、".join(child_dirs[:3])
    more = " 等" if len(child_dirs) > 3 else ""
    return (
        f"共 {count} 张知识卡，分布在 {len(child_dirs)} 个子目录（{shown}{more}）。"
    )


def section_title(directory: Path) -> str:
    summary_card = directory / f"{directory.name}.md"
    if summary_card.name != "index.md" and summary_card.is_file():
        return read_card_description(summary_card)[0]
    index = directory / "index.md"
    if index.is_file():
        match = re.search(
            r"(?m)^#\s+(.+?)\s*$", index.read_text(encoding="utf-8")
        )
        if match is not None:
            return match.group(1)
    return directory.name.replace("_", " ")


def markdown_link_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def desired_entries(directory: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in sorted(directory.iterdir()):
        if item.name == "index.md":
            continue
        if item.is_dir() and (
            (item / "index.md").is_file()
            or any(path.suffix == ".md" for path in item.rglob("*.md"))
        ):
            target = f"{item.name}/index.md"
            result.append(
                (
                    target,
                    f"- [{markdown_link_label(section_title(item))}]({target})：{summarize_subdir(item)}",
                )
            )
        elif item.is_file() and item.suffix == ".md":
            title, description = read_card_description(item)
            result.append(
                (
                    item.name,
                    f"- [{markdown_link_label(title)}]({item.name})：{description}",
                )
            )
    return result


def direct_entry_target(line: str, reference_definitions: str = "") -> str | None:
    match = DIRECT_ENTRY_RE.fullmatch(line)
    target = match.group(1) if match is not None else None
    if target is None and REFERENCE_ENTRY_RE.fullmatch(line):
        targets = markdown_targets(line + "\n" + reference_definitions)
        if len(targets) == 1 and not targets[0][1]:
            target = targets[0][0].split("#", 1)[0]
    if target is None:
        return None
    path = Path(target)
    if (
        len(path.parts) == 1
        and path.suffix == ".md"
        or len(path.parts) == 2
        and path.parts[1] == "index.md"
    ):
        return target
    return None


def refreshed_entry(existing: str, generated: str) -> str:
    if "：" not in existing or "：" not in generated:
        return existing
    summary = existing.split("：", 1)[1].strip()
    if not WEAK_SUMMARY_RE.match(summary) and "> [!" not in summary:
        return existing
    return existing.split("：", 1)[0] + "：" + generated.split("：", 1)[1]


def initial_index(directory: Path, knowledge_root: Path) -> str:
    relative = directory.relative_to(knowledge_root)
    if directory == knowledge_root:
        return '---\nokf_version: "0.2"\n---\n\n# 知识正文\n\nDomain 入口保持完整。\n'
    if relative.parts == ("ops",):
        return (
            "# 算子知识\n\n"
            "按 DSL 隔离语言、API、编译器和实现约束；只有真正不依赖单一 DSL 的结论进入 shared。\n"
        )
    if relative.parts == ("ops", "ascendc"):
        return (
            "# Ascend C 算子知识\n\n"
            "Ascend C 的 API 参考、开发指南、算子实践和调试调优。\n"
        )
    if relative.parts == ("ops", "ascendc", "apis"):
        return "# Ascend C API\n"
    if relative.parts == ("ops", "ascendc", "guides"):
        return "# Ascend C 开发指南\n"
    return f"# {section_title(directory)}\n"


def synchronized_index(directory: Path, knowledge_root: Path) -> str:
    index = directory / "index.md"
    existing = (
        index.read_text(encoding="utf-8")
        if index.is_file() and index.read_text(encoding="utf-8").strip()
        else initial_index(directory, knowledge_root)
    )
    entries = desired_entries(directory)
    desired = dict(entries)
    lines = existing.rstrip().splitlines()
    reference_definitions = "\n".join(
        line for line in lines if REFERENCE_DEFINITION_LINE_RE.match(line)
    )
    preserved: list[str] = []
    seen: set[str] = set()
    first_navigation: int | None = None
    after_navigation: int | None = None
    relative = directory.relative_to(knowledge_root)
    refresh_mirror_entries = relative.parts[:3] in {
        ("ops", "ascendc", "apis"),
        ("ops", "ascendc", "guides"),
    }
    for line in lines:
        target = direct_entry_target(line, reference_definitions)
        if target is None:
            preserved.append(line)
            continue
        if first_navigation is None:
            first_navigation = len(preserved)
        if target in desired and target not in seen:
            preserved.append(
                desired[target]
                if refresh_mirror_entries
                else refreshed_entry(line, desired[target])
            )
            seen.add(target)
            after_navigation = len(preserved)

    missing = [line for target, line in entries if target not in seen]
    if missing:
        insert_at = after_navigation
        if insert_at is None:
            insert_at = (
                first_navigation if first_navigation is not None else len(preserved)
            )
        if insert_at and preserved[insert_at - 1].strip():
            missing.insert(0, "")
        if insert_at < len(preserved) and preserved[insert_at].strip():
            missing.append("")
        preserved[insert_at:insert_at] = missing
    return "\n".join(preserved).rstrip() + "\n"


def index_updates(root: Path) -> tuple[int, list[IndexUpdate]]:
    knowledge_root = (root / "knowledge").resolve()
    directories = [
        knowledge_root,
        *sorted(path for path in knowledge_root.rglob("*") if path.is_dir()),
    ]
    stale: list[IndexUpdate] = []
    for directory in directories:
        expected = synchronized_index(directory, knowledge_root)
        index = directory / "index.md"
        actual = index.read_text(encoding="utf-8") if index.is_file() else None
        if actual != expected:
            stale.append(IndexUpdate(index, expected))
    return len(directories), stale
