#!/usr/bin/env python3
"""Read a selected knowledge card or resolve its local resources read-only."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
import shlex
import sys
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
QUERY_SCRIPTS = Path(__file__).resolve().parents[2] / "knowledge-query/scripts"
sys.path.insert(0, str(QUERY_SCRIPTS))

import yaml  # noqa: E402

from retrieval.index import load_contract  # noqa: E402
from retrieval.root import local_markdown_link, resolve_root  # noqa: E402


PAGE_CHAR_BUDGET = 10000
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")


def selected_card(root: Path, card: Path) -> Path:
    if not card.is_absolute():
        raise ValueError("card must be the absolute local_path returned by Query or a resolved knowledge link")
    card = card.resolve(strict=True)
    if not card.is_relative_to(root / "knowledge") or card.suffix != ".md" or not card.is_file():
        raise ValueError("card must be an existing Markdown file inside this repository's knowledge/ Bundle")
    return card


def read_card_page(root: Path, card: Path, start_line: int = 1, expected_sha256: str | None = None) -> str:
    """Emit unchanged numbered lines; EOF alone does not prove earlier pages were read."""
    root = root.resolve(strict=True)
    card = selected_card(root, card)
    if card.name == "index.md":
        raise ValueError("index.md is navigation; select a knowledge card before using read")
    _, split_frontmatter = load_contract(root)
    raw = card.read_bytes()
    text = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("card changed since the preceding page; restart reading from line 1")
    if start_line != 1 and expected_sha256 is None:
        raise ValueError("continuation requires the file_sha256 returned by the first page")
    metadata, _ = split_frontmatter(text)
    title = (metadata or {}).get("title")
    if not isinstance(title, str) or not title.strip():
        title = card.stem
    lines = text.splitlines()
    if not 1 <= start_line <= len(lines):
        raise ValueError("start-line must be within the selected card")
    navigation = []
    selected = []
    size = 0
    for number in range(start_line, len(lines) + 1):
        role = "导航" if any(first <= number <= last for first, last in navigation) else "卡片"
        line = f"{number}\t{role}\t{lines[number - 1]}"
        if selected and size + len(line) + 1 > PAGE_CHAR_BUDGET:
            break
        selected.append(line)
        size += len(line) + 1
    end_line = start_line + len(selected) - 1
    framing = {
        "line_format": "number-tab-role-tab-text",
        "card": str(card), "file_sha256": digest, "card_lines": len(lines),
        "start_line": start_line,
        "navigation_ranges": navigation,
    }
    footer = {
        "end_line": end_line,
        "end_of_file": end_line == len(lines),
        "next_start_line": end_line + 1 if end_line < len(lines) else None,
        "next_command": None,
        "file_sha256": digest,
    }
    if footer["next_start_line"] is not None:
        continuation = shlex.join([
            "python3", str(Path(__file__).resolve()), "read", "--knowledge-root", str(root),
            "--card", str(card), "--start-line", str(footer["next_start_line"]), "--sha256", digest,
        ])
        footer["next_command"] = continuation
        progress = [
            f"本卡尚未读完：已返回 {start_line}–{end_line} / {len(lines)} 行。本页不足以采用这张卡或转去其源码。",
            "继续阅读：" + continuation,
        ]
    else:
        progress = [
            "本页已到末尾；只有同一指纹的第 1 行至末行均已连续读过，才能采用此卡。",
            "采用前检查：只回答当前子问题。未打开的相关卡不列入结论或‘进一步印证’；导航的展示数量/扫描范围可按用户问题说明，但导航的技术理由不是目标卡证据。",
            "跨卡引用赋值或公式时，用下列命令对照原始字段；把每个实际赋值按定义单位换算一次，再判断是否一致。只核对定性含义还不够，不能把两个不同量纲的式子都当作正确依据。",
            "字段对照命令：" + shlex.join([
                "python3", str(Path(__file__).resolve()), "compare", "--knowledge-root", str(root),
                "--card", str(card), "--card", "<另一张已读卡绝对路径>", "--field", "<准确字段名>",
            ]),
            "需要相关知识、源码或 include 时，按补证参考继续：" + str(Path(__file__).resolve().parents[1] / "references/evidence-check.md"),
            "引用此卡：" + local_markdown_link(title, card),
        ]
    return "\n".join([
        "CARD_PAGE " + json.dumps(framing, ensure_ascii=False),
        *selected,
        "END_CARD_PAGE " + json.dumps(footer),
        *progress,
    ])


def outline_card(root: Path, card: Path) -> str:
    """Emit the heading skeleton of a selected card for triage, not adoption."""
    root = root.resolve(strict=True)
    card = selected_card(root, card)
    if card.name == "index.md":
        raise ValueError("index.md is navigation; select a knowledge card before using outline")
    _, split_frontmatter = load_contract(root)
    raw = card.read_bytes()
    text = raw.decode("utf-8")
    metadata, body = split_frontmatter(text)
    title = (metadata or {}).get("title")
    if not isinstance(title, str) or not title.strip():
        title = card.stem
    lines = text.splitlines()
    normalized = "\n".join(lines)
    if body:
        body_start = normalized[: normalized.rfind(body)].count("\n") + 1
    else:
        body_start = len(lines) + 1
    headings: list[tuple[int, int, str]] = []
    fenced = False
    for number in range(body_start, len(lines) + 1):
        line = lines[number - 1]
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = HEADING_RE.match(line)
        if match is not None:
            headings.append((number, len(match.group(1)), match.group(2)))
    bundle = root / "knowledge"
    framing = {
        "card": card.relative_to(bundle).as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "card_lines": len(lines), "title": title,
        "body_start_line": body_start, "headings": len(headings),
        "heading_style": "atx",
    }
    footer = {
        "outline_only": True,
        "read_command": shlex.join([
            "python3", str(Path(__file__).resolve()), "read", "--knowledge-root", str(root),
            "--card", str(card),
        ]),
        "adoption_rule": "outline 仅用于判断相关性和定位章节；采用本卡前仍须按 read 在同一指纹下连续读完全部页面。",
    }
    return "\n".join([
        "CARD_OUTLINE " + json.dumps(framing, ensure_ascii=False),
        *(f"{number}\t{'#' * level} {heading}" for number, level, heading in headings),
        "END_CARD_OUTLINE " + json.dumps(footer, ensure_ascii=False),
    ])


def compare_card_field(root: Path, cards: list[Path], field: str) -> dict:
    """Juxtapose literal field context from already selected cards, excluding frontmatter."""
    if not field or field != field.strip() or any(ord(char) < 32 for char in field):
        raise ValueError("field must be a nonempty literal name without control characters")
    root = root.resolve(strict=True)
    _, split_frontmatter = load_contract(root)
    resolved = list(dict.fromkeys(selected_card(root, card) for card in cards))
    if len(resolved) < 2 or any(card.name == "index.md" for card in resolved):
        raise ValueError("compare requires at least two distinct selected knowledge cards, not navigation")
    results = []
    for card in resolved:
        raw = card.read_bytes()
        original_lines = raw.decode("utf-8").splitlines()
        text = "\n".join(original_lines)
        metadata, body = split_frontmatter(text)
        # The returned body is a stripped suffix. Work in normalized lines, but emit original lines.
        body_start = text[:text.rfind(body)].count("\n") + 1 if body else len(original_lines) + 1
        navigation = {number for first, last in [] for number in range(first, last + 1)}
        selected_lines = set()
        for number in range(body_start, len(original_lines) + 1):
            if number not in navigation and field in original_lines[number - 1]:
                selected_lines.update(range(max(body_start, number - 4), min(len(original_lines), number + 4) + 1))
        snippets = [f"{number}\t{original_lines[number - 1]}" for number in sorted(selected_lines - navigation)]
        results.append({
            "card": str(card), "file_sha256": hashlib.sha256(raw).hexdigest(),
            "platforms": (metadata or {}).get("platforms"), "status": (metadata or {}).get("status"),
            "numbered_context": snippets,
        })
    return {
        "field": field, "partial_context_only": True, "cards": results,
        "adoption_check": "将每张卡的原始定义、单位和实际赋值放在同一字段下核对；冲突时明确指出，不照搬两份互斥结论。片段不能代替此前的全文阅读。",
    }


def resolve_headers(package_root: Path, headers: list[str]) -> dict:
    if not package_root.is_absolute():
        raise ValueError("package-root must be an absolute CANN Ascend C component path")
    root = package_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"package-root is not a directory: {root}")
    if not headers:
        raise ValueError("select at least one source_headers entry from the API card")
    resolved = []
    for header in dict.fromkeys(headers):
        parts = header.split("/")
        if (
            len(parts) < 2 or parts[0] != "include"
            or any(part in {"", ".", ".."} for part in parts)
            or header != header.strip()
            or any(char in header for char in "\\?#")
            or any(ord(char) < 32 for char in header)
        ):
            raise ValueError(f"expected a safe package-relative include/... path: {header!r}")
        target = root.joinpath(*parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"header escapes the confirmed package-root: {header!r}")
        if not target.is_file():
            raise ValueError(f"header file missing: {target}; confirm the package root and CANN version")
        resolved.append({"relative": header, "absolute_path": str(target)})
    return {
        "package_root": str(root),
        "headers": resolved,
        "version_evidence": "not checked; a directory name or another component's metadata is not version evidence",
    }


def resolve_card_resource(root: Path, card: Path, value: str, *, source: bool) -> dict:
    """Resolve exactly one declared target, never search for substitute files."""
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError("target must be a nonempty declared path without control characters")
    root = root.resolve(strict=True)
    bundle = root / "knowledge"
    card = selected_card(root, card)
    contract, split_frontmatter = load_contract(root)
    module = importlib.import_module(f"{type(contract).__module__.rsplit('.', 1)[0]}.source_resources")
    metadata, body = split_frontmatter(card.read_text(encoding="utf-8"))
    anchor = ""
    pinned = None
    if source:
        sources = (metadata or {}).get("sources") or []
        if not isinstance(sources, list):
            raise ValueError("selected card sources must be an array")
        resources = [item.get("resource") for item in sources if isinstance(item, dict)]
        if value not in resources:
            raise ValueError("source must exactly match a sources[].resource entry in the selected card")
        pinned = module.git_source(contract, value)
        if pinned is not None:
            target = pinned.local_path(root)
        else:
            definition = contract.local_source_definition(value)
            if definition is None or not module.safe_path(value):
                raise ValueError("source is not a registered local resource; no network lookup was attempted")
            target = (root / value).resolve()
            if not target.is_relative_to(root / definition["root"]):
                raise ValueError("source escapes its registered root")
    else:
        knowledge = importlib.import_module(type(contract).__module__)
        targets = {target for target, is_image in knowledge.markdown_targets(body) if not is_image}
        if value not in targets:
            raise ValueError("link must exactly match a Markdown target in the selected card")
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or parsed.query:
            raise ValueError("link must be local to the Bundle or a registered source; no network lookup was attempted")
        if parsed.path:
            links = importlib.import_module(
                f"{type(contract).__module__.rsplit('.', 1)[0]}.checks.links"
            )
            targets_by_value = {
                raw: target for raw, target, _ in links.links_in_text(body, card, bundle)
            }
            target = targets_by_value[value]
        else:
            target = card
        anchor = parsed.fragment
        if not target.is_relative_to(bundle):
            pinned = module.local_source_link(contract, card, value)
            if pinned is None:
                raise ValueError("link leaves the Bundle without a registered pinned source")
            target = pinned.local_path(root)
        elif target.suffix != ".md":
            raise ValueError("local Bundle links must point to Markdown")
    if not target.exists():
        raise ValueError(f"selected resource is unavailable at its resolved absolute path: {target}")
    if not target.is_file() and not target.is_dir():
        raise ValueError(f"selected resource is not a regular file or directory: {target}")
    if not source and not target.is_file():
        raise ValueError(f"selected link is not a file: {target}")
    if pinned is not None:
        definition = contract.local_source_definition(pinned.resource)
        module.validate_checkout(root, definition, pinned.revision)
    result = {
        "repository_root": str(root), "bundle_root": str(bundle),
        "card": str(card), "input": value, "absolute_path": str(target),
        "kind": "directory" if target.is_dir() else "file", "anchor": anchor,
    }
    if target.is_relative_to(bundle) and target.suffix == ".md" and target.name != "index.md":
        result["card_lines"] = len(target.read_text(encoding="utf-8").splitlines())
    if pinned is not None:
        result.update({"repository": pinned.repository, "revision": pinned.revision})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read", help="read the selected card in complete consecutive pages")
    read.add_argument("--knowledge-root")
    read.add_argument("--card", type=Path, required=True)
    read.add_argument("--link", help="read this exact Markdown link from --card instead of reconstructing its path")
    read.add_argument("--start-line", type=int, default=1)
    read.add_argument("--sha256", help="file_sha256 returned by the preceding page; required for continuation")
    outline = commands.add_parser("outline", help="list the heading skeleton of a selected card for triage")
    outline.add_argument("--knowledge-root")
    outline.add_argument("--card", type=Path, required=True)
    compare = commands.add_parser("compare", help="compare a literal field across already fully read cards")
    compare.add_argument("--knowledge-root")
    compare.add_argument("--card", type=Path, action="append", required=True)
    compare.add_argument("--field", required=True)
    headers = commands.add_parser("headers", help="resolve selected source_headers below a confirmed component")
    headers.add_argument("--package-root", type=Path, required=True)
    headers.add_argument("headers", nargs="+", help="selected include/... entries from source_headers")
    for name in ("source", "link"):
        selected = commands.add_parser(name, help=f"resolve one selected card's declared {name}")
        selected.add_argument("--knowledge-root")
        selected.add_argument("--card", type=Path, required=True)
        selected.add_argument("target", help="copy the original source resource or Markdown link target verbatim")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "headers":
            result = resolve_headers(args.package_root, args.headers)
        else:
            root = resolve_root(args.knowledge_root, __file__)
            if args.command == "read":
                card = args.card
                if args.link is not None:
                    resolved = resolve_card_resource(root, card, args.link, source=False)
                    card = Path(resolved["absolute_path"])
                result = read_card_page(root, card, args.start_line, args.sha256)
            elif args.command == "outline":
                result = outline_card(root, args.card)
            elif args.command == "compare":
                result = compare_card_field(root, args.card, args.field)
            else:
                result = resolve_card_resource(root, args.card, args.target, source=args.command == "source")
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
