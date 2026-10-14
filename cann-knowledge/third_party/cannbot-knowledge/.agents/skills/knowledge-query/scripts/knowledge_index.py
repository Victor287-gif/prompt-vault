#!/usr/bin/env python3
"""Build or verify the deterministic cannbot-knowledge search index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retrieval.index import (  # noqa: E402
    DEFAULT_TOKENIZER,
    TOKENIZERS,
    build_index,
    index_path,
    verify_index,
)
from retrieval.root import extract_knowledge_root, resolve_root  # noqa: E402


sys.dont_write_bytecode = True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="knowledge_index",
        allow_abbrev=False,
        description=__doc__,
        epilog="Global root option accepted anywhere: --knowledge-root PATH.",
    )
    result.add_argument("--knowledge-root", help=argparse.SUPPRESS)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", allow_abbrev=False)
    build.add_argument("--json", action="store_true")
    build.add_argument(
        "--tokenizer",
        choices=TOKENIZERS,
        default=DEFAULT_TOKENIZER,
        help="v3 keeps master retrieval; subword adds camel/separator pieces and IDF (index v4).",
    )
    verify = commands.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--level", choices=["quick", "full"], default="quick")
    verify.add_argument("--json", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        cleaned, explicit = extract_knowledge_root(sys.argv[1:] if argv is None else argv)
        args = parser().parse_args(cleaned)
        root = resolve_root(explicit, __file__)
        if args.command == "build":
            metadata = build_index(root, tokenizer=args.tokenizer)
        else:
            metadata = verify_index(root, full=args.level == "full")
        payload = {
            "ok": True,
            "command": args.command,
            "level": getattr(args, "level", None),
            "index_path": str(index_path(root)),
            "index": metadata,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(
                f"{args.command}: ok cards={metadata['card_count']} "
                f"index={index_path(root)}"
            )
        return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(
            json.dumps(
                {"error": {"type": exc.__class__.__name__, "message": str(exc)}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
