#!/usr/bin/env python3
"""Validate decision-scoped retrieval state and optionally create a ready receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retrieval.index import load_cards, load_contract, open_index  # noqa: E402
from retrieval.root import (  # noqa: E402
    Audit,
    atomic_create_json,
    canonical_json_sha256,
    extract_knowledge_root,
    nonempty,
    read_json,
    resolve_root,
    sha256_file,
)
from knowledge_query import search_args_from_request, search_payload  # noqa: E402


sys.dont_write_bytecode = True
STATE_SCHEMA = "cannbot-retrieval-state.v1"
REVIEW_SCHEMA = "cannbot-retrieval-review.v1"
READY_SCHEMA = "cannbot-retrieval-ready.v1"
RECEIPT_SCHEMA = "cannbot-knowledge-search.v3"
SEARCH_SCHEMA = "knowledge-search.v5"
DECISION_STATUSES = {"covered", "conflicting", "insufficient", "gap"}
CURRENT_INDEX_FIELDS = (
    "schema_version",
    "index_version",
    "content_fingerprint",
    "governance_fingerprint",
    "card_count",
    "posting_count",
)
REPLAY_FIELDS = (
    "schema_version",
    "request_sha256",
    "request",
    "query",
    "match",
    "filters",
    "limit",
    "count",
    "results",
    "hits",
)


def string_list(value: Any, audit: Audit, location: str) -> list[str]:
    if not isinstance(value, list):
        audit.error("value.list", location, "must be a list")
        return []
    result = []
    for index, item in enumerate(value):
        result.append(nonempty(item, audit, f"{location}[{index}]"))
    if len(result) != len(set(result)):
        audit.error("value.duplicate", location, "must not contain duplicates")
    return [item for item in result if item]


def validate_state(
    root: Path,
    state_path: Path,
    review_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Audit, bool]:
    audit = Audit()
    state = read_json(state_path, audit, "state")
    connection, index_metadata = open_index(root, full=True)
    indexed_cards = load_cards(connection)
    cards = {card.path: card for card in indexed_cards}
    if state.get("schema_version") != STATE_SCHEMA:
        audit.error("state.schema", "state.schema_version", f"expected {STATE_SCHEMA}")

    target = state.get("target")
    if not isinstance(target, dict):
        audit.error("target.object", "state.target", "must be an object")
        target = {}
    nonempty(target.get("entity"), audit, "state.target.entity")
    platforms = string_list(target.get("platforms"), audit, "state.target.platforms")
    contract, _ = load_contract(root)
    known_platforms = set(contract.platforms)
    unknown = sorted(set(platforms) - known_platforms)
    if unknown:
        audit.error(
            "target.platform",
            "state.target.platforms",
            f"unknown platforms: {', '.join(unknown)}",
        )
    if not platforms:
        audit.error("target.platform", "state.target.platforms", "at least one platform is required")

    decisions_value = state.get("decisions")
    if not isinstance(decisions_value, list) or not decisions_value:
        audit.error("decision.list", "state.decisions", "must be a non-empty list")
        decisions_value = []
    decision_ids: set[str] = set()
    decision_statuses: dict[str, str] = {}
    receipt_sha256s: dict[str, str] = {}
    for index, decision in enumerate(decisions_value):
        location = f"state.decisions[{index}]"
        if not isinstance(decision, dict):
            audit.error("decision.object", location, "must be an object")
            continue
        decision_id = nonempty(decision.get("id"), audit, location + ".id")
        nonempty(decision.get("question"), audit, location + ".question")
        if decision_id in decision_ids:
            audit.error("decision.duplicate", location + ".id", f"duplicate decision: {decision_id}")
        decision_ids.add(decision_id)
        status = nonempty(decision.get("status"), audit, location + ".status")
        if status not in DECISION_STATUSES:
            audit.error(
                "decision.status",
                location + ".status",
                f"must be one of: {', '.join(sorted(DECISION_STATUSES))}",
            )
        decision_statuses[decision_id] = status

        receipts = string_list(decision.get("receipts"), audit, location + ".receipts")
        if status != "gap" and not receipts:
            audit.error("decision.receipts", location + ".receipts", "non-gap decision needs a receipt")
        receipt_evidence: dict[str, str] = {}
        for receipt_index, receipt_value in enumerate(receipts):
            receipt_path = Path(receipt_value)
            receipt_location = f"{location}.receipts[{receipt_index}]"
            receipt = read_json(receipt_path, audit, receipt_location)
            if receipt_path.is_absolute() and receipt_path.is_file():
                receipt_sha256s[str(receipt_path)] = sha256_file(receipt_path)
            if receipt.get("schema_version") != RECEIPT_SCHEMA:
                audit.error("receipt.schema", receipt_location, f"expected {RECEIPT_SCHEMA}")
                continue
            request = receipt.get("request")
            if not isinstance(request, dict):
                audit.error("receipt.request", receipt_location + ".request", "must be an object")
                request = {}
            request_sha256 = canonical_json_sha256(request)
            if receipt.get("request_sha256") != request_sha256:
                audit.error(
                    "receipt.request_hash",
                    receipt_location + ".request_sha256",
                    "does not match the canonical request content",
                )
            if request.get("decision") != decision_id:
                audit.error(
                    "receipt.decision",
                    receipt_location,
                    "receipt request.decision must equal the decision id",
                )
            search_args = None
            try:
                search_args = search_args_from_request(request)
            except ValueError as exc:
                audit.error("receipt.request_shape", receipt_location + ".request", str(exc))
            if search_args is not None and set(search_args.platform) != set(platforms):
                audit.error(
                    "receipt.platform",
                    receipt_location + ".request.platforms",
                    "receipt target platforms must equal state.target.platforms",
                )
            if receipt.get("knowledge_root") != str(root):
                audit.error(
                    "receipt.root",
                    receipt_location + ".knowledge_root",
                    "does not match the current knowledge root",
                )
            index_value = receipt.get("index")
            if not isinstance(index_value, dict):
                audit.error("receipt.index", receipt_location, "receipt index provenance is missing")
            else:
                for field in CURRENT_INDEX_FIELDS:
                    if index_value.get(field) != index_metadata.get(field):
                        audit.error(
                            "receipt.stale",
                            receipt_location + ".index." + field,
                            "receipt does not match the current prepared index",
                        )
                if index_value.get("built_repository_state") != index_metadata.get(
                    "repository_state"
                ):
                    audit.error(
                        "receipt.stale",
                        receipt_location + ".index.built_repository_state",
                        "receipt does not match the index build repository state",
                    )
            result = receipt.get("result")
            if not isinstance(result, dict):
                audit.error("receipt.result", receipt_location + ".result", "must be an object")
                result = {}
            if result.get("schema_version") != SEARCH_SCHEMA:
                audit.error(
                    "receipt.result_schema",
                    receipt_location + ".result.schema_version",
                    f"expected {SEARCH_SCHEMA}",
                )
            if result.get("request") != request:
                audit.error(
                    "receipt.result_request",
                    receipt_location + ".result.request",
                    "does not match the receipt request",
                )
            if result.get("request_sha256") != request_sha256:
                audit.error(
                    "receipt.result_request_hash",
                    receipt_location + ".result.request_sha256",
                    "does not match the canonical request content",
                )
            if result.get("index") != index_value:
                audit.error(
                    "receipt.result_index",
                    receipt_location + ".result.index",
                    "does not match the receipt index provenance",
                )
            results = result.get("results")
            hits_alias = result.get("hits")
            if not isinstance(results, list):
                audit.error("receipt.results", receipt_location + ".result.results", "must be a list")
                results = []
            if hits_alias != results:
                audit.error(
                    "receipt.hits_alias",
                    receipt_location + ".result.hits",
                    "must equal result.results",
                )
            if result.get("count") != len(results):
                audit.error(
                    "receipt.result_count",
                    receipt_location + ".result.count",
                    "does not match the number of results",
                )
            if search_args is not None:
                try:
                    replayed = search_payload(
                        root,
                        indexed_cards,
                        index_metadata,
                        connection,
                        search_args,
                        include_current_repository=False,
                    )
                except ValueError as exc:
                    audit.error("receipt.replay", receipt_location + ".result", str(exc))
                else:
                    for field in REPLAY_FIELDS:
                        if result.get(field) != replayed.get(field):
                            audit.error(
                                "receipt.replay",
                                receipt_location + ".result." + field,
                                "does not match a deterministic replay against the current index",
                            )
            for hit in results:
                if isinstance(hit, dict) and isinstance(hit.get("path"), str):
                    receipt_evidence[hit["path"]] = str(hit.get("card_sha256", ""))

        evidence = decision.get("evidence")
        if not isinstance(evidence, list):
            audit.error("evidence.list", location + ".evidence", "must be a list")
            evidence = []
        if status in {"covered", "conflicting"} and not evidence:
            audit.error("evidence.required", location + ".evidence", f"{status} decision needs evidence")
        for evidence_index, item in enumerate(evidence):
            item_location = f"{location}.evidence[{evidence_index}]"
            if not isinstance(item, dict):
                audit.error("evidence.object", item_location, "must be an object")
                continue
            path = nonempty(item.get("path"), audit, item_location + ".path")
            fingerprint = nonempty(
                item.get("card_sha256"), audit, item_location + ".card_sha256"
            )
            if receipt_evidence.get(path) != fingerprint:
                audit.error(
                    "evidence.receipt",
                    item_location,
                    "evidence path and fingerprint must occur in a bound search receipt",
                )
            current = cards.get(path)
            if current is None:
                audit.error("evidence.path", item_location + ".path", "card no longer exists")
            elif current.card_sha256 != fingerprint:
                audit.error(
                    "evidence.stale",
                    item_location + ".card_sha256",
                    "card content changed after retrieval",
                )

    state_sha256 = sha256_file(state_path) if state_path.is_file() else ""
    review: dict[str, Any] = {}
    if review_path is not None:
        review = read_json(review_path, audit, "review")
        if review.get("schema_version") != REVIEW_SCHEMA:
            audit.error("review.schema", "review.schema_version", f"expected {REVIEW_SCHEMA}")
        if review.get("state_sha256") != state_sha256:
            audit.error("review.state", "review.state_sha256", "does not match state content")
        if review.get("verdict") not in {"approve", "reject"}:
            audit.error("review.verdict", "review.verdict", "must be approve or reject")
        if review.get("verdict") == "reject":
            audit.error("review.reject", "review.verdict", "review rejected the retrieval state")

    ready = not audit.errors and bool(decision_statuses) and all(
        status == "covered" for status in decision_statuses.values()
    )
    details = {
        "state_sha256": state_sha256,
        "decision_statuses": decision_statuses,
        "index": {
            **{field: index_metadata[field] for field in CURRENT_INDEX_FIELDS},
            "built_repository_state": index_metadata.get("repository_state"),
        },
        "review_sha256": sha256_file(review_path) if review_path and review_path.is_file() else None,
        "receipt_sha256s": dict(sorted(receipt_sha256s.items())),
    }
    connection.close()
    return state, details, audit, ready


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="retrieval_state",
        allow_abbrev=False,
        description=__doc__,
        epilog="Global root option accepted anywhere: --knowledge-root PATH.",
    )
    result.add_argument("--knowledge-root", help=argparse.SUPPRESS)
    result.add_argument("state")
    result.add_argument("--review")
    result.add_argument("--receipt")
    result.add_argument("--require-ready", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        cleaned, explicit = extract_knowledge_root(sys.argv[1:] if argv is None else argv)
        args = parser().parse_args(cleaned)
        root = resolve_root(explicit, __file__)
        state_path = Path(args.state).expanduser().resolve()
        review_path = Path(args.review).expanduser().resolve() if args.review else None
        _, details, audit, ready = validate_state(root, state_path, review_path)
        if args.require_ready and not ready and not audit.errors:
            audit.error("state.not_ready", "state.decisions", "all decisions must be covered")
        if args.receipt:
            if not ready or audit.errors:
                audit.error("receipt.not_ready", "--receipt", "ready receipt requires a ready state")
            else:
                atomic_create_json(
                    Path(args.receipt),
                    {
                        "schema_version": READY_SCHEMA,
                        "state": str(state_path),
                        **details,
                    },
                )
        payload = {
            "ok": not audit.errors,
            "ready": ready and not audit.errors,
            "errors": audit.errors,
            "warnings": [],
            **details,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not audit.errors else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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
