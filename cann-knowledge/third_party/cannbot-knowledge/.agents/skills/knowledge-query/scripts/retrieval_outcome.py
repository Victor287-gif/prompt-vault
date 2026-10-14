#!/usr/bin/env python3
"""Validate hash-bound outcomes for a ready retrieval state."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retrieval.root import (  # noqa: E402
    Audit,
    atomic_create_json,
    nonempty,
    read_json,
    sha256_file,
)


sys.dont_write_bytecode = True
MANIFEST_SCHEMA = "cannbot-retrieval-outcome.v1"
READY_SCHEMA = "cannbot-retrieval-ready.v1"
RECEIPT_SCHEMA = "cannbot-retrieval-outcome-receipt.v1"
OUTCOMES = {"verified", "rejected", "inconclusive", "not_run"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_manifest(path: Path, require_complete: bool) -> tuple[dict[str, Any], Audit]:
    audit = Audit()
    manifest = read_json(path, audit, "manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        audit.error("manifest.schema", "manifest.schema_version", f"expected {MANIFEST_SCHEMA}")
    state_path = Path(str(manifest.get("state", "")))
    ready_path = Path(str(manifest.get("ready_receipt", "")))
    state = read_json(state_path, audit, "manifest.state")
    ready = read_json(ready_path, audit, "manifest.ready_receipt")
    decisions_value = state.get("decisions")
    state_statuses = {
        item.get("id"): item.get("status")
        for item in decisions_value
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(decisions_value, list) else {}
    state_receipts = {
        receipt
        for item in decisions_value
        if isinstance(item, dict) and isinstance(item.get("receipts"), list)
        for receipt in item["receipts"]
        if isinstance(receipt, str)
    } if isinstance(decisions_value, list) else set()
    state_sha = sha256_file(state_path) if state_path.is_file() else ""
    ready_sha = sha256_file(ready_path) if ready_path.is_file() else ""
    if manifest.get("state_sha256") != state_sha:
        audit.error("manifest.state", "manifest.state_sha256", "does not match state content")
    if manifest.get("ready_receipt_sha256") != ready_sha:
        audit.error(
            "manifest.ready",
            "manifest.ready_receipt_sha256",
            "does not match ready receipt content",
        )
    if ready.get("schema_version") != READY_SCHEMA:
        audit.error("ready.schema", "ready.schema_version", f"expected {READY_SCHEMA}")
    if ready.get("state_sha256") != state_sha:
        audit.error("ready.state", "ready.state_sha256", "does not bind the manifest state")
    if ready.get("state") != str(state_path):
        audit.error("ready.state_path", "ready.state", "does not identify the manifest state")
    ready_statuses = ready.get("decision_statuses")
    if ready_statuses != state_statuses or not state_statuses or any(
        value != "covered" for value in state_statuses.values()
    ):
        audit.error(
            "ready.decisions",
            "ready.decision_statuses",
            "must equal the covered decisions in state",
        )
    receipt_sha256s = ready.get("receipt_sha256s")
    if not isinstance(receipt_sha256s, dict) or not receipt_sha256s:
        audit.error(
            "ready.receipts",
            "ready.receipt_sha256s",
            "must bind at least one search receipt",
        )
        receipt_sha256s = {}
    if set(receipt_sha256s) != state_receipts:
        audit.error(
            "ready.receipt_set",
            "ready.receipt_sha256s",
            "must bind exactly the search receipts declared by state",
        )
    for receipt_value, declared in receipt_sha256s.items():
        location = f"ready.receipt_sha256s[{receipt_value!r}]"
        receipt_path = Path(str(receipt_value))
        if not receipt_path.is_absolute() or not receipt_path.is_file():
            audit.error("ready.receipt_path", location, "must identify an existing absolute file")
            continue
        if not isinstance(declared, str) or not SHA256_RE.fullmatch(declared):
            audit.error("ready.receipt_sha256", location, "must be lowercase SHA-256")
        elif sha256_file(receipt_path) != declared:
            audit.error("ready.receipt_sha256", location, "does not match the search receipt")

    decision_ids = {
        item.get("id")
        for item in decisions_value
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(decisions_value, list) else set()
    events = manifest.get("events")
    if not isinstance(events, list):
        audit.error("event.list", "manifest.events", "must be a list")
        events = []
    seen: set[str] = set()
    outcomes: dict[str, str] = {}
    for index, event in enumerate(events):
        location = f"manifest.events[{index}]"
        if not isinstance(event, dict):
            audit.error("event.object", location, "must be an object")
            continue
        decision = nonempty(event.get("decision"), audit, location + ".decision")
        if decision not in decision_ids:
            audit.error("event.decision", location + ".decision", "unknown decision")
        if decision in seen:
            audit.error("event.duplicate", location + ".decision", "duplicate outcome event")
        seen.add(decision)
        outcome = nonempty(event.get("outcome"), audit, location + ".outcome")
        if outcome not in OUTCOMES:
            audit.error(
                "event.outcome",
                location + ".outcome",
                f"must be one of: {', '.join(sorted(OUTCOMES))}",
            )
        outcomes[decision] = outcome
        nonempty(event.get("summary"), audit, location + ".summary")
        observed = event.get("observed")
        if not isinstance(observed, list) or not all(
            isinstance(item, str) and item.strip() for item in observed
        ):
            audit.error("event.observed", location + ".observed", "must be a non-empty string list")
        elif outcome in {"verified", "rejected", "inconclusive"} and not observed:
            audit.error("event.observed", location + ".observed", f"{outcome} needs observed signals")
        artifacts = event.get("artifacts")
        if not isinstance(artifacts, list):
            audit.error("artifact.list", location + ".artifacts", "must be a list")
            artifacts = []
        if outcome in {"verified", "rejected"} and not artifacts:
            audit.error("artifact.required", location + ".artifacts", f"{outcome} needs artifacts")
        for artifact_index, artifact in enumerate(artifacts):
            artifact_location = f"{location}.artifacts[{artifact_index}]"
            if not isinstance(artifact, dict):
                audit.error("artifact.object", artifact_location, "must be an object")
                continue
            artifact_path = Path(str(artifact.get("path", "")))
            declared = str(artifact.get("sha256", ""))
            if not artifact_path.is_absolute() or not artifact_path.is_file():
                audit.error("artifact.path", artifact_location + ".path", "must be an existing absolute file")
            if not SHA256_RE.fullmatch(declared):
                audit.error("artifact.sha256", artifact_location + ".sha256", "must be lowercase SHA-256")
            elif artifact_path.is_file() and sha256_file(artifact_path) != declared:
                audit.error("artifact.sha256", artifact_location + ".sha256", "does not match artifact")
    if require_complete and seen != decision_ids:
        missing = sorted(decision_ids - seen)
        audit.error("event.complete", "manifest.events", f"missing decisions: {', '.join(missing)}")
    details = {
        "manifest_sha256": sha256_file(path) if path.is_file() else "",
        "state_sha256": state_sha,
        "ready_receipt_sha256": ready_sha,
        "outcomes": outcomes,
        "complete": seen == decision_ids and bool(decision_ids),
    }
    return details, audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="retrieval_outcome",
        allow_abbrev=False,
        description=__doc__,
    )
    result.add_argument("manifest")
    result.add_argument("--receipt")
    result.add_argument("--require-complete", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(sys.argv[1:] if argv is None else argv)
        manifest_path = Path(args.manifest).expanduser().resolve()
        details, audit = validate_manifest(manifest_path, args.require_complete)
        if args.receipt:
            if audit.errors:
                audit.error("receipt.invalid", "--receipt", "outcome receipt requires a valid manifest")
            else:
                atomic_create_json(
                    Path(args.receipt),
                    {"schema_version": RECEIPT_SCHEMA, **details},
                )
        payload = {"ok": not audit.errors, "errors": audit.errors, **details}
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
