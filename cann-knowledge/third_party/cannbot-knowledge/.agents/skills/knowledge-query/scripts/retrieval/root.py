"""Knowledge-root resolution and audit-safe file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from html import escape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


ENV_KEY = "CANNBOT_KNOWLEDGE_ROOT"
PLATFORM_ENV_KEY = "CANNBOT_KNOWLEDGE_PLATFORM"


def local_markdown_link(title: str, path: str | Path) -> str:
    """Format a selected local file without turning path characters into URL syntax."""
    label = escape(" ".join(title.splitlines()), quote=False)
    label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return f"[{label}](<{quote(str(path), safe='/ ')}>)"


class Audit:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []

    def error(self, code: str, location: str, message: str) -> None:
        self.errors.append({"code": code, "location": location, "message": message})


def read_json(path: Path, audit: Audit, location: str) -> dict[str, Any]:
    if not path.is_absolute():
        audit.error("path.absolute", location, "path must be absolute")
        return {}
    if not path.is_file():
        audit.error("path.exists", location, f"file does not exist: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        audit.error("json.invalid", location, str(exc))
        return {}
    if not isinstance(value, dict):
        audit.error("json.object", location, "top-level JSON must be an object")
        return {}
    return value


def nonempty(value: Any, audit: Audit, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        audit.error("value.string", location, "must be a non-empty string")
        return ""
    return value.strip()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_knowledge_root(argv: Iterable[str]) -> tuple[list[str], str | None]:
    """Remove a global --knowledge-root option accepted anywhere in argv."""
    values = list(argv)
    cleaned: list[str] = []
    roots: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--knowledge-root":
            if index + 1 >= len(values):
                raise ValueError("--knowledge-root requires a path")
            roots.append(values[index + 1])
            index += 2
            continue
        if value.startswith("--knowledge-root="):
            roots.append(value.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(value)
        index += 1
    if any(not value.strip() for value in roots):
        raise ValueError("--knowledge-root must not be empty")
    if len(set(roots)) > 1:
        raise ValueError("conflicting --knowledge-root values")
    return cleaned, roots[0] if roots else None


def _project_env_value(path: Path, key: str = ENV_KEY) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read knowledge-root config {path}: {exc}") from exc
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid shell syntax in {path}: {exc}") from exc
        if tokens and tokens[0] == "export":
            tokens = tokens[1:]
        for token in tokens:
            if token.startswith(f"{key}="):
                return token.split("=", 1)[1]
    return None


def project_env_value(key: str) -> str | None:
    """Return a project config value, preferring the process environment."""
    configured = os.environ.get(key)
    if configured:
        return configured
    for parent in (Path.cwd(), *Path.cwd().parents):
        env_file = parent / ".cannbot" / "knowledge.env"
        if env_file.is_file():
            value = _project_env_value(env_file, key)
            if value:
                return value
            break
    return None


def resolve_root(explicit: str | None, script_file: str | Path) -> Path:
    def validated(source: str, candidate: Path) -> Path:
        root = candidate.expanduser().resolve()
        if (root / "knowledge").is_dir() and (
            root / "governance/schemas/registries.yaml"
        ).is_file():
            return root
        raise ValueError(f"invalid cannbot-knowledge root from {source}: {root}")

    if explicit:
        return validated("--knowledge-root", Path(explicit))
    if os.environ.get(ENV_KEY):
        return validated(ENV_KEY, Path(os.environ[ENV_KEY]))
    for parent in (Path.cwd(), *Path.cwd().parents):
        env_file = parent / ".cannbot" / "knowledge.env"
        if env_file.is_file():
            value = _project_env_value(env_file)
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = env_file.parent / candidate
                return validated(str(env_file), candidate)
            break
    return validated("skill location", Path(script_file).resolve().parents[4])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def atomic_create_bytes(target: Path, payload: bytes) -> None:
    """Atomically create target and refuse to replace any existing entry."""
    target = target.expanduser()
    # Resolve parents only: resolving the leaf would follow a dangling symlink
    # and create its referent instead of rejecting the existing directory entry.
    target = target.parent.resolve() / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ValueError(f"refusing to overwrite receipt: {target}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def atomic_create_json(target: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    atomic_create_bytes(target, encoded)
