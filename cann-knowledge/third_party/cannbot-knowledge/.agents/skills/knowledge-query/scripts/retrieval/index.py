"""Persistent deterministic search index for the governed knowledge bundle."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .root import repository_state, sha256_file


TOKENIZER_V3 = "v3"
TOKENIZER_SUBWORD = "subword"
TOKENIZERS = (TOKENIZER_V3, TOKENIZER_SUBWORD)
DEFAULT_TOKENIZER = TOKENIZER_V3
# v3: whole TOKEN_RE terms + CJK bigrams, uniform posting sum (master default).
# v4: camel/separator pieces + get_ stem + BM25-style IDF; rebuild required.
INDEX_VERSION_V3 = 3
INDEX_VERSION_SUBWORD = 4
INDEX_VERSION = INDEX_VERSION_V3
INDEX_SCHEMA = "cannbot-knowledge-index.v1"
_POSTING_FIELD_SQL = (
    "4.0 * title_tf + 2.5 * description_tf + 2.0 * tags_tf"
    " + 2.0 * aliases_tf + 0.5 * MIN(body_tf, 4)"
)
INDEX_RELATIVE_PATH = Path("artifacts/indexes/knowledge.sqlite3")
TOKEN_RE = re.compile(r"[a-zA-Z0-9_+./:-]+|[\u4e00-\u9fff]{2,}")
GOVERNANCE_INPUTS = (
    "governance/schemas/frontmatter.schema.json",
    "governance/schemas/profiles.yaml",
    "governance/schemas/registries.yaml",
)


@dataclass(frozen=True)
class IndexedCard:
    doc_id: int
    path: str
    relative_path: str
    domain: str
    technology: str | None
    profile: str
    metadata: dict[str, Any]
    card_lines: int
    card_sha256: str


def index_path(root: Path) -> Path:
    return root / INDEX_RELATIVE_PATH


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(entries: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path, value in sorted(entries):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def governance_fingerprint(root: Path) -> str:
    contracts = sorted((root / "governance/contracts").rglob("*.py"))
    paths = [root / relative for relative in GOVERNANCE_INPUTS] + contracts
    entries = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            raise ValueError(f"missing index governance input: {relative}")
        entries.append((relative, sha256_file(path)))
    return _fingerprint(entries)


def card_fingerprint(metadata: dict[str, Any], body: str) -> str:
    return hashlib.sha256(
        f"{_canonical_json(metadata)}\n{body}".encode("utf-8")
    ).hexdigest()


# camel/分隔符子词切分: TOKEN_RE 的 ASCII token 把 "DataCopyPad"/"rms_norm"
# 当作整体, 查询 "copy"/"rms norm" 打不中 posting; 先在保留大小写的原文上
# 切出子词 (camel 边界 + _ / . : 等分隔符), 再小写化。全词 token 始终保留,
# 子词只保留 ≥2 字符且含字母的片段 (排除 "950"/"9" 这类纯数字噪声)。
_WORD_PIECE_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_GET_PREFIX_RE = re.compile(r"^(?:[Gg]et[_./:-]|[Gg]et(?=[A-Z]))")


def index_version_for(tokenizer: str) -> int:
    if tokenizer == TOKENIZER_SUBWORD:
        return INDEX_VERSION_SUBWORD
    if tokenizer == TOKENIZER_V3:
        return INDEX_VERSION_V3
    raise ValueError(f"unknown tokenizer: {tokenizer}")


def index_tokenizer(metadata: dict[str, Any] | None = None) -> str:
    if not metadata:
        return DEFAULT_TOKENIZER
    value = metadata.get("tokenizer", TOKENIZER_V3)
    if value not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer: {value}")
    return str(value)


def _accessor_stem(raw: str) -> str:
    """Drop a leading get_/Get camel accessor without touching words like getting."""
    match = _GET_PREFIX_RE.match(raw)
    if match is None:
        return raw
    stem = raw[match.end() :]
    return stem or raw


def _word_pieces(raw: str) -> list[str]:
    pieces: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", _accessor_stem(raw)):
        for part in _WORD_PIECE_RE.findall(chunk):
            if len(part) >= 2 and re.search(r"[A-Za-z]", part):
                pieces.append(part.lower())
    return pieces


def _v3_token_counts(value: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for token in TOKEN_RE.findall(value.lower()):
        result[token] += 1
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for offset in range(len(token) - 1):
                result[token[offset : offset + 2]] += 1
    return result


def _subword_token_counts(value: str) -> Counter[str]:
    result: Counter[str] = Counter()
    # Tokenize before lowercasing so camelCase boundaries survive.
    for raw in TOKEN_RE.findall(value):
        token = raw.lower()
        result[token] += 1
        for piece in _word_pieces(raw):
            if piece != token:
                result[piece] += 1
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for offset in range(len(token) - 1):
                result[token[offset : offset + 2]] += 1
    return result


def token_counts(value: str, tokenizer: str = DEFAULT_TOKENIZER) -> Counter[str]:
    if tokenizer == TOKENIZER_SUBWORD:
        return _subword_token_counts(value)
    if tokenizer == TOKENIZER_V3:
        return _v3_token_counts(value)
    raise ValueError(f"unknown tokenizer: {tokenizer}")


def query_tokens(value: str, tokenizer: str = DEFAULT_TOKENIZER) -> list[str]:
    return list(token_counts(value, tokenizer=tokenizer))


def searchable_body(body: str) -> str:
    """Index the card body, including ordinary knowledge links."""
    return body.strip()


def load_contract(root: Path):
    """Load this Skill's Contract after comparing the target copy without executing it."""
    source = Path(__file__).resolve().parents[5] / "governance/contracts"
    target = root / "governance/contracts"
    if not (target / "knowledge.py").is_file():
        raise ValueError(f"not a governed cannbot-knowledge root: {root}")
    if source.resolve() != target.resolve():
        for source_file in sorted(source.rglob("*.py")):
            target_file = target / source_file.relative_to(source)
            if (
                not target_file.is_file()
                or source_file.read_bytes() != target_file.read_bytes()
            ):
                raise ValueError(
                    "target governance contract does not match this knowledge-query version"
                )
    package_name = "_cannbot_knowledge_query_contracts"
    package = sys.modules.get(package_name)
    package_file = source / "__init__.py"
    if package is None:
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_file,
            submodule_search_locations=[str(source)],
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load governance contract: {source}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    elif Path(package.__file__).resolve() != package_file.resolve():
        raise ValueError("governance contract source changed in the current process")
    module = importlib.import_module(f"{package_name}.knowledge")
    return module.Contract(root), module.split_frontmatter


def _card_files(root: Path) -> list[Path]:
    return sorted(
        path for path in (root / "knowledge").rglob("*.md") if path.name != "index.md"
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE manifest (
            relative_path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            file_sha256 TEXT NOT NULL
        );
        CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY,
            concept_id TEXT NOT NULL UNIQUE,
            relative_path TEXT NOT NULL UNIQUE,
            domain TEXT NOT NULL,
            technology TEXT,
            profile TEXT NOT NULL,
            okf_type TEXT NOT NULL,
            status TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            body TEXT NOT NULL,
            search_text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            card_lines INTEGER NOT NULL,
            card_sha256 TEXT NOT NULL
        );
        CREATE TABLE postings (
            term TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            title_tf INTEGER NOT NULL,
            description_tf INTEGER NOT NULL,
            tags_tf INTEGER NOT NULL,
            aliases_tf INTEGER NOT NULL,
            body_tf INTEGER NOT NULL,
            PRIMARY KEY (term, doc_id),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        ) WITHOUT ROWID;
        CREATE INDEX documents_domain_idx ON documents(domain);
        CREATE INDEX documents_technology_idx ON documents(technology);
        CREATE INDEX documents_profile_idx ON documents(profile);
        CREATE INDEX documents_status_idx ON documents(status);
        CREATE INDEX postings_doc_idx ON postings(doc_id);
        """
    )


def build_index(root: Path, tokenizer: str = DEFAULT_TOKENIZER) -> dict[str, Any]:
    tokenizer = index_tokenizer({"tokenizer": tokenizer})
    root = root.resolve()
    contract, split_frontmatter = load_contract(root)
    destination = index_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_value)
    content_entries: list[tuple[str, str]] = []
    card_count = 0
    posting_count = 0
    try:
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
            for doc_id, path in enumerate(_card_files(root), start=1):
                before = path.stat()
                raw = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError(f"knowledge card changed while indexing: {path}")
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"knowledge card is not UTF-8: {path}") from exc
                metadata, body = split_frontmatter(text)
                if metadata is None:
                    raise ValueError(f"invalid card {path}: missing frontmatter")
                relative = path.relative_to(root)
                path_info = contract.derive_card_path(relative)
                issues = contract.validate_card(relative, metadata, body)
                if issues:
                    detail = "; ".join(
                        f"{item.code}: {item.message}" for item in issues
                    )
                    raise ValueError(f"invalid card {path}: {detail}")

                relative_text = relative.as_posix()
                file_sha256 = hashlib.sha256(raw).hexdigest()
                content_entries.append((relative_text, file_sha256))
                stat = after
                connection.execute(
                    "INSERT INTO manifest VALUES (?, ?, ?, ?)",
                    (relative_text, stat.st_size, stat.st_mtime_ns, file_sha256),
                )
                title = str(metadata.get("title", ""))
                description = str(metadata.get("description", ""))
                tags = " ".join(str(value) for value in metadata.get("tags", []))
                aliases = " ".join(str(value) for value in metadata.get("aliases", []))
                indexed_body = searchable_body(body)
                search_text = "\n".join(
                    (title, description, tags, aliases, indexed_body)
                ).lower()
                fingerprint = card_fingerprint(metadata, body)
                connection.execute(
                    """
                    INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        path_info.doc_id,
                        relative_text,
                        path_info.domain,
                        path_info.route,
                        path_info.card_type,
                        str(metadata.get("type", "")),
                        str(metadata.get("status", "")),
                        title,
                        description,
                        body,
                        search_text,
                        _canonical_json(metadata),
                        len(text.splitlines()),
                        fingerprint,
                    ),
                )
                fields = (
                    token_counts(title, tokenizer=tokenizer),
                    token_counts(description, tokenizer=tokenizer),
                    token_counts(tags, tokenizer=tokenizer),
                    token_counts(aliases, tokenizer=tokenizer),
                    token_counts(indexed_body, tokenizer=tokenizer),
                )
                all_terms = set().union(*(set(field) for field in fields))
                connection.executemany(
                    "INSERT INTO postings VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        (term, doc_id, *(field.get(term, 0) for field in fields))
                        for term in sorted(all_terms)
                    ),
                )
                card_count += 1
                posting_count += len(all_terms)

            metadata = {
                "schema_version": INDEX_SCHEMA,
                "index_version": index_version_for(tokenizer),
                "tokenizer": tokenizer,
                "content_fingerprint": _fingerprint(content_entries),
                "governance_fingerprint": governance_fingerprint(root),
                "card_count": card_count,
                "posting_count": posting_count,
                "repository_state": repository_state(root),
            }
            connection.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                (
                    (key, _canonical_json(value))
                    for key, value in sorted(metadata.items())
                ),
            )
            connection.commit()
        finally:
            connection.close()
        if os.name == "posix":
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if os.name == "posix":
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return metadata
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_meta(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key, value FROM meta")
    }


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _verify_query_tables(
    connection: sqlite3.Connection,
    root: Path,
    current_paths: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """Bind the query-facing SQLite rows to the governed source cards."""
    contract, split_frontmatter = load_contract(root)
    document_count = int(
        connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    )
    if document_count != len(current_paths):
        raise ValueError(
            "knowledge index document count is inconsistent; rebuild the index"
        )

    expected_posting_count = 0
    for expected_doc_id, (relative, card_path) in enumerate(
        current_paths.items(), start=1
    ):
        raw = card_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"knowledge card is not UTF-8: {card_path}") from exc
        card_metadata, body = split_frontmatter(text)
        if card_metadata is None:
            raise ValueError(f"invalid card {card_path}: missing frontmatter")
        path_info = contract.derive_card_path(Path(relative))
        issues = contract.validate_card(Path(relative), card_metadata, body)
        if issues:
            detail = "; ".join(f"{item.code}: {item.message}" for item in issues)
            raise ValueError(f"invalid card {card_path}: {detail}")

        title = str(card_metadata.get("title", ""))
        description = str(card_metadata.get("description", ""))
        tags = " ".join(str(value) for value in card_metadata.get("tags", []))
        aliases = " ".join(str(value) for value in card_metadata.get("aliases", []))
        indexed_body = searchable_body(body)
        search_text = "\n".join(
            (title, description, tags, aliases, indexed_body)
        ).lower()
        expected_document = (
            expected_doc_id,
            path_info.doc_id,
            relative,
            path_info.domain,
            path_info.route,
            path_info.card_type,
            str(card_metadata.get("type", "")),
            str(card_metadata.get("status", "")),
            title,
            description,
            body,
            search_text,
            _canonical_json(card_metadata),
            len(text.splitlines()),
            card_fingerprint(card_metadata, body),
        )
        row = connection.execute(
            """
            SELECT doc_id, concept_id, relative_path, domain, technology, profile,
                   okf_type, status, title, description, body, search_text,
                   metadata_json, card_lines, card_sha256
            FROM documents WHERE relative_path = ?
            """,
            (relative,),
        ).fetchone()
        if row is None or tuple(row) != expected_document:
            raise ValueError(
                f"knowledge index document differs from governed card: {relative}; rebuild the index"
            )

        tokenizer = index_tokenizer(metadata)
        fields = (
            token_counts(title, tokenizer=tokenizer),
            token_counts(description, tokenizer=tokenizer),
            token_counts(tags, tokenizer=tokenizer),
            token_counts(aliases, tokenizer=tokenizer),
            token_counts(indexed_body, tokenizer=tokenizer),
        )
        all_terms = set().union(*(set(field) for field in fields))
        expected_postings = {
            term: tuple(field.get(term, 0) for field in fields) for term in all_terms
        }
        actual_postings = {
            str(posting["term"]): (
                int(posting["title_tf"]),
                int(posting["description_tf"]),
                int(posting["tags_tf"]),
                int(posting["aliases_tf"]),
                int(posting["body_tf"]),
            )
            for posting in connection.execute(
                """
                SELECT term, title_tf, description_tf, tags_tf, aliases_tf, body_tf
                FROM postings WHERE doc_id = ?
                """,
                (expected_doc_id,),
            )
        }
        if actual_postings != expected_postings:
            raise ValueError(
                f"knowledge index postings differ from governed card: {relative}; rebuild the index"
            )
        expected_posting_count += len(expected_postings)

    actual_posting_count = int(
        connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
    )
    if (
        actual_posting_count != expected_posting_count
        or metadata.get("posting_count") != expected_posting_count
    ):
        raise ValueError(
            "knowledge index posting count is inconsistent; rebuild the index"
        )


def _verify_connection(
    root: Path,
    path: Path,
    connection: sqlite3.Connection,
    *,
    full: bool,
) -> dict[str, Any]:
    try:
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise ValueError(f"knowledge index quick_check failed: {check}")
        metadata = _read_meta(connection)
        if metadata.get("schema_version") != INDEX_SCHEMA:
            raise ValueError("knowledge index schema is unsupported; rebuild the index")
        tokenizer = index_tokenizer(metadata)
        if metadata.get("index_version") != index_version_for(tokenizer):
            raise ValueError("knowledge index version is stale; rebuild the index")
        current_governance = governance_fingerprint(root)
        if metadata.get("governance_fingerprint") != current_governance:
            raise ValueError(
                "knowledge governance changed after index build; rebuild the index"
            )
        manifest = {
            row["relative_path"]: row
            for row in connection.execute(
                "SELECT relative_path, size, mtime_ns, file_sha256 FROM manifest"
            )
        }
        current_paths = {
            card.relative_to(root).as_posix(): card for card in _card_files(root)
        }
        if set(manifest) != set(current_paths):
            raise ValueError(
                "knowledge card set changed after index build; rebuild the index"
            )
        content_entries: list[tuple[str, str]] = []
        for relative, card_path in current_paths.items():
            stat = card_path.stat()
            expected = manifest[relative]
            if (
                stat.st_size != expected["size"]
                or stat.st_mtime_ns != expected["mtime_ns"]
            ):
                raise ValueError(
                    f"knowledge card changed after index build: {relative}"
                )
            file_sha256 = expected["file_sha256"]
            if full:
                file_sha256 = sha256_file(card_path)
                if file_sha256 != expected["file_sha256"]:
                    raise ValueError(
                        f"knowledge card content changed after index build: {relative}"
                    )
            content_entries.append((relative, file_sha256))
        if full and metadata.get("content_fingerprint") != _fingerprint(
            content_entries
        ):
            raise ValueError(
                "knowledge content fingerprint is stale; rebuild the index"
            )
        if metadata.get("card_count") != len(manifest):
            raise ValueError(
                "knowledge index card count is inconsistent; rebuild the index"
            )
        if full:
            _verify_query_tables(connection, root, current_paths, metadata)
        return metadata
    except sqlite3.Error as exc:
        raise ValueError(f"invalid knowledge index {path}: {exc}") from exc


def _open_verified_index(
    root: Path,
    *,
    full: bool,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    root = root.resolve()
    path = index_path(root)
    if not path.is_file():
        raise ValueError(
            f"knowledge index is missing: {path}; run knowledge_index.py build"
        )
    try:
        connection = _connect_readonly(path)
    except sqlite3.Error as exc:
        raise ValueError(f"cannot open knowledge index {path}: {exc}") from exc
    try:
        metadata = _verify_connection(root, path, connection, full=full)
    except Exception:
        connection.close()
        raise
    return connection, metadata


def verify_index(root: Path, *, full: bool = False) -> dict[str, Any]:
    connection, metadata = _open_verified_index(root, full=full)
    connection.close()
    return metadata


def open_index(
    root: Path,
    *,
    full: bool = False,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    return _open_verified_index(root, full=full)


def load_cards(connection: sqlite3.Connection) -> list[IndexedCard]:
    result = []
    for row in connection.execute(
        """
        SELECT doc_id, concept_id, relative_path, domain, technology, profile,
               metadata_json, card_lines, card_sha256
        FROM documents ORDER BY concept_id
        """
    ):
        result.append(
            IndexedCard(
                doc_id=row["doc_id"],
                path=row["concept_id"],
                relative_path=row["relative_path"],
                domain=row["domain"],
                technology=row["technology"],
                profile=row["profile"],
                metadata=json.loads(row["metadata_json"]),
                card_lines=row["card_lines"],
                card_sha256=row["card_sha256"],
            )
        )
    return result


def posting_scores(
    connection: sqlite3.Connection,
    terms: list[str],
    tokenizer: str = DEFAULT_TOKENIZER,
) -> dict[int, float]:
    """Score postings. v3 sums field weights; subword applies BM25-style IDF."""
    if not terms:
        return {}
    tokenizer = index_tokenizer({"tokenizer": tokenizer})
    unique = sorted(set(terms))
    placeholders = ",".join("?" for _ in unique)
    if tokenizer != TOKENIZER_SUBWORD:
        rows = connection.execute(
            f"""
            SELECT doc_id,
                   SUM({_POSTING_FIELD_SQL}) AS score
            FROM postings WHERE term IN ({placeholders}) GROUP BY doc_id
            """,
            unique,
        )
        return {int(row["doc_id"]): float(row["score"]) for row in rows}

    total = int(
        connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    )
    rows = connection.execute(
        f"""
        SELECT p.doc_id AS doc_id,
               {_POSTING_FIELD_SQL} AS s,
               d.df AS df
        FROM postings AS p
        JOIN (
            SELECT term, COUNT(DISTINCT doc_id) AS df
            FROM postings WHERE term IN ({placeholders})
            GROUP BY term
        ) AS d ON d.term = p.term
        WHERE p.term IN ({placeholders})
        """,
        (*unique, *unique),
    )
    scores: dict[int, float] = {}
    for row in rows:
        df = max(1, int(row["df"]))
        weight = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
        doc_id = int(row["doc_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + weight * float(row["s"])
    return scores


def literal_scores(connection: sqlite3.Connection, phrase: str) -> dict[int, float]:
    """Rank literal hits by the field that contains the complete phrase."""
    if not phrase:
        return {}
    normalized = phrase.lower()
    rows = connection.execute(
        """
        SELECT doc_id,
               CASE
                   WHEN lower(title) = ? THEN 12.0
                   WHEN instr(lower(title), ?) > 0 THEN 8.0
                   WHEN instr(lower(description), ?) > 0 THEN 4.0
                   WHEN instr(lower(body), ?) > 0 THEN 1.0
                   ELSE 0.5
               END AS score
        FROM documents
        WHERE instr(search_text, ?) > 0
        """,
        (normalized, normalized, normalized, normalized, normalized),
    )
    return {int(row["doc_id"]): float(row["score"]) for row in rows}


def phrase_matches(connection: sqlite3.Connection, phrase: str) -> set[int]:
    if not phrase:
        return set()
    return {
        int(row["doc_id"])
        for row in connection.execute(
            "SELECT doc_id FROM documents WHERE instr(search_text, ?) > 0",
            (phrase.lower(),),
        )
    }
