"""Redundancy-verdict cache — content-addressed reuse of LLM judgements.

Sibling of `embedcache.SqliteEmbeddingCache` and
`describe.SqliteDescriptionCache`. Where those make the embed and
describe steps incremental, this one makes the
`Repo.confirm_redundancy` LLM step incremental — without it, the
publisher would burn a fresh ~5s LLM call per pair on every nightly
run.

Key = SHA-1 over (a_source + "\\n---\\n" + b_source), with both source
slices ordered canonically (sorted by guid) before concatenation so
the pair (A, B) and (B, A) collide. Hashing the literal model inputs
makes a hit mean precisely "we have already judged this same pair of
bytes with this model" — no reasoning about which function got picked
as canonical or whether the description changed is needed. `llm_model`
is in the key because swapping the judge invalidates: a verdict from
a small model isn't a verdict from a big one.

Verdicts are stored as denormalized columns (is_duplicate, confidence,
kind, reason) rather than JSON — they're small, fixed-shape, and
queryable for offline analysis without parsing.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Protocol

from otter_docs.llm_direct import RedundancyVerdict

__all__ = [
    "InMemoryRedundancyCache",
    "RedundancyCache",
    "SqliteRedundancyCache",
    "llm_model_id",
    "verdict_content_hash",
]


def llm_model_id(llm: object) -> str:
    """Stable id for the judge's model — same trick `embed_model_id` uses."""
    return str(getattr(llm, "model", None) or type(llm).__name__)


def verdict_content_hash(a_source: str, b_source: str) -> str:
    """SHA-1 over the canonically-ordered concatenation of both bodies.

    Ordering by string compare (not by guid — guids change when the
    surrounding file does) keeps the pair (A, B) ≡ (B, A) so we don't
    double-cache when the detector emits them in the other order.
    """
    first, second = sorted([a_source, b_source])
    h = hashlib.sha1()
    h.update(first.encode("utf-8", errors="replace"))
    h.update(b"\n---\n")
    h.update(second.encode("utf-8", errors="replace"))
    return h.hexdigest()


class RedundancyCache(Protocol):
    """Pluggable storage for (content_hash, llm_model) → RedundancyVerdict."""

    def get(
        self, content_hash: str, *, llm_model: str
    ) -> RedundancyVerdict | None: ...

    def get_any(self, content_hash: str) -> RedundancyVerdict | None:
        """Model-agnostic read: the most recent verdict for this content
        hash regardless of which LLM produced it. For consumers with no
        LLM handle (e.g. the renderer) that want to reflect confirmed
        verdicts. Returns None if nothing is cached for the hash."""
        ...

    def put(
        self,
        content_hash: str,
        *,
        llm_model: str,
        verdict: RedundancyVerdict,
    ) -> None: ...


class SqliteRedundancyCache:
    """SQLite-backed cache, in the same graph.db as `code_descriptions`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS redundancy_verdicts (
                content_hash    TEXT NOT NULL,
                llm_model       TEXT NOT NULL,
                is_duplicate    INTEGER NOT NULL,
                confidence      REAL NOT NULL,
                kind            TEXT NOT NULL,
                reason          TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (content_hash, llm_model)
            )
        """)
        self._conn.commit()

    def get(
        self, content_hash: str, *, llm_model: str
    ) -> RedundancyVerdict | None:
        row = self._conn.execute(
            "SELECT is_duplicate, confidence, kind, reason "
            "FROM redundancy_verdicts "
            "WHERE content_hash = ? AND llm_model = ?",
            (content_hash, llm_model),
        ).fetchone()
        if row is None:
            return None
        return RedundancyVerdict(
            is_duplicate=bool(row[0]),
            confidence=float(row[1]),
            kind=row[2],
            reason=row[3],
        )

    def get_any(self, content_hash: str) -> RedundancyVerdict | None:
        row = self._conn.execute(
            "SELECT is_duplicate, confidence, kind, reason "
            "FROM redundancy_verdicts WHERE content_hash = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return RedundancyVerdict(
            is_duplicate=bool(row[0]),
            confidence=float(row[1]),
            kind=row[2],
            reason=row[3],
        )

    def put(
        self,
        content_hash: str,
        *,
        llm_model: str,
        verdict: RedundancyVerdict,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO redundancy_verdicts "
                "(content_hash, llm_model, is_duplicate, confidence, kind, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    content_hash,
                    llm_model,
                    1 if verdict.is_duplicate else 0,
                    verdict.confidence,
                    verdict.kind,
                    verdict.reason,
                ),
            )


class InMemoryRedundancyCache:
    """In-memory cache for tests / one-shot runs."""

    def __init__(self) -> None:
        self._d: dict[tuple[str, str], RedundancyVerdict] = {}

    def get(
        self, content_hash: str, *, llm_model: str
    ) -> RedundancyVerdict | None:
        return self._d.get((content_hash, llm_model))

    def get_any(self, content_hash: str) -> RedundancyVerdict | None:
        # Most-recent insertion wins (dict preserves insertion order).
        for (ch, _model), verdict in reversed(self._d.items()):
            if ch == content_hash:
                return verdict
        return None

    def put(
        self,
        content_hash: str,
        *,
        llm_model: str,
        verdict: RedundancyVerdict,
    ) -> None:
        self._d[(content_hash, llm_model)] = verdict
