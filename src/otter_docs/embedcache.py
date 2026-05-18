"""Embedding cache — content-addressed reuse of the three per-symbol vectors.

Sibling of `describe.DescriptionCache`. Where the description cache makes
the *LLM* step incremental, this makes the *embedder* step incremental —
the part that actually dominates a re-run's wall time (every nightly
enrich was re-embedding every symbol in the repo, not just changed ones).

Key = SHA-1 over the exact three strings handed to the embedder
(description / code / docstring, post-truncation), plus the embedder's
model id and dim. Hashing the literal embedder inputs makes a hit mean
precisely "we have already embedded these three texts with this model"
— no reasoning about determinism of truncation or the describe step is
needed; it is self-evidently correct. `embed_model` + `dim` are in the
key because swapping the embedder must invalidate: stale vectors at the
wrong dim would corrupt the backend's vec0 tables (the same failure the
backend's dim guard already rejects).

Vectors are stored as JSON text, not struct-packed float32: a cache hit
must return the embedder's exact values. Float32 truncation is the
backend's job at vec0 write time, not the cache's — keeping full
precision here also keeps re-run vectors bit-identical for tests.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "CachedVectors",
    "EmbeddingCache",
    "InMemoryEmbeddingCache",
    "SqliteEmbeddingCache",
    "embed_model_id",
]


@dataclass(frozen=True)
class CachedVectors:
    """The three vectors for one symbol. `docstring_vec` is None when the
    symbol has no docstring (mirrors the record field's nullability)."""

    description_vec: list[float]
    code_vec: list[float]
    docstring_vec: list[float] | None


def embed_model_id(embedder: object) -> str:
    """A stable identifier for an embedder's model.

    OpenAI-compat / Ollama clients carry `.model`; test fakes don't.
    Fall back to the class name so distinct fakes still key distinctly.
    """
    return str(getattr(embedder, "model", None) or type(embedder).__name__)


class EmbeddingCache(Protocol):
    """Pluggable storage for (content_hash, embed_model, dim) → vectors.

    Looked up on the hot path of every enriched symbol; implementations
    should be fast. The default lives in the same SQLite file as the
    graph (see `SqliteEmbeddingCache`)."""

    def get(
        self, content_hash: str, *, embed_model: str, dim: int
    ) -> CachedVectors | None: ...

    def put(
        self,
        content_hash: str,
        *,
        embed_model: str,
        dim: int,
        kind: str,
        guid: str,
        vectors: CachedVectors,
    ) -> None: ...


class SqliteEmbeddingCache:
    """SQLite-backed cache, in the same graph.db as `code_descriptions`.

    kind/guid are stored as inspection metadata only; they do not enter
    the lookup key, so a renamed/moved symbol with an unchanged body
    still reuses its vectors (same reasoning as the description cache).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS code_embeddings (
                content_hash    TEXT NOT NULL,
                embed_model     TEXT NOT NULL,
                dim             INTEGER NOT NULL,
                kind            TEXT NOT NULL,
                guid            TEXT NOT NULL,
                description_vec TEXT NOT NULL,
                code_vec        TEXT NOT NULL,
                docstring_vec   TEXT,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (content_hash, embed_model, dim)
            )
        """)
        self._conn.commit()

    def get(
        self, content_hash: str, *, embed_model: str, dim: int
    ) -> CachedVectors | None:
        row = self._conn.execute(
            "SELECT description_vec, code_vec, docstring_vec "
            "FROM code_embeddings "
            "WHERE content_hash = ? AND embed_model = ? AND dim = ?",
            (content_hash, embed_model, dim),
        ).fetchone()
        if row is None:
            return None
        return CachedVectors(
            description_vec=json.loads(row[0]),
            code_vec=json.loads(row[1]),
            docstring_vec=json.loads(row[2]) if row[2] is not None else None,
        )

    def put(
        self,
        content_hash: str,
        *,
        embed_model: str,
        dim: int,
        kind: str,
        guid: str,
        vectors: CachedVectors,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO code_embeddings "
                "(content_hash, embed_model, dim, kind, guid, "
                " description_vec, code_vec, docstring_vec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    content_hash,
                    embed_model,
                    dim,
                    kind,
                    guid,
                    json.dumps(vectors.description_vec),
                    json.dumps(vectors.code_vec),
                    None
                    if vectors.docstring_vec is None
                    else json.dumps(vectors.docstring_vec),
                ),
            )


class InMemoryEmbeddingCache:
    """In-memory cache for tests / one-shot runs."""

    def __init__(self) -> None:
        self._d: dict[tuple[str, str, int], CachedVectors] = {}

    def get(
        self, content_hash: str, *, embed_model: str, dim: int
    ) -> CachedVectors | None:
        return self._d.get((content_hash, embed_model, dim))

    def put(
        self,
        content_hash: str,
        *,
        embed_model: str,
        dim: int,
        kind: str,
        guid: str,
        vectors: CachedVectors,
    ) -> None:
        self._d[(content_hash, embed_model, dim)] = vectors
