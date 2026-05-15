"""Describer — LLM-generated structured prose for symbols.

A description is a short text blob written from a function/class/module
body. It's the input to the **description vector** (one of the three
vectors per symbol) and stands on its own for human readers too — when
the doc renderer wants "what does this do in one sentence", this is
where it asks.

Caching is keyed by SHA-1 over the source bytes (matches git's blob
hash but doesn't require git). Re-describing the same content always
hits the cache; the model is only re-invoked when the source actually
changes.

The describer never writes to the graph itself — it returns
`Description` records and lets the caller decide what to do with them
(usually: feed text → embedder → backend.add_function(..., description_vec=...)).
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

DEFAULT_PROMPT_TEMPLATE = """\
Given this {kind}, produce a structured description.

Source:
```{language}
{source}
```

Format your response exactly like this:

Purpose: [one sentence on WHAT this does, ignoring HOW]
Inputs: [semantic meaning of each parameter, not types]
Outputs: [semantic meaning of return, not types]
Side effects: [I/O, mutations, network calls, db writes — "none" if pure]
Category: [one or more of: io.read, io.write, compute.hash, \
compute.transform, compute.aggregate, network.http, network.email, \
database.query, database.mutation, validation, serialization, \
deserialization, control_flow, ui, configuration]
"""


@dataclass(frozen=True)
class Description:
    """A described symbol — the text we'd embed as `description_vec`.

    `content_hash` is the SHA-1 of the source bytes that produced this
    description. It's the cache key and also a way to detect drift if
    a downstream consumer wants to check whether a stored description
    is still current.
    """

    kind: str  # "function" | "class" | "module"
    guid: str
    content_hash: str
    text: str


def content_hash(source: bytes) -> str:
    """SHA-1 over source bytes — same algorithm git uses for blob ids.

    Not collision-resistant in the cryptographic sense, but the
    threat model here is "did this function change", not adversarial
    input. SHA-1 keeps cache keys short and aligns with the way
    every other tool in this ecosystem identifies content.
    """
    return hashlib.sha1(source).hexdigest()


class DescriptionCache(Protocol):
    """Pluggable storage for (content_hash, kind, guid) → description text.

    Cache lookups happen on the hot path of every describer call;
    implementations should be fast. The default lives in the same
    SQLite file as the graph (see `SqliteDescriptionCache`).
    """

    def get(self, content_hash: str) -> str | None: ...

    def put(self, content_hash: str, *, kind: str, guid: str, text: str) -> None: ...


class SqliteDescriptionCache:
    """SQLite-backed cache keyed solely by content_hash.

    Storing only the content_hash means a function that gets renamed
    or moved to a different file still reuses the cached description —
    because the description is about what the code *does*, not where
    it lives. kind/guid are kept as metadata for inspection but don't
    enter the lookup key.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS code_descriptions (
                content_hash TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,
                guid         TEXT NOT NULL,
                text         TEXT NOT NULL,
                created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def get(self, content_hash: str) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM code_descriptions WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        return row[0] if row else None

    def put(self, content_hash: str, *, kind: str, guid: str, text: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO code_descriptions "
                "(content_hash, kind, guid, text) VALUES (?, ?, ?, ?)",
                (content_hash, kind, guid, text),
            )


class _DictCache:
    """In-memory cache for tests."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, content_hash: str) -> str | None:
        return self._d.get(content_hash)

    def put(self, content_hash: str, *, kind: str, guid: str, text: str) -> None:
        self._d[content_hash] = text


class Describer:
    """Describes symbols by source bytes; caches by content hash.

    Parameters
    ----------
    llm :
        An LLMClient (real or fake). Called only on cache miss.
    cache :
        A DescriptionCache. Use `SqliteDescriptionCache` for persistence
        or `_DictCache` for tests / one-shot runs.
    prompt_template :
        Override the default if you want a different schema. Must
        accept `{kind}`, `{language}`, and `{source}`.
    llm_options :
        Forwarded to LLMClient.complete each call. Default is
        temperature 0 + a tight num_predict to keep descriptions short.
    """

    def __init__(
        self,
        llm,
        cache: DescriptionCache | None = None,
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        llm_options: dict[str, object] | None = None,
    ) -> None:
        self.llm = llm
        self.cache: DescriptionCache = cache or _DictCache()
        self.prompt_template = prompt_template
        self.llm_options: dict[str, object] = (
            llm_options if llm_options is not None
            else {"temperature": 0.0, "num_predict": 256}
        )

    def describe(
        self,
        *,
        kind: str,
        guid: str,
        language: str,
        source: bytes,
    ) -> Description:
        h = content_hash(source)
        cached = self.cache.get(h)
        if cached is not None:
            return Description(kind=kind, guid=guid, content_hash=h, text=cached)

        prompt = self.prompt_template.format(
            kind=kind,
            language=language,
            source=source.decode("utf-8", errors="replace"),
        )
        text = self.llm.complete(prompt, **self.llm_options)
        self.cache.put(h, kind=kind, guid=guid, text=text)
        return Description(kind=kind, guid=guid, content_hash=h, text=text)

    def describe_many(
        self,
        items: Iterable[tuple[str, str, str, bytes]],
    ) -> list[Description]:
        """Convenience: items is an iterable of (kind, guid, language, source).

        Just a `describe` loop — kept around so test setup can stay terse.
        """
        return [
            self.describe(kind=k, guid=g, language=lang, source=src)
            for k, g, lang, src in items
        ]
