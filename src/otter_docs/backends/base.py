"""GraphBackend protocol.

Every backend implements this surface. Detectors and renderers operate
against the protocol, not against any concrete backend, so swapping
backends (SQLite → Neo4j → DuckDB) is a config change, not a code change.

The protocol is deliberately small. CRUD on the three node types, edges,
similarity search, and a raw query escape hatch. Anything richer (path
finding, transitive closure, multi-hop traversal) is composed on top by
the `graph` module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol, runtime_checkable

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    ModuleRecord,
    SimilarityHit,
    VectorKind,
)


@runtime_checkable
class GraphBackend(Protocol):
    """Persistent store for the codebase model.

    All write methods are idempotent — calling `add_module` twice with the
    same `(repo, path)` updates the existing record rather than erroring.
    All vectors are optional; backends without vector support raise
    `NotImplementedError` on `find_similar` and store `None` for the vector
    columns.
    """

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the backend connection and apply migrations if needed."""

    def close(self) -> None:
        """Close the backend connection."""

    # ── writes (idempotent upserts) ─────────────────────────────────

    def add_module(self, module: ModuleRecord) -> None: ...

    def add_function(self, function: FunctionRecord) -> None: ...

    def add_class(self, cls: ClassRecord) -> None: ...

    def add_edge(self, edge: Edge) -> None: ...

    # ── reads ────────────────────────────────────────────────────────

    def get_module(self, repo: str, path: str) -> ModuleRecord | None: ...

    def get_function(self, repo: str, guid: str) -> FunctionRecord | None: ...

    def get_class(self, repo: str, guid: str) -> ClassRecord | None: ...

    def list_modules(self, repo: str | None = None) -> Iterator[ModuleRecord]: ...

    def list_functions(self, repo: str | None = None) -> Iterator[FunctionRecord]: ...

    def list_classes(self, repo: str | None = None) -> Iterator[ClassRecord]: ...

    # ── graph traversal ──────────────────────────────────────────────

    def callers_of(self, repo: str, target_guid: str) -> list[str]:
        """GUIDs of functions/classes that CALL the target."""

    def edges_from(self, repo: str, src_id: str, kind: str | None = None) -> list[Edge]: ...

    def edges_to(self, repo: str, dst_id: str, kind: str | None = None) -> list[Edge]: ...

    # ── vector similarity ───────────────────────────────────────────

    def find_similar(
        self,
        repo: str,
        vector: list[float],
        *,
        vector_kind: VectorKind,
        node_kind: str = "function",  # "module" | "function" | "class"
        k: int = 20,
        min_similarity: float = 0.0,
    ) -> list[SimilarityHit]:
        """Nearest neighbors by cosine similarity over the chosen vector slot.

        Backends without vector support raise NotImplementedError.
        """

    # ── lifecycle helpers ────────────────────────────────────────────

    def reset(self, *, repo: str | None = None) -> None:
        """Drop all Code* nodes and edges, optionally scoped to one repo.

        Used by `index --reset` for a clean rebuild after schema changes.
        Without `repo`, wipes everything.
        """

    def query(self, raw: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        """Backend-specific raw query escape hatch.

        For SQLite: SQL. For Neo4j: Cypher. Use sparingly; prefer the
        typed methods above. Returned rows are lists of dicts keyed by
        column name.
        """
