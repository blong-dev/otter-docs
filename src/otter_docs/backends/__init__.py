"""Graph backends for otter-docs.

A backend implements `GraphBackend` (see `.base`) and stores the codebase
model — modules, functions, classes, edges, and per-symbol vectors.

The default is `SqliteBackend` (embedded, sqlite-vec for HNSW vector
indexes, zero install friction). Neo4j and DuckDB adapters land in
follow-on phases.
"""

from otter_docs.backends.base import GraphBackend
from otter_docs.backends.sqlite import SqliteBackend


def __getattr__(name: str):
    # Lazy-import Neo4jBackend so the `neo4j` driver only loads when
    # actually requested. Keeps the default `pip install otter-docs`
    # footprint small and avoids ImportError at import time when the
    # optional [neo4j] extra isn't installed.
    if name == "Neo4jBackend":
        from otter_docs.backends.neo4j import Neo4jBackend
        return Neo4jBackend
    raise AttributeError(f"module 'otter_docs.backends' has no attribute {name!r}")


__all__ = ["GraphBackend", "Neo4jBackend", "SqliteBackend"]
