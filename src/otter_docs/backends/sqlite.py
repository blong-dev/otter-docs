"""SQLite + sqlite-vec implementation of GraphBackend.

This is the default backend. Zero install friction (sqlite ships with
Python), embeddable, fast enough for the codebase sizes we care about
in v0.1. Uses sqlite-vec for HNSW vector indexes over the three vector
slots per symbol (description / code / docstring).

Schema overview:

    code_modules    one row per source file
    code_functions  one row per def/method
    code_classes    one row per class
    code_edges      one row per (kind, src, dst) typed edge
    code_meta       single-row table for schema version + vector dim

    vec_modules_description   vec0 virtual table mirroring code_modules.rowid
    vec_modules_code          ...
    vec_modules_docstring     ...
    vec_functions_description, vec_functions_code, vec_functions_docstring
    vec_classes_description,   vec_classes_code,   vec_classes_docstring

The vec tables share rowid with their parent so a vector hit can be joined
back to the full record cheaply.

Vector dimension is fixed at backend creation and stored in code_meta. A
later connection with a mismatched dim raises rather than silently
corrupting indexes.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlite_vec

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language,
    ModuleRecord,
    SimilarityHit,
    VectorKind,
)

SCHEMA_VERSION = 1
DEFAULT_VECTOR_DIM = 768  # matches nomic-embed-text, our local-default embedder

# Singular → plural mapping. English plurals are irregular enough that
# `f"{kind}s"` breaks on "class" → "classs"; the parent tables and vec
# tables both use the proper plural.
_NODE_KINDS: dict[str, str] = {
    "module": "modules",
    "function": "functions",
    "class": "classes",
}
_VEC_KINDS = ("description", "code", "docstring")


def _serialize_vec(values: list[float]) -> bytes:
    """Pack a Python list[float] into the byte format sqlite-vec expects."""
    return struct.pack(f"{len(values)}f", *values)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class SqliteBackend:
    """GraphBackend implementation backed by SQLite + sqlite-vec."""

    def __init__(self, path: str | Path = ":memory:", *, vector_dim: int = DEFAULT_VECTOR_DIM):
        self.path = str(path)
        self.vector_dim = vector_dim
        self._conn: sqlite3.Connection | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._conn is not None:
            return
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        self._conn = conn
        self._migrate()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SqliteBackend:
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _migrate(self) -> None:
        assert self._conn is not None
        c = self._conn

        # Bootstrap meta table; check dim compatibility on reconnect.
        c.executescript("""
            CREATE TABLE IF NOT EXISTS code_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        existing_dim_row = c.execute(
            "SELECT value FROM code_meta WHERE key = 'vector_dim'"
        ).fetchone()
        if existing_dim_row is None:
            c.execute(
                "INSERT INTO code_meta (key, value) VALUES (?, ?)",
                ("vector_dim", str(self.vector_dim)),
            )
            c.execute(
                "INSERT OR REPLACE INTO code_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
        else:
            stored = int(existing_dim_row[0])
            if stored != self.vector_dim:
                raise ValueError(
                    f"vector_dim mismatch: backend constructed with {self.vector_dim}, "
                    f"existing db has {stored}. Use --reset to rebuild."
                )

        # Primary tables.
        c.executescript("""
            CREATE TABLE IF NOT EXISTS code_modules (
                rowid INTEGER PRIMARY KEY,
                repo TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                docstring TEXT NOT NULL DEFAULT '',
                imports TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT,
                UNIQUE(repo, path)
            );
            CREATE INDEX IF NOT EXISTS idx_modules_repo ON code_modules(repo);

            CREATE TABLE IF NOT EXISTS code_functions (
                rowid INTEGER PRIMARY KEY,
                repo TEXT NOT NULL,
                guid TEXT NOT NULL,
                name TEXT NOT NULL,
                module_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                docstring TEXT NOT NULL DEFAULT '',
                args TEXT NOT NULL DEFAULT '[]',
                returns TEXT NOT NULL DEFAULT '',
                is_async INTEGER NOT NULL DEFAULT 0,
                tags TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT,
                UNIQUE(repo, guid)
            );
            CREATE INDEX IF NOT EXISTS idx_functions_repo ON code_functions(repo);
            CREATE INDEX IF NOT EXISTS idx_functions_module
                ON code_functions(repo, module_path);

            CREATE TABLE IF NOT EXISTS code_classes (
                rowid INTEGER PRIMARY KEY,
                repo TEXT NOT NULL,
                guid TEXT NOT NULL,
                name TEXT NOT NULL,
                module_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                docstring TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT,
                UNIQUE(repo, guid)
            );
            CREATE INDEX IF NOT EXISTS idx_classes_repo ON code_classes(repo);
            CREATE INDEX IF NOT EXISTS idx_classes_module
                ON code_classes(repo, module_path);

            CREATE TABLE IF NOT EXISTS code_edges (
                repo TEXT NOT NULL,
                kind TEXT NOT NULL,
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                updated_at TEXT,
                PRIMARY KEY (repo, kind, src_id, dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON code_edges(repo, src_id, kind);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON code_edges(repo, dst_id, kind);
        """)

        # Vector tables (vec0 virtual tables). One per (node_kind, vec_kind).
        for plural in _NODE_KINDS.values():
            for vec_kind in _VEC_KINDS:
                table = f"vec_{plural}_{vec_kind}"
                c.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
                    f"embedding float[{self.vector_dim}]"
                    f")"
                )

        c.commit()

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Backend not connected — call .connect() or use 'with'")
        return self._conn

    def _upsert_vector(self, table: str, rowid: int, vec: list[float] | None) -> None:
        """Insert or replace a vector for the given rowid in the named vec table.

        If vec is None, deletes any existing entry. We always delete first
        because vec0 doesn't have an UPDATE primitive — replace via
        delete+insert.
        """
        self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
        if vec is None:
            return
        if len(vec) != self.vector_dim:
            raise ValueError(
                f"Vector dim mismatch: got {len(vec)}, expected {self.vector_dim}"
            )
        self.conn.execute(
            f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_vec(vec)),
        )

    # ── writes ────────────────────────────────────────────────────────

    def add_module(self, module: ModuleRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO code_modules (repo, path, language, docstring,
                                          imports, tags, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo, path) DO UPDATE SET
                    language=excluded.language,
                    docstring=excluded.docstring,
                    imports=excluded.imports,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (
                    module.repo, module.path, module.language.value,
                    module.docstring,
                    json.dumps(module.imports), json.dumps(module.tags),
                    _iso(module.updated_at),
                ),
            )
            rowid = self.conn.execute(
                "SELECT rowid FROM code_modules WHERE repo = ? AND path = ?",
                (module.repo, module.path),
            ).fetchone()[0]
            self._upsert_vector("vec_modules_description", rowid, module.description_vec)
            self._upsert_vector("vec_modules_code", rowid, module.code_vec)
            self._upsert_vector("vec_modules_docstring", rowid, module.docstring_vec)

    def add_function(self, function: FunctionRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO code_functions (repo, guid, name, module_path,
                                            line, end_line, docstring,
                                            args, returns, is_async, tags,
                                            updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo, guid) DO UPDATE SET
                    name=excluded.name,
                    module_path=excluded.module_path,
                    line=excluded.line,
                    end_line=excluded.end_line,
                    docstring=excluded.docstring,
                    args=excluded.args,
                    returns=excluded.returns,
                    is_async=excluded.is_async,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (
                    function.repo, function.guid, function.name,
                    function.module_path, function.line, function.end_line,
                    function.docstring,
                    json.dumps(function.args), function.returns,
                    1 if function.is_async else 0,
                    json.dumps(function.tags), _iso(function.updated_at),
                ),
            )
            rowid = self.conn.execute(
                "SELECT rowid FROM code_functions WHERE repo = ? AND guid = ?",
                (function.repo, function.guid),
            ).fetchone()[0]
            self._upsert_vector("vec_functions_description", rowid, function.description_vec)
            self._upsert_vector("vec_functions_code", rowid, function.code_vec)
            self._upsert_vector("vec_functions_docstring", rowid, function.docstring_vec)

    def add_class(self, cls: ClassRecord) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO code_classes (repo, guid, name, module_path,
                                          line, end_line, docstring, tags,
                                          updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo, guid) DO UPDATE SET
                    name=excluded.name,
                    module_path=excluded.module_path,
                    line=excluded.line,
                    end_line=excluded.end_line,
                    docstring=excluded.docstring,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (
                    cls.repo, cls.guid, cls.name, cls.module_path,
                    cls.line, cls.end_line, cls.docstring,
                    json.dumps(cls.tags), _iso(cls.updated_at),
                ),
            )
            rowid = self.conn.execute(
                "SELECT rowid FROM code_classes WHERE repo = ? AND guid = ?",
                (cls.repo, cls.guid),
            ).fetchone()[0]
            self._upsert_vector("vec_classes_description", rowid, cls.description_vec)
            self._upsert_vector("vec_classes_code", rowid, cls.code_vec)
            self._upsert_vector("vec_classes_docstring", rowid, cls.docstring_vec)

    def add_edge(self, edge: Edge) -> None:
        # Edges don't carry repo on the model (the model defines a typed
        # relationship between two ids); we infer repo from the src node's
        # presence. For simplicity v0.1 requires the caller to pass a
        # repo-scoped edge implicitly via separate sources. Edges are
        # repo-scoped in storage; pull the repo from add_edge's call site
        # by widening Edge in a later phase. For now, callers using add_edge
        # must work through `_add_edge_with_repo`.
        raise NotImplementedError(
            "Edge needs a repo. Use _add_edge_with_repo(edge, repo=...) for v0.1; "
            "Edge model will gain a `repo` field in a later phase."
        )

    def _add_edge_with_repo(self, edge: Edge, repo: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO code_edges (repo, kind, src_id, dst_id, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repo, kind, src_id, dst_id) DO UPDATE SET
                updated_at = excluded.updated_at
            """,
            (repo, edge.kind, edge.src_id, edge.dst_id, now),
        )
        self.conn.commit()

    # ── reads ────────────────────────────────────────────────────────

    def get_module(self, repo: str, path: str) -> ModuleRecord | None:
        row = self.conn.execute(
            "SELECT * FROM code_modules WHERE repo = ? AND path = ?",
            (repo, path),
        ).fetchone()
        return self._row_to_module(row) if row else None

    def get_function(self, repo: str, guid: str) -> FunctionRecord | None:
        row = self.conn.execute(
            "SELECT * FROM code_functions WHERE repo = ? AND guid = ?",
            (repo, guid),
        ).fetchone()
        return self._row_to_function(row) if row else None

    def get_class(self, repo: str, guid: str) -> ClassRecord | None:
        row = self.conn.execute(
            "SELECT * FROM code_classes WHERE repo = ? AND guid = ?",
            (repo, guid),
        ).fetchone()
        return self._row_to_class(row) if row else None

    def list_modules(self, repo: str | None = None) -> Iterator[ModuleRecord]:
        if repo is None:
            cursor = self.conn.execute("SELECT * FROM code_modules ORDER BY repo, path")
        else:
            cursor = self.conn.execute(
                "SELECT * FROM code_modules WHERE repo = ? ORDER BY path", (repo,)
            )
        for row in cursor:
            yield self._row_to_module(row)

    def list_functions(self, repo: str | None = None) -> Iterator[FunctionRecord]:
        if repo is None:
            cursor = self.conn.execute(
                "SELECT * FROM code_functions ORDER BY repo, module_path, line"
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM code_functions WHERE repo = ? ORDER BY module_path, line",
                (repo,),
            )
        for row in cursor:
            yield self._row_to_function(row)

    def list_classes(self, repo: str | None = None) -> Iterator[ClassRecord]:
        if repo is None:
            cursor = self.conn.execute(
                "SELECT * FROM code_classes ORDER BY repo, module_path, line"
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM code_classes WHERE repo = ? ORDER BY module_path, line",
                (repo,),
            )
        for row in cursor:
            yield self._row_to_class(row)

    # ── graph traversal ──────────────────────────────────────────────

    def callers_of(self, repo: str, target_guid: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT src_id FROM code_edges WHERE repo = ? AND dst_id = ? AND kind = 'CALLS'",
            (repo, target_guid),
        ).fetchall()
        return [r[0] for r in rows]

    def edges_from(self, repo: str, src_id: str, kind: str | None = None) -> list[Edge]:
        if kind is None:
            rows = self.conn.execute(
                "SELECT kind, src_id, dst_id FROM code_edges WHERE repo = ? AND src_id = ?",
                (repo, src_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT kind, src_id, dst_id FROM code_edges "
                "WHERE repo = ? AND src_id = ? AND kind = ?",
                (repo, src_id, kind),
            ).fetchall()
        return [Edge(kind=r[0], src_id=r[1], dst_id=r[2]) for r in rows]

    def edges_to(self, repo: str, dst_id: str, kind: str | None = None) -> list[Edge]:
        if kind is None:
            rows = self.conn.execute(
                "SELECT kind, src_id, dst_id FROM code_edges WHERE repo = ? AND dst_id = ?",
                (repo, dst_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT kind, src_id, dst_id FROM code_edges "
                "WHERE repo = ? AND dst_id = ? AND kind = ?",
                (repo, dst_id, kind),
            ).fetchall()
        return [Edge(kind=r[0], src_id=r[1], dst_id=r[2]) for r in rows]

    # ── vector similarity ───────────────────────────────────────────

    def find_similar(
        self,
        repo: str,
        vector: list[float],
        *,
        vector_kind: VectorKind,
        node_kind: str = "function",
        k: int = 20,
        min_similarity: float = 0.0,
    ) -> list[SimilarityHit]:
        if node_kind not in _NODE_KINDS:
            raise ValueError(
                f"node_kind must be one of {list(_NODE_KINDS)}, got {node_kind!r}"
            )
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Query vector dim {len(vector)} != backend dim {self.vector_dim}"
            )
        plural = _NODE_KINDS[node_kind]
        vec_table = f"vec_{plural}_{vector_kind.value}"
        parent_table = f"code_{plural}"
        id_col = "path" if node_kind == "module" else "guid"

        # KNN over the vec table, then join to the parent for repo filtering
        # and id retrieval. sqlite-vec MATCH returns L2 distance; we convert
        # to cosine similarity assuming unit vectors (see embedding pipeline).
        rows = self.conn.execute(
            f"""
            SELECT p.{id_col} AS node_id, v.distance AS distance
            FROM {vec_table} v
            JOIN {parent_table} p ON p.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
              AND p.repo = ?
            ORDER BY v.distance
            """,
            (_serialize_vec(vector), k, repo),
        ).fetchall()

        hits: list[SimilarityHit] = []
        for row in rows:
            # On unit-length vectors: L2² = 2 - 2·cos, so cos = 1 - L2²/2.
            # Normalize to [0, 1] (matching Neo4j's vector index convention):
            #   normalized = (cos + 1) / 2 = 1 - L2²/4
            # Identical vectors → 1.0; orthogonal → 0.5; opposite → 0.0.
            # Callers expecting unit-length vectors get consistent scores
            # across SqliteBackend and Neo4jBackend.
            d = row["distance"]
            similarity = 1.0 - (d * d) / 4.0
            if similarity < min_similarity:
                continue
            hits.append(
                SimilarityHit(node_kind=node_kind, node_id=row["node_id"], similarity=similarity)
            )
        return hits

    # ── lifecycle helpers ────────────────────────────────────────────

    def reset(self, *, repo: str | None = None) -> None:
        with self.conn:
            if repo is None:
                # Wipe everything across all repos.
                for t in ("code_modules", "code_functions", "code_classes", "code_edges"):
                    self.conn.execute(f"DELETE FROM {t}")
                for plural in _NODE_KINDS.values():
                    for vec_kind in _VEC_KINDS:
                        self.conn.execute(f"DELETE FROM vec_{plural}_{vec_kind}")
            else:
                # Scoped to one repo. Vec tables have no repo column; we
                # cascade via rowid lookup against the parent tables.
                for plural in _NODE_KINDS.values():
                    parent = f"code_{plural}"
                    rowids = [
                        r[0]
                        for r in self.conn.execute(
                            f"SELECT rowid FROM {parent} WHERE repo = ?", (repo,)
                        )
                    ]
                    if rowids:
                        placeholders = ",".join("?" * len(rowids))
                        for vec_kind in _VEC_KINDS:
                            self.conn.execute(
                                f"DELETE FROM vec_{plural}_{vec_kind} "
                                f"WHERE rowid IN ({placeholders})",
                                rowids,
                            )
                    self.conn.execute(f"DELETE FROM {parent} WHERE repo = ?", (repo,))
                self.conn.execute("DELETE FROM code_edges WHERE repo = ?", (repo,))

    def query(self, raw: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        cursor = self.conn.execute(raw, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    # ── row → model converters ───────────────────────────────────────

    def _vectors_for(self, table_prefix: str, rowid: int) -> dict[str, list[float] | None]:
        out: dict[str, list[float] | None] = {}
        for vec_kind in _VEC_KINDS:
            row = self.conn.execute(
                f"SELECT embedding FROM vec_{table_prefix}_{vec_kind} WHERE rowid = ?",
                (rowid,),
            ).fetchone()
            if row is None or row[0] is None:
                out[vec_kind] = None
            else:
                blob = row[0]
                count = len(blob) // 4
                out[vec_kind] = list(struct.unpack(f"{count}f", blob))
        return out

    def _row_to_module(self, row: sqlite3.Row) -> ModuleRecord:
        vecs = self._vectors_for("modules", row["rowid"])
        return ModuleRecord(
            repo=row["repo"],
            path=row["path"],
            language=Language(row["language"]),
            docstring=row["docstring"],
            imports=json.loads(row["imports"]),
            tags=json.loads(row["tags"]),
            updated_at=_parse_iso(row["updated_at"]),
            description_vec=vecs["description"],
            code_vec=vecs["code"],
            docstring_vec=vecs["docstring"],
        )

    def _row_to_function(self, row: sqlite3.Row) -> FunctionRecord:
        vecs = self._vectors_for("functions", row["rowid"])
        return FunctionRecord(
            repo=row["repo"],
            guid=row["guid"],
            name=row["name"],
            module_path=row["module_path"],
            line=row["line"],
            end_line=row["end_line"],
            docstring=row["docstring"],
            args=json.loads(row["args"]),
            returns=row["returns"],
            is_async=bool(row["is_async"]),
            tags=json.loads(row["tags"]),
            updated_at=_parse_iso(row["updated_at"]),
            description_vec=vecs["description"],
            code_vec=vecs["code"],
            docstring_vec=vecs["docstring"],
        )

    def _row_to_class(self, row: sqlite3.Row) -> ClassRecord:
        vecs = self._vectors_for("classes", row["rowid"])
        return ClassRecord(
            repo=row["repo"],
            guid=row["guid"],
            name=row["name"],
            module_path=row["module_path"],
            line=row["line"],
            end_line=row["end_line"],
            docstring=row["docstring"],
            tags=json.loads(row["tags"]),
            updated_at=_parse_iso(row["updated_at"]),
            description_vec=vecs["description"],
            code_vec=vecs["code"],
            docstring_vec=vecs["docstring"],
        )
