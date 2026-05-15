"""Neo4j 5.x implementation of GraphBackend.

Uses Neo4j's native vector indexes (introduced in 5.11, stable in 5.13+)
for the three vector slots per symbol. Requires the `neo4j` Python driver,
installed via the optional `[neo4j]` extra:

    pip install otter-docs[neo4j]

Schema:

    (:CodeModule    {repo, path, language, docstring, imports, tags,
                     updated_at, description_vec, code_vec, docstring_vec})
    (:CodeFunction  {repo, guid, name, module_path, line, end_line,
                     docstring, args, returns, is_async, tags, updated_at,
                     description_vec, code_vec, docstring_vec})
    (:CodeClass     {repo, guid, name, module_path, ...})

    (a)-[:DEFINED_IN]->(b)
    (a)-[:IMPORTS]->(b)
    (a)-[:CALLS]->(b)
    (a)-[:MEMBER_OF]->(b)

Uniqueness: (repo, path) for modules; (repo, guid) for functions and classes.
Edges keyed by (repo, kind, src_id, dst_id).

Vector indexes (one per (label, vector_kind)):
    `module_description`, `module_code`, `module_docstring`,
    `function_description`, ...
all using cosine similarity. We never use Neo4j's deprecated VECTOR INDEX
syntax; only the modern CREATE VECTOR INDEX form.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language,
    ModuleRecord,
    SimilarityHit,
    VectorKind,
)

if TYPE_CHECKING:
    from neo4j import Driver, Session

DEFAULT_VECTOR_DIM = 768

_NODE_LABELS = {
    "module": "CodeModule",
    "function": "CodeFunction",
    "class": "CodeClass",
}


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _parse_iso(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class Neo4jBackend:
    """GraphBackend backed by a Neo4j instance.

    Parameters
    ----------
    uri :
        Bolt URI, e.g. `bolt://10.0.0.2:7687`.
    user, password :
        Auth credentials.
    database :
        Neo4j database name. Defaults to "neo4j".
    vector_dim :
        Dimensionality of the embeddings the backend will store. Must match
        the embedding client. Default 768 (nomic-embed-text).
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        vector_dim: int = DEFAULT_VECTOR_DIM,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.vector_dim = vector_dim
        self._driver: Driver | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def connect(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase  # local import: optional dep
        except ImportError as e:
            raise ImportError(
                "Neo4jBackend requires the `neo4j` package. "
                "Install via: pip install otter-docs[neo4j]"
            ) from e
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._driver.verify_connectivity()
        self._migrate()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Neo4jBackend:
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            raise RuntimeError("Backend not connected — call .connect() or use 'with'")
        return self._driver

    def _session(self) -> Session:
        return self.driver.session(database=self.database)

    def _migrate(self) -> None:
        """Idempotent: create uniqueness constraints + vector indexes.

        Neo4j allows only one vector index per (label, property), so
        vector_dim is effectively database-wide. If an existing CodeFunction
        description_vec index uses a different dim than we were constructed
        with, raise a clear error rather than silently writing vectors at
        the wrong size.
        """
        with self._session() as s:
            existing = s.run(
                "SHOW VECTOR INDEXES YIELD name, options "
                "WHERE name = 'function_description' "
                "RETURN options.indexConfig.`vector.dimensions` AS dim"
            ).single()
            if existing is not None and existing["dim"] != self.vector_dim:
                raise ValueError(
                    f"Neo4j vector_dim mismatch: existing index uses "
                    f"{existing['dim']}, this backend configured for "
                    f"{self.vector_dim}. Drop the existing indexes or "
                    f"match the dim."
                )
        with self._session() as s:
            # Uniqueness constraints.
            s.run(
                "CREATE CONSTRAINT module_repo_path IF NOT EXISTS "
                "FOR (m:CodeModule) REQUIRE (m.repo, m.path) IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT function_repo_guid IF NOT EXISTS "
                "FOR (f:CodeFunction) REQUIRE (f.repo, f.guid) IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT class_repo_guid IF NOT EXISTS "
                "FOR (c:CodeClass) REQUIRE (c.repo, c.guid) IS UNIQUE"
            )

            # Vector indexes per (label, vector_kind). Neo4j enforces
            # uniqueness on (label, property), so we can't run parallel
            # indexes at different dims on the same database; vector_dim
            # is therefore database-wide. Mismatches are caught below.
            for node_kind, label in _NODE_LABELS.items():
                for vec_kind in ("description", "code", "docstring"):
                    index_name = f"{node_kind}_{vec_kind}"
                    prop = f"{vec_kind}_vec"
                    s.run(
                        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                        f"FOR (n:{label}) ON n.{prop} "
                        f"OPTIONS {{ indexConfig: {{ "
                        f"`vector.dimensions`: $dim, "
                        f"`vector.similarity_function`: 'cosine' "
                        f"}} }}",
                        dim=self.vector_dim,
                    )

    # ── writes ───────────────────────────────────────────────────────

    def _module_props(self, m: ModuleRecord) -> dict[str, Any]:
        return {
            "repo": m.repo,
            "path": m.path,
            "language": m.language.value,
            "docstring": m.docstring,
            "imports": json.dumps(m.imports),
            "tags": json.dumps(m.tags),
            "updated_at": _iso(m.updated_at),
            "description_vec": m.description_vec,
            "code_vec": m.code_vec,
            "docstring_vec": m.docstring_vec,
        }

    def _function_props(self, f: FunctionRecord) -> dict[str, Any]:
        return {
            "repo": f.repo,
            "guid": f.guid,
            "name": f.name,
            "module_path": f.module_path,
            "line": f.line,
            "end_line": f.end_line,
            "docstring": f.docstring,
            "args": json.dumps(f.args),
            "returns": f.returns,
            "is_async": f.is_async,
            "tags": json.dumps(f.tags),
            "updated_at": _iso(f.updated_at),
            "description_vec": f.description_vec,
            "code_vec": f.code_vec,
            "docstring_vec": f.docstring_vec,
        }

    def _class_props(self, c: ClassRecord) -> dict[str, Any]:
        return {
            "repo": c.repo,
            "guid": c.guid,
            "name": c.name,
            "module_path": c.module_path,
            "line": c.line,
            "end_line": c.end_line,
            "docstring": c.docstring,
            "tags": json.dumps(c.tags),
            "updated_at": _iso(c.updated_at),
            "description_vec": c.description_vec,
            "code_vec": c.code_vec,
            "docstring_vec": c.docstring_vec,
        }

    def add_module(self, module: ModuleRecord) -> None:
        self._check_vec_dims(module)
        with self._session() as s:
            s.run(
                """
                MERGE (m:CodeModule {repo: $repo, path: $path})
                SET m += $props
                """,
                repo=module.repo, path=module.path,
                props=self._module_props(module),
            )

    def add_function(self, function: FunctionRecord) -> None:
        self._check_vec_dims(function)
        with self._session() as s:
            s.run(
                """
                MERGE (f:CodeFunction {repo: $repo, guid: $guid})
                SET f += $props
                """,
                repo=function.repo, guid=function.guid,
                props=self._function_props(function),
            )

    def add_class(self, cls: ClassRecord) -> None:
        self._check_vec_dims(cls)
        with self._session() as s:
            s.run(
                """
                MERGE (c:CodeClass {repo: $repo, guid: $guid})
                SET c += $props
                """,
                repo=cls.repo, guid=cls.guid,
                props=self._class_props(cls),
            )

    def add_edge(self, edge: Edge) -> None:
        # Same constraint as SqliteBackend: Edge needs a repo to scope
        # storage. Use _add_edge_with_repo for v0.1.
        raise NotImplementedError(
            "Edge needs a repo. Use _add_edge_with_repo(edge, repo=...) for v0.1."
        )

    def _add_edge_with_repo(self, edge: Edge, repo: str) -> None:
        # Find both endpoints across all three node kinds (we don't know
        # which label src/dst belong to without context). The MATCH covers
        # functions and classes (path-keyed modules don't appear as edge
        # endpoints in the standard relationships).
        with self._session() as s:
            s.run(
                """
                MATCH (a) WHERE a.repo = $repo
                    AND (a.guid = $src OR a.path = $src)
                MATCH (b) WHERE b.repo = $repo
                    AND (b.guid = $dst OR b.path = $dst)
                MERGE (a)-[r:`__KIND__`]->(b)
                SET r.updated_at = $now
                """.replace("__KIND__", edge.kind),
                repo=repo, src=edge.src_id, dst=edge.dst_id,
                now=datetime.now(UTC).isoformat(),
            )

    def _check_vec_dims(
        self, record: ModuleRecord | FunctionRecord | ClassRecord
    ) -> None:
        for vec in (record.description_vec, record.code_vec, record.docstring_vec):
            if vec is not None and len(vec) != self.vector_dim:
                raise ValueError(
                    f"Vector dim mismatch: got {len(vec)}, expected {self.vector_dim}"
                )

    # ── reads ────────────────────────────────────────────────────────

    def get_module(self, repo: str, path: str) -> ModuleRecord | None:
        with self._session() as s:
            row = s.run(
                "MATCH (m:CodeModule {repo: $repo, path: $path}) RETURN m",
                repo=repo, path=path,
            ).single()
        return self._node_to_module(row["m"]) if row else None

    def get_function(self, repo: str, guid: str) -> FunctionRecord | None:
        with self._session() as s:
            row = s.run(
                "MATCH (f:CodeFunction {repo: $repo, guid: $guid}) RETURN f",
                repo=repo, guid=guid,
            ).single()
        return self._node_to_function(row["f"]) if row else None

    def get_class(self, repo: str, guid: str) -> ClassRecord | None:
        with self._session() as s:
            row = s.run(
                "MATCH (c:CodeClass {repo: $repo, guid: $guid}) RETURN c",
                repo=repo, guid=guid,
            ).single()
        return self._node_to_class(row["c"]) if row else None

    def list_modules(self, repo: str | None = None) -> Iterator[ModuleRecord]:
        cypher = (
            "MATCH (m:CodeModule) RETURN m ORDER BY m.repo, m.path"
            if repo is None
            else "MATCH (m:CodeModule {repo: $repo}) RETURN m ORDER BY m.path"
        )
        with self._session() as s:
            for record in s.run(cypher, repo=repo):
                yield self._node_to_module(record["m"])

    def list_functions(self, repo: str | None = None) -> Iterator[FunctionRecord]:
        cypher = (
            "MATCH (f:CodeFunction) RETURN f ORDER BY f.repo, f.module_path, f.line"
            if repo is None
            else "MATCH (f:CodeFunction {repo: $repo}) RETURN f ORDER BY f.module_path, f.line"
        )
        with self._session() as s:
            for record in s.run(cypher, repo=repo):
                yield self._node_to_function(record["f"])

    def list_classes(self, repo: str | None = None) -> Iterator[ClassRecord]:
        cypher = (
            "MATCH (c:CodeClass) RETURN c ORDER BY c.repo, c.module_path, c.line"
            if repo is None
            else "MATCH (c:CodeClass {repo: $repo}) RETURN c ORDER BY c.module_path, c.line"
        )
        with self._session() as s:
            for record in s.run(cypher, repo=repo):
                yield self._node_to_class(record["c"])

    # ── graph traversal ──────────────────────────────────────────────

    def callers_of(self, repo: str, target_guid: str) -> list[str]:
        with self._session() as s:
            rows = s.run(
                "MATCH (caller)-[:CALLS]->(target) "
                "WHERE target.repo = $repo AND target.guid = $guid "
                "RETURN caller.guid AS guid",
                repo=repo, guid=target_guid,
            ).data()
        return [r["guid"] for r in rows if r["guid"] is not None]

    def edges_from(self, repo: str, src_id: str, kind: str | None = None) -> list[Edge]:
        if kind is None:
            cypher = (
                "MATCH (a)-[r]->(b) WHERE a.repo = $repo "
                "AND (a.guid = $src OR a.path = $src) "
                "RETURN type(r) AS kind, "
                "       coalesce(a.guid, a.path) AS src_id, "
                "       coalesce(b.guid, b.path) AS dst_id"
            )
        else:
            cypher = (
                f"MATCH (a)-[r:`{kind}`]->(b) WHERE a.repo = $repo "
                "AND (a.guid = $src OR a.path = $src) "
                "RETURN type(r) AS kind, "
                "       coalesce(a.guid, a.path) AS src_id, "
                "       coalesce(b.guid, b.path) AS dst_id"
            )
        with self._session() as s:
            return [
                Edge(kind=r["kind"], src_id=r["src_id"], dst_id=r["dst_id"])
                for r in s.run(cypher, repo=repo, src=src_id).data()
            ]

    def edges_to(self, repo: str, dst_id: str, kind: str | None = None) -> list[Edge]:
        if kind is None:
            cypher = (
                "MATCH (a)-[r]->(b) WHERE b.repo = $repo "
                "AND (b.guid = $dst OR b.path = $dst) "
                "RETURN type(r) AS kind, "
                "       coalesce(a.guid, a.path) AS src_id, "
                "       coalesce(b.guid, b.path) AS dst_id"
            )
        else:
            cypher = (
                f"MATCH (a)-[r:`{kind}`]->(b) WHERE b.repo = $repo "
                "AND (b.guid = $dst OR b.path = $dst) "
                "RETURN type(r) AS kind, "
                "       coalesce(a.guid, a.path) AS src_id, "
                "       coalesce(b.guid, b.path) AS dst_id"
            )
        with self._session() as s:
            return [
                Edge(kind=r["kind"], src_id=r["src_id"], dst_id=r["dst_id"])
                for r in s.run(cypher, repo=repo, dst=dst_id).data()
            ]

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
        if node_kind not in _NODE_LABELS:
            raise ValueError(
                f"node_kind must be one of {list(_NODE_LABELS)}, got {node_kind!r}"
            )
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Query vector dim {len(vector)} != backend dim {self.vector_dim}"
            )
        index_name = f"{node_kind}_{vector_kind.value}"
        id_property = "path" if node_kind == "module" else "guid"
        # Neo4j vector index returns native cosine similarity score in [0, 1]
        # (1 = identical). We over-fetch a bit to allow the repo filter and
        # min_similarity cutoff to still hit k results in most cases.
        overfetch = max(k * 3, 50)
        with self._session() as s:
            rows = s.run(
                f"""
                CALL db.index.vector.queryNodes($index, $overfetch, $vec)
                YIELD node, score
                WHERE node.repo = $repo
                RETURN node.{id_property} AS node_id, score
                ORDER BY score DESC
                LIMIT $k
                """,
                index=index_name, overfetch=overfetch, vec=vector,
                repo=repo, k=k,
            ).data()

        hits = [
            SimilarityHit(node_kind=node_kind, node_id=r["node_id"], similarity=r["score"])
            for r in rows
            if r["score"] >= min_similarity
        ]
        return hits

    # ── lifecycle helpers ────────────────────────────────────────────

    def reset(self, *, repo: str | None = None) -> None:
        with self._session() as s:
            if repo is None:
                s.run(
                    "MATCH (n) WHERE n:CodeModule OR n:CodeFunction OR n:CodeClass "
                    "DETACH DELETE n"
                )
            else:
                s.run(
                    "MATCH (n) WHERE (n:CodeModule OR n:CodeFunction OR n:CodeClass) "
                    "AND n.repo = $repo DETACH DELETE n",
                    repo=repo,
                )

    def query(self, raw: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        # Neo4j uses named parameters; we accept positional for protocol
        # parity but discourage it. Callers passing Cypher should prefer
        # session.run() directly via .driver.session() for full control.
        with self._session() as s:
            return s.run(raw).data()

    # ── node → model converters ──────────────────────────────────────

    @staticmethod
    def _node_to_module(node: Any) -> ModuleRecord:
        return ModuleRecord(
            repo=node["repo"],
            path=node["path"],
            language=Language(node["language"]),
            docstring=node.get("docstring", ""),
            imports=json.loads(node.get("imports", "[]")),
            tags=json.loads(node.get("tags", "[]")),
            updated_at=_parse_iso(node.get("updated_at")),
            description_vec=node.get("description_vec"),
            code_vec=node.get("code_vec"),
            docstring_vec=node.get("docstring_vec"),
        )

    @staticmethod
    def _node_to_function(node: Any) -> FunctionRecord:
        return FunctionRecord(
            repo=node["repo"],
            guid=node["guid"],
            name=node["name"],
            module_path=node["module_path"],
            line=node["line"],
            end_line=node["end_line"],
            docstring=node.get("docstring", ""),
            args=json.loads(node.get("args", "[]")),
            returns=node.get("returns", ""),
            is_async=bool(node.get("is_async", False)),
            tags=json.loads(node.get("tags", "[]")),
            updated_at=_parse_iso(node.get("updated_at")),
            description_vec=node.get("description_vec"),
            code_vec=node.get("code_vec"),
            docstring_vec=node.get("docstring_vec"),
        )

    @staticmethod
    def _node_to_class(node: Any) -> ClassRecord:
        return ClassRecord(
            repo=node["repo"],
            guid=node["guid"],
            name=node["name"],
            module_path=node["module_path"],
            line=node["line"],
            end_line=node["end_line"],
            docstring=node.get("docstring", ""),
            tags=json.loads(node.get("tags", "[]")),
            updated_at=_parse_iso(node.get("updated_at")),
            description_vec=node.get("description_vec"),
            code_vec=node.get("code_vec"),
            docstring_vec=node.get("docstring_vec"),
        )
