"""SqliteBackend test suite — schema, CRUD, vectors, traversal."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from otter_docs.backends import SqliteBackend
from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language,
    ModuleRecord,
    VectorKind,
)
from tests.conftest import unit


# ── migration ───────────────────────────────────────────────────────────────


def test_migration_idempotent(vector_dim):
    """Reconnecting to an existing db must not error."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        with SqliteBackend(path, vector_dim=vector_dim) as b1:
            b1.add_module(ModuleRecord(repo="r", path="a.py", language=Language.PYTHON))
        # Reconnect — migration runs again, must be a no-op
        with SqliteBackend(path, vector_dim=vector_dim) as b2:
            assert b2.get_module("r", "a.py") is not None
    finally:
        import os
        os.unlink(path)


def test_vector_dim_mismatch_raises(vector_dim):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    try:
        with SqliteBackend(path, vector_dim=vector_dim):
            pass
        with pytest.raises(ValueError, match="vector_dim mismatch"):
            with SqliteBackend(path, vector_dim=vector_dim + 1):
                pass
    finally:
        os.unlink(path)


# ── module round-trip ───────────────────────────────────────────────────────


def test_module_round_trip(backend, vector_dim):
    m = ModuleRecord(
        repo="v3", path="pkg/a.py", language=Language.PYTHON,
        docstring="hello", imports=["os", "sys"], tags=["util"],
        updated_at=datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc),
        description_vec=unit([1.0, 0.0], vector_dim),
    )
    backend.add_module(m)
    got = backend.get_module("v3", "pkg/a.py")
    assert got is not None
    assert got.docstring == "hello"
    assert got.imports == ["os", "sys"]
    assert got.tags == ["util"]
    assert got.description_vec is not None
    assert len(got.description_vec) == vector_dim
    assert got.code_vec is None
    assert got.docstring_vec is None


def test_module_upsert_updates(backend):
    m = ModuleRecord(repo="r", path="a.py", language=Language.PYTHON, docstring="v1")
    backend.add_module(m)
    m2 = ModuleRecord(repo="r", path="a.py", language=Language.PYTHON, docstring="v2")
    backend.add_module(m2)
    got = backend.get_module("r", "a.py")
    assert got.docstring == "v2"
    # only one row exists
    assert len(list(backend.list_modules("r"))) == 1


# ── function round-trip ─────────────────────────────────────────────────────


def test_function_round_trip(backend, vector_dim):
    f = FunctionRecord(
        repo="v3", guid="g-1", name="hello", module_path="a.py",
        line=10, end_line=20, docstring="hi", args=["x", "y"],
        returns="int", is_async=True, tags=["io", "async"],
        description_vec=unit([1.0, 0.5], vector_dim),
        code_vec=unit([0.0, 1.0], vector_dim),
    )
    backend.add_function(f)
    got = backend.get_function("v3", "g-1")
    assert got is not None
    assert got.name == "hello"
    assert got.is_async is True
    assert got.args == ["x", "y"]
    assert got.returns == "int"
    assert got.description_vec is not None
    assert got.code_vec is not None
    assert got.docstring_vec is None


def test_function_dim_mismatch_raises(backend, vector_dim):
    f = FunctionRecord(
        repo="r", guid="g", name="n", module_path="a.py", line=1, end_line=2,
        description_vec=[0.1] * (vector_dim + 1),
    )
    with pytest.raises(ValueError, match="Vector dim mismatch"):
        backend.add_function(f)


# ── class round-trip ────────────────────────────────────────────────────────


def test_class_round_trip(backend):
    c = ClassRecord(
        repo="v3", guid="c-1", name="Foo", module_path="a.py",
        line=5, end_line=50, docstring="a class", tags=["model"],
    )
    backend.add_class(c)
    got = backend.get_class("v3", "c-1")
    assert got is not None
    assert got.name == "Foo"
    assert got.tags == ["model"]


# ── list operations ─────────────────────────────────────────────────────────


def test_list_modules_repo_filter(backend):
    backend.add_module(ModuleRecord(repo="r1", path="a.py", language=Language.PYTHON))
    backend.add_module(ModuleRecord(repo="r1", path="b.py", language=Language.PYTHON))
    backend.add_module(ModuleRecord(repo="r2", path="a.py", language=Language.PYTHON))

    r1_paths = [m.path for m in backend.list_modules("r1")]
    all_paths = [(m.repo, m.path) for m in backend.list_modules()]
    assert r1_paths == ["a.py", "b.py"]
    assert len(all_paths) == 3


def test_list_functions_ordered_by_line(backend):
    for i in [30, 10, 20]:
        backend.add_function(FunctionRecord(
            repo="r", guid=f"g-{i}", name=f"fn{i}", module_path="a.py",
            line=i, end_line=i + 5,
        ))
    fns = list(backend.list_functions("r"))
    assert [f.line for f in fns] == [10, 20, 30]


# ── vector similarity ──────────────────────────────────────────────────────


def test_find_similar_ranks_by_proximity(backend, vector_dim):
    backend.add_function(FunctionRecord(
        repo="r", guid="a", name="alpha", module_path="m.py",
        line=1, end_line=2, description_vec=unit([1.0, 0.0], vector_dim),
    ))
    backend.add_function(FunctionRecord(
        repo="r", guid="b", name="beta", module_path="m.py",
        line=3, end_line=4, description_vec=unit([0.9, 0.1], vector_dim),
    ))
    backend.add_function(FunctionRecord(
        repo="r", guid="c", name="gamma", module_path="m.py",
        line=5, end_line=6, description_vec=unit([0.0, 1.0], vector_dim),
    ))

    hits = backend.find_similar(
        "r", unit([1.0, 0.0], vector_dim),
        vector_kind=VectorKind.DESCRIPTION, node_kind="function", k=5,
    )
    ids = [h.node_id for h in hits]
    # a is identical to the query, b is close, c is orthogonal
    assert ids[0] == "a"
    assert ids[1] == "b"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert 0.45 < hits[2].similarity < 0.55, f"orthogonal expected ~0.5, got {hits[2].similarity}"


def test_find_similar_min_threshold_filters(backend, vector_dim):
    backend.add_function(FunctionRecord(
        repo="r", guid="a", name="alpha", module_path="m.py",
        line=1, end_line=2, description_vec=unit([1.0, 0.0], vector_dim),
    ))
    backend.add_function(FunctionRecord(
        repo="r", guid="c", name="gamma", module_path="m.py",
        line=5, end_line=6, description_vec=unit([0.0, 1.0], vector_dim),
    ))
    hits = backend.find_similar(
        "r", unit([1.0, 0.0], vector_dim),
        vector_kind=VectorKind.DESCRIPTION, node_kind="function",
        k=5, min_similarity=0.9,
    )
    # only "a" (similarity 1.0) clears the bar; "c" at ~0.5 doesn't
    assert [h.node_id for h in hits] == ["a"]


def test_find_similar_dim_mismatch_raises(backend, vector_dim):
    with pytest.raises(ValueError, match="dim"):
        backend.find_similar(
            "r", [0.1] * (vector_dim + 1),
            vector_kind=VectorKind.DESCRIPTION, node_kind="function",
        )


def test_find_similar_respects_repo_scope(backend, vector_dim):
    backend.add_function(FunctionRecord(
        repo="r1", guid="a", name="alpha", module_path="m.py",
        line=1, end_line=2, description_vec=unit([1.0, 0.0], vector_dim),
    ))
    backend.add_function(FunctionRecord(
        repo="r2", guid="a", name="alpha", module_path="m.py",
        line=1, end_line=2, description_vec=unit([1.0, 0.0], vector_dim),
    ))
    r1_hits = backend.find_similar(
        "r1", unit([1.0, 0.0], vector_dim),
        vector_kind=VectorKind.DESCRIPTION, node_kind="function",
    )
    assert len(r1_hits) == 1
    # We can't distinguish which "a" we got back without joining further, but
    # the query against r1 must not have returned r2's row.


# ── edges + traversal ───────────────────────────────────────────────────────


def test_edge_and_callers(backend):
    backend.add_function(FunctionRecord(
        repo="r", guid="caller", name="caller", module_path="m.py",
        line=1, end_line=2,
    ))
    backend.add_function(FunctionRecord(
        repo="r", guid="callee", name="callee", module_path="m.py",
        line=3, end_line=4,
    ))
    backend._add_edge_with_repo(
        Edge(kind="CALLS", src_id="caller", dst_id="callee"), repo="r",
    )
    assert backend.callers_of("r", "callee") == ["caller"]


def test_edges_from_and_to(backend):
    backend._add_edge_with_repo(Edge(kind="CALLS", src_id="a", dst_id="b"), repo="r")
    backend._add_edge_with_repo(Edge(kind="IMPORTS", src_id="a", dst_id="c"), repo="r")

    from_a_all = backend.edges_from("r", "a")
    from_a_calls = backend.edges_from("r", "a", kind="CALLS")
    to_b = backend.edges_to("r", "b")
    assert len(from_a_all) == 2
    assert len(from_a_calls) == 1 and from_a_calls[0].dst_id == "b"
    assert len(to_b) == 1 and to_b[0].kind == "CALLS"


def test_edge_idempotent(backend):
    e = Edge(kind="CALLS", src_id="a", dst_id="b")
    backend._add_edge_with_repo(e, repo="r")
    backend._add_edge_with_repo(e, repo="r")  # second insert should be a no-op
    assert len(backend.edges_from("r", "a")) == 1


# ── reset ───────────────────────────────────────────────────────────────────


def test_reset_global_wipes_everything(backend):
    backend.add_module(ModuleRecord(repo="r1", path="a.py", language=Language.PYTHON))
    backend.add_module(ModuleRecord(repo="r2", path="a.py", language=Language.PYTHON))
    backend.reset()
    assert list(backend.list_modules()) == []


def test_reset_scoped_preserves_other_repos(backend):
    backend.add_module(ModuleRecord(repo="r1", path="a.py", language=Language.PYTHON))
    backend.add_module(ModuleRecord(repo="r2", path="a.py", language=Language.PYTHON))
    backend.reset(repo="r1")
    remaining = [m.repo for m in backend.list_modules()]
    assert remaining == ["r2"]


# ── raw query escape hatch ──────────────────────────────────────────────────


def test_query_raw(backend):
    backend.add_module(ModuleRecord(repo="r", path="a.py", language=Language.PYTHON))
    rows = backend.query("SELECT path, repo FROM code_modules WHERE repo = ?", ["r"])
    assert rows == [{"path": "a.py", "repo": "r"}]


# ── unconnected guards ──────────────────────────────────────────────────────


def test_unconnected_backend_raises(vector_dim):
    be = SqliteBackend(":memory:", vector_dim=vector_dim)
    with pytest.raises(RuntimeError, match="not connected"):
        _ = be.conn
