"""Neo4jBackend integration tests.

Marked `integration` — skipped unless `--run-integration` is passed AND the
following env vars are set:

    NEO4J_URI       (default: bolt://localhost:7687)
    NEO4J_USER      (default: neo4j)
    NEO4J_PASSWORD  (required)

Each test scopes its data to repo='otter-test-neo4j' and cleans up before
and after so it can run safely against a shared Neo4j instance without
disturbing other graphs.
"""

from __future__ import annotations

import os

import pytest

from otter_docs.models import (
    Edge,
    FunctionRecord,
    Language,
    ModuleRecord,
    VectorKind,
)
from tests.conftest import unit


TEST_REPO = "otter-test-neo4j"


@pytest.fixture
def neo4j_vector_dim() -> int:
    """Neo4j integration tests use 768 to match real-world embedder output.

    Neo4j allows only one vector index per (label, property), so dim is
    database-wide. The v3 Neo4j we run integration tests against uses
    nomic-embed-text (768); we use the same here to avoid the test
    needing to drop and recreate indexes that other tooling depends on.
    """
    return 768


@pytest.fixture
def neo4j_backend(neo4j_vector_dim):
    """Live Neo4j 5.x backend, scoped to TEST_REPO."""
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not pwd:
        pytest.skip("NEO4J_PASSWORD not set")
    from otter_docs.backends import Neo4jBackend  # lazy import to skip cleanly

    be = Neo4jBackend(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=pwd,
        vector_dim=neo4j_vector_dim,
    )
    be.connect()
    be.reset(repo=TEST_REPO)
    try:
        yield be
    finally:
        be.reset(repo=TEST_REPO)
        be.close()


pytestmark = pytest.mark.integration


def test_module_round_trip(neo4j_backend, neo4j_vector_dim):
    m = ModuleRecord(
        repo=TEST_REPO, path="pkg/a.py", language=Language.PYTHON,
        docstring="hello", imports=["os"], tags=["util"],
        description_vec=unit([1.0, 0.0], neo4j_vector_dim),
    )
    neo4j_backend.add_module(m)
    got = neo4j_backend.get_module(TEST_REPO, "pkg/a.py")
    assert got is not None
    assert got.docstring == "hello"
    assert got.imports == ["os"]
    assert got.description_vec is not None
    assert len(got.description_vec) == neo4j_vector_dim


def test_function_round_trip(neo4j_backend, neo4j_vector_dim):
    f = FunctionRecord(
        repo=TEST_REPO, guid="g-1", name="hello", module_path="a.py",
        line=10, end_line=20, is_async=True, args=["x"],
        description_vec=unit([1.0, 0.0], neo4j_vector_dim),
    )
    neo4j_backend.add_function(f)
    got = neo4j_backend.get_function(TEST_REPO, "g-1")
    assert got is not None
    assert got.name == "hello"
    assert got.is_async is True
    assert got.args == ["x"]


def test_upsert(neo4j_backend):
    f = FunctionRecord(repo=TEST_REPO, guid="g", name="v1",
                       module_path="a.py", line=1, end_line=2)
    neo4j_backend.add_function(f)
    f2 = FunctionRecord(repo=TEST_REPO, guid="g", name="v2",
                        module_path="a.py", line=1, end_line=2)
    neo4j_backend.add_function(f2)
    got = neo4j_backend.get_function(TEST_REPO, "g")
    assert got.name == "v2"
    assert len(list(neo4j_backend.list_functions(TEST_REPO))) == 1


def test_find_similar_ranks_by_proximity(neo4j_backend, neo4j_vector_dim):
    import time

    neo4j_backend.add_function(FunctionRecord(
        repo=TEST_REPO, guid="a", name="alpha", module_path="m.py",
        line=1, end_line=2, description_vec=unit([1.0, 0.0], neo4j_vector_dim),
    ))
    neo4j_backend.add_function(FunctionRecord(
        repo=TEST_REPO, guid="b", name="beta", module_path="m.py",
        line=3, end_line=4, description_vec=unit([0.9, 0.1], neo4j_vector_dim),
    ))
    neo4j_backend.add_function(FunctionRecord(
        repo=TEST_REPO, guid="c", name="gamma", module_path="m.py",
        line=5, end_line=6, description_vec=unit([0.0, 1.0], neo4j_vector_dim),
    ))

    # Neo4j vector indexes update asynchronously — give them a beat.
    time.sleep(2)

    hits = neo4j_backend.find_similar(
        TEST_REPO, unit([1.0, 0.0], neo4j_vector_dim),
        vector_kind=VectorKind.DESCRIPTION, node_kind="function", k=5,
    )
    ids = [h.node_id for h in hits]
    assert ids[0] == "a"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-3)
    # orthogonal ~0.5 in Neo4j's normalized cosine
    c_hit = next(h for h in hits if h.node_id == "c")
    assert 0.45 < c_hit.similarity < 0.55


def test_callers_of(neo4j_backend):
    neo4j_backend.add_function(FunctionRecord(
        repo=TEST_REPO, guid="caller", name="caller", module_path="m.py",
        line=1, end_line=2,
    ))
    neo4j_backend.add_function(FunctionRecord(
        repo=TEST_REPO, guid="callee", name="callee", module_path="m.py",
        line=3, end_line=4,
    ))
    neo4j_backend._add_edge_with_repo(
        Edge(kind="CALLS", src_id="caller", dst_id="callee"),
        repo=TEST_REPO,
    )
    assert neo4j_backend.callers_of(TEST_REPO, "callee") == ["caller"]


def test_reset_scoped_to_repo(neo4j_backend, neo4j_vector_dim):
    neo4j_backend.add_module(ModuleRecord(
        repo=TEST_REPO, path="a.py", language=Language.PYTHON,
    ))
    # Sanity: it's there
    assert neo4j_backend.get_module(TEST_REPO, "a.py") is not None
    neo4j_backend.reset(repo=TEST_REPO)
    assert neo4j_backend.get_module(TEST_REPO, "a.py") is None
