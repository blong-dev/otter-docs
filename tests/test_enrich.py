"""Enrichment pass tests.

These verify the end-to-end pipeline using fake clients:

    Repo.scan() → Repo.enrich(FakeLLM, FakeEmb) → find_similar works
"""

from __future__ import annotations

from pathlib import Path

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.clients import FakeEmbeddingClient, FakeLLMClient
from otter_docs.models import VectorKind


def _write_repo(tmp_path: Path) -> Path:
    (tmp_path / "math.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def subtract(a, b):\n    return a - b\n"
    )
    (tmp_path / "io.py").write_text(
        '"""IO helpers."""\n\n'
        "def read_file(path):\n    with open(path) as f:\n        return f.read()\n"
    )
    return tmp_path


def test_enrich_writes_three_vectors_per_function(tmp_path: Path):
    _write_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)

    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        report = repo.enrich(llm, emb)
        assert report.errors == []
        assert report.functions_enriched == 3  # add, subtract, read_file
        # Spot-check one function carries all three vector slots.
        fns = list(repo.graph.list_functions(repo.name))
        for fn in fns:
            assert fn.description_vec is not None
            assert len(fn.description_vec) == 8
            assert fn.code_vec is not None
            # docstring_vec is None for functions without docstrings
        # read_file has no docstring; add/subtract too
        # but the *module* io.py has one
        io_mod = repo.graph.get_module(repo.name, "io.py")
        assert io_mod.docstring_vec is not None


def test_enrich_is_idempotent_via_cache(tmp_path: Path):
    _write_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        report_a = repo.enrich(llm, emb)
        first_llm_calls = len(llm.calls)
        # Run again — every symbol's content hash should hit the cache.
        report_b = repo.enrich(llm, emb)
        # LLM was called only during the first pass; second pass is all hits.
        assert len(llm.calls) == first_llm_calls
        # Cache-hit count on the second pass should equal the total
        # symbols enriched on either pass.
        total = (
            report_a.modules_enriched
            + report_a.functions_enriched
            + report_a.classes_enriched
        )
        assert report_b.cache_hits == total


def test_enrich_then_find_similar_works(tmp_path: Path):
    """End-to-end: scan + enrich + find_similar returns the same function."""
    (tmp_path / "a.py").write_text(
        "def hello_world():\n    return 'hi'\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        repo.enrich(llm, emb)
        fns = list(repo.graph.list_functions(repo.name))
        assert len(fns) == 1
        # Query with the function's own description_vec — it must rank itself first.
        query_vec = fns[0].description_vec
        hits = repo.graph.find_similar(
            repo.name, query_vec,
            vector_kind=VectorKind.DESCRIPTION, node_kind="function", k=5,
        )
        assert len(hits) == 1
        assert hits[0].node_id == fns[0].guid
        assert hits[0].similarity > 0.999  # identical vectors


def test_enrich_handles_empty_repo_gracefully(tmp_path: Path):
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        report = repo.enrich(llm, emb)
        assert report.errors == []
        assert report.modules_enriched == 0
        assert report.functions_enriched == 0


def test_enrich_class_records_get_vectors(tmp_path: Path):
    (tmp_path / "models.py").write_text(
        "class User:\n"
        "    \"\"\"A user.\"\"\"\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), FakeEmbeddingClient(dim=8))
        classes = list(repo.graph.list_classes(repo.name))
        assert len(classes) == 1
        assert classes[0].description_vec is not None
        assert classes[0].code_vec is not None
        assert classes[0].docstring_vec is not None  # has a docstring


def test_enrich_skips_vec_when_no_docstring(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f(): return 1\n")
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), FakeEmbeddingClient(dim=8))
        fn = next(iter(repo.graph.list_functions(repo.name)))
        assert fn.docstring_vec is None
        # But the other two vectors are populated.
        assert fn.description_vec is not None
        assert fn.code_vec is not None
