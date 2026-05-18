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


def test_enrich_truncates_oversized_code_for_embedder(tmp_path: Path):
    """Whole-module sources past the embedder budget must be truncated.

    Reproduces the real-model failure where nomic-embed-text returned
    HTTP 500 on 40K-character module sources. The truncation is in
    enrich._truncate_for_embed; this test asserts the embedder never
    sees more than MAX_EMBED_CHARS + the marker string.
    """
    from otter_docs.enrich import MAX_EMBED_CHARS

    # 30K lines × ~10 chars/line = ~300K characters — definitely over budget.
    big_body = "    return 1\n" * 30000
    (tmp_path / "huge.py").write_text(f"def f():\n{big_body}")

    seen_text_lengths: list[int] = []
    class _RecordingEmbedder:
        @property
        def dim(self) -> int:
            return 8
        def embed(self, texts):
            for t in texts:
                seen_text_lengths.append(len(t))
            return [[1.0] + [0.0] * 7 for _ in texts]

    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        report = repo.enrich(FakeLLMClient(), _RecordingEmbedder())
        assert report.errors == []
        # Every embed call received text within budget (plus the truncation
        # marker — about 40 chars of overhead).
        budget = MAX_EMBED_CHARS + 100
        assert max(seen_text_lengths) <= budget, (
            f"embedder saw text of length {max(seen_text_lengths)} > {budget}"
        )


def test_truncate_for_embed_preserves_short_text():
    from otter_docs.enrich import _truncate_for_embed
    short = "def f(): pass"
    assert _truncate_for_embed(short) == short


def test_truncate_for_embed_keeps_head_and_tail():
    from otter_docs.enrich import MAX_EMBED_CHARS, _truncate_for_embed
    head = "HEAD_MARKER_" * 1000  # at the front
    tail = "_TAIL_MARKER_" * 1000  # at the back
    middle = "X" * MAX_EMBED_CHARS  # so total length >> budget
    text = head + middle + tail
    result = _truncate_for_embed(text)
    assert len(result) <= MAX_EMBED_CHARS + 100  # + truncation marker
    assert "HEAD_MARKER_" in result
    assert "_TAIL_MARKER_" in result
    assert "truncated for embedding" in result


class _ModeledEmb(FakeEmbeddingClient):
    """A deterministic fake that also advertises a `.model` so the
    embedding cache key can distinguish embedders."""

    def __init__(self, model: str, dim: int = 8) -> None:
        super().__init__(dim=dim)
        self.model = model


def test_embed_model_id_falls_back_to_classname():
    from otter_docs.embedcache import embed_model_id

    assert embed_model_id(FakeEmbeddingClient(dim=8)) == "FakeEmbeddingClient"
    assert embed_model_id(_ModeledEmb("nomic-x", dim=8)) == "nomic-x"


def test_embedding_is_incremental_via_cache(tmp_path: Path):
    """A no-op re-run makes ZERO embedder calls — the whole point."""
    _write_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        report_a = repo.enrich(llm, emb)
        total = (
            report_a.modules_enriched
            + report_a.functions_enriched
            + report_a.classes_enriched
        )
        assert report_a.embedding_calls == total
        assert report_a.embedding_cache_hits == 0
        batches_after_first = len(emb.calls)

        report_b = repo.enrich(llm, emb)
        # Second pass: every symbol is an embedding-cache hit, the
        # embedder is never called again, but rows are still upserted.
        assert report_b.embedding_calls == 0
        assert report_b.embedding_cache_hits == total
        assert len(emb.calls) == batches_after_first
        assert (
            report_b.modules_enriched
            + report_b.functions_enriched
            + report_b.classes_enriched
        ) == total


def test_changed_symbol_re_embeds_only_itself(tmp_path: Path):
    """Editing one function re-embeds that function + its module only;
    everything else stays a cache hit."""
    _write_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        repo.enrich(llm, emb)  # cold: 5 symbols embedded

        # Edit the *second* function so `add`'s slice stays byte-identical.
        (tmp_path / "math.py").write_text(
            "def add(a, b):\n    return a + b\n\n"
            "def subtract(a, b):\n    return (a - b) + 0  # changed\n"
        )
        repo.scan()
        report = repo.enrich(llm, emb)

        # Re-embedded: math.py module (file content changed) + subtract.
        # Cache hits: math.py::add, io.py module, io.py::read_file.
        assert report.embedding_calls == 2
        assert report.embedding_cache_hits == 3


def test_embed_cache_keyed_on_model(tmp_path: Path):
    """Same code, different embedder model → cache miss (re-embed),
    and switching back hits again. Stale-dim vectors must never be
    served for a different model."""
    _write_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        a1 = repo.enrich(llm, _ModeledEmb("model-a", dim=8))
        total = (
            a1.modules_enriched + a1.functions_enriched + a1.classes_enriched
        )
        assert a1.embedding_calls == total  # cold

        b = repo.enrich(llm, _ModeledEmb("model-b", dim=8))
        # Different model → every symbol re-embeds despite identical code.
        assert b.embedding_calls == total
        assert b.embedding_cache_hits == 0

        a2 = repo.enrich(llm, _ModeledEmb("model-a", dim=8))
        # model-a's entries are still there and still valid.
        assert a2.embedding_calls == 0
        assert a2.embedding_cache_hits == total


def test_embed_cache_persists_in_sqlite_graph_db(tmp_path: Path):
    """The cache lives in graph.db, so a fresh Repo over the same dir
    reuses vectors without re-embedding."""
    (tmp_path / "a.py").write_text("def hello():\n    return 'hi'\n")
    data_dir = tmp_path / ".otter-docs"
    data_dir.mkdir()
    db = data_dir / "graph.db"

    # First process: file-backed backend (cache lives in this graph.db).
    with Repo(tmp_path, backend=SqliteBackend(db, vector_dim=8)) as repo:
        repo.scan()
        first = repo.enrich(FakeLLMClient(), FakeEmbeddingClient(dim=8))
        assert first.errors == []
        assert first.embedding_calls > 0

    # Second process: brand-new Repo + backend + clients over the same file.
    with Repo(tmp_path, backend=SqliteBackend(db, vector_dim=8)) as repo:
        repo.scan()
        second = repo.enrich(FakeLLMClient(), FakeEmbeddingClient(dim=8))
        assert second.errors == []
        # Cache lived in graph.db across the two opens: no embedder call,
        # every symbol an embed-cache hit.
        assert second.embedding_calls == 0
        assert second.embedding_cache_hits == (
            second.modules_enriched
            + second.functions_enriched
            + second.classes_enriched
        )


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
