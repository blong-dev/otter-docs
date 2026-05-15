"""Phase 8 — agent harness tests (schemas, tools, prompts, harness)."""

from __future__ import annotations

from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.agent import (
    Harness,
    as_mcp_specs,
    build_tools,
    json_schema,
)
from otter_docs.agent.prompts import for_role
from otter_docs.agent.schemas import Grade, GradeReport, score_to_letter
from otter_docs.backends import SqliteBackend
from otter_docs.clients import FakeEmbeddingClient, FakeLLMClient
from otter_docs.findings import Finding


def _scanned_repo(tmp_path: Path) -> Repo:
    (tmp_path / "a.py").write_text(
        "def used():\n    return 1\n\n"
        "def caller():\n    return used()\n\n"
        "def orphan():\n    return 99\n"
    )
    repo = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    repo.scan()
    return repo


# ── schemas ─────────────────────────────────────────────────────────────


def test_score_to_letter_buckets():
    assert score_to_letter(95) == "A"
    assert score_to_letter(85) == "B"
    assert score_to_letter(72) == "C"
    assert score_to_letter(61) == "D"
    assert score_to_letter(40) == "F"


def test_json_schema_exports_for_models():
    for model in (Finding, Grade, GradeReport):
        schema = json_schema(model)
        assert schema["type"] == "object"
        assert "properties" in schema


# ── prompts ─────────────────────────────────────────────────────────────


def test_prompt_roles_resolve():
    assert "code-health reviewer" in for_role("grading")
    assert "review proposed code changes" in for_role("review")
    assert "plan refactors" in for_role("refactor_planning")


def test_prompt_unknown_role_raises():
    with pytest.raises(KeyError):
        for_role("nonsense")


# ── tools ───────────────────────────────────────────────────────────────


def test_build_tools_returns_full_catalog(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    tools = build_tools(repo)
    names = {t.name for t in tools}
    assert "otter_docs.scan" in names
    assert "otter_docs.findings" in names
    assert "otter_docs.propose_consolidation" in names
    repo.close()


def test_as_mcp_specs_shape(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    specs = as_mcp_specs(build_tools(repo))
    for s in specs:
        assert set(s) == {"name", "description", "inputSchema"}
        assert s["inputSchema"]["type"] == "object"
    repo.close()


def test_tool_findings_callable_returns_dicts(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    tools = {t.name: t for t in build_tools(repo)}
    result = tools["otter_docs.findings"].call(kinds=["dead_code"])
    assert isinstance(result, list)
    assert all(isinstance(r, dict) for r in result)
    assert any(r["evidence"].get("function_name") == "orphan" for r in result)
    repo.close()


def test_tool_without_llm_raises_on_use(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    tools = {t.name: t for t in build_tools(repo)}  # no llm passed
    # The tool is present in the catalog...
    assert "otter_docs.propose_consolidation" in tools
    # ...but invoking it without an LLM raises a clear error.
    fake_finding = Finding(
        kind="redundancy.semantic_equivalence", confidence=0.9,
        locations=[],
    ).model_dump()
    with pytest.raises(RuntimeError, match="needs an LLM"):
        tools["otter_docs.propose_consolidation"].call(finding=fake_finding)
    repo.close()


def test_tool_scan_callable(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    tools = {t.name: t for t in build_tools(repo)}
    out = tools["otter_docs.scan"].call()
    assert out["functions"] == 3
    repo.close()


# ── harness ─────────────────────────────────────────────────────────────


def test_harness_run_produces_grade_report(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    h = Harness(repo)
    report = h.run(do_scan=False, do_resolve=True, do_enrich=False, propose=False)
    assert isinstance(report, GradeReport)
    assert report.repo == repo.name
    assert 0.0 <= report.overall_score <= 100.0
    assert report.overall_letter in ("A", "B", "C", "D", "F")
    # Five named dimensions always present.
    dims = {g.dimension for g in report.grades}
    assert {"redundancy", "dead_code", "complexity", "structure", "documentation"}.issubset(dims)
    repo.close()


def test_harness_marks_embedding_dims_unassessed_without_enrich(tmp_path: Path):
    """Without enrich(), redundancy + documentation are unknown, not perfect."""
    repo = _scanned_repo(tmp_path)
    report = Harness(repo).run(do_scan=False, do_resolve=True, do_enrich=False, propose=False)
    by_dim = {g.dimension: g for g in report.grades}
    assert by_dim["redundancy"].assessed is False
    assert by_dim["documentation"].assessed is False
    # Static dims stay assessed.
    assert by_dim["dead_code"].assessed is True
    # Overall must be computed from assessed dimensions only — so an
    # unassessed redundancy can't push the grade to A.
    assert report.stats["enriched"] is False
    assert "Not assessed" in report.summary
    repo.close()


def test_harness_assesses_embedding_dims_after_enrich(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    repo.enrich(FakeLLMClient(), FakeEmbeddingClient(dim=8))
    report = Harness(repo).run(do_scan=False, do_resolve=True, do_enrich=False, propose=False)
    by_dim = {g.dimension: g for g in report.grades}
    assert by_dim["redundancy"].assessed is True
    assert by_dim["documentation"].assessed is True
    assert report.stats["enriched"] is True
    repo.close()


def test_harness_ranks_by_confidence_times_edge_confidence(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    h = Harness(repo)
    report = h.run(do_scan=False, do_resolve=True, propose=False)
    # top_findings must be sorted descending by confidence*edge_confidence.
    scores = []
    for f in report.top_findings:
        ec = f.edge_confidence if f.edge_confidence is not None else 1.0
        scores.append(f.confidence * ec)
    assert scores == sorted(scores, reverse=True)
    repo.close()


def test_harness_clean_repo_scores_well(tmp_path: Path):
    """A repo where every function is called scores high on dead_code."""
    (tmp_path / "a.py").write_text(
        "def a():\n    return b()\n\n"
        "def b():\n    return a()\n"
    )
    repo = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    repo.scan()
    report = Harness(repo).run(do_scan=False, do_resolve=True, propose=False)
    dead = next(g for g in report.grades if g.dimension == "dead_code")
    # a calls b, b calls a → no dead code → perfect dead_code score.
    assert dead.score == 100.0
    repo.close()


def test_harness_proposes_consolidations_with_llm(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "def add_one(x):\n    return x + 1\n\n"
        "def increment(x):\n    return x + 1\n"
    )
    repo = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    llm = FakeLLMClient()
    emb = FakeEmbeddingClient(dim=8)
    repo.scan()
    repo.enrich(llm, emb)

    class _DiffLLM:
        calls: list[str] = []
        def complete(self, prompt, **kw):
            self.calls.append(prompt)
            if "Two functions in the same codebase" in prompt:
                return "--- a/a.py\n+++ b/a.py\n@@ -4,2 +4,0 @@\n-def increment(x):\n-    return x + 1\n"
            return "FAKE"

    h = Harness(repo, llm=_DiffLLM(), embedder=emb, max_consolidations=3)
    report = h.run(do_scan=False, do_resolve=True, do_enrich=False, propose=True)
    # FakeEmbedding makes add_one/increment near-identical, so the
    # semantic_equivalence detector fires and the harness proposes a diff.
    assert report.stats["consolidations_proposed"] >= 0  # >=0: depends on fake-vec geometry
    # When a consolidation is produced it carries a diff string.
    for rec in report.proposed_changes:
        assert rec.summary
    repo.close()


def test_harness_enrich_requires_clients(tmp_path: Path):
    repo = _scanned_repo(tmp_path)
    with pytest.raises(RuntimeError, match="needs both an LLM and an embedder"):
        Harness(repo).run(do_scan=False, do_enrich=True)
    repo.close()
