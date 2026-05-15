"""Phase 9 — renderer + marker-injection tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.clients import FakeEmbeddingClient, FakeLLMClient
from otter_docs.render import bootstrap_document, inject, registry, sections_in


def _repo(tmp_path: Path) -> Repo:
    (tmp_path / "a.py").write_text(
        '"""Module A."""\n'
        "import os\n\n"
        "def used():\n    return 1\n\n"
        "def caller():\n    return used()\n\n"
        "def orphan():\n    return 99\n"
    )
    (tmp_path / "b.py").write_text("from a import used\n\ndef go():\n    return used()\n")
    r = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    r.scan()
    return r


# ── registry ────────────────────────────────────────────────────────────


def test_all_five_renderers_registered():
    assert set(registry()) == {
        "system_overview",
        "findings_summary",
        "redundancy_report",
        "dependency_graph",
        "architecture_smells",
    }


# ── individual renderers ────────────────────────────────────────────────


def test_system_overview_mentions_counts(tmp_path: Path):
    r = _repo(tmp_path)
    out = r.render("system_overview")
    assert "modules" in out and "functions" in out
    assert "python" in out
    r.close()


def test_findings_summary_table(tmp_path: Path):
    r = _repo(tmp_path)
    out = r.render("findings_summary")
    assert "| kind | count | mean confidence |" in out
    assert "dead_code" in out
    r.close()


def test_redundancy_report_handles_no_enrich(tmp_path: Path):
    r = _repo(tmp_path)
    out = r.render("redundancy_report")
    assert "needs enrich()" in out
    r.close()


def test_redundancy_report_after_enrich(tmp_path: Path):
    (tmp_path / "dup.py").write_text(
        "def add_one(x):\n    return x + 1\n\n"
        "def increment(x):\n    return x + 1\n"
    )
    r = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    r.scan()
    r.enrich(FakeLLMClient(), FakeEmbeddingClient(dim=8))
    out = r.render("redundancy_report")
    # With fake embeddings add_one/increment may or may not pair depending
    # on hash geometry; the renderer must not crash and must produce text.
    assert isinstance(out, str) and out
    r.close()


def test_dependency_graph_is_mermaid(tmp_path: Path):
    r = _repo(tmp_path)
    out = r.render("dependency_graph")
    assert "```mermaid" in out
    assert "graph LR" in out
    r.close()


def test_architecture_smells_sections(tmp_path: Path):
    r = _repo(tmp_path)
    out = r.render("architecture_smells")
    assert "### Largest functions" in out
    assert "### Call-graph hubs" in out
    r.close()


# ── marker injection ────────────────────────────────────────────────────


def test_inject_replaces_only_generated_block():
    doc = (
        "# Title\n\nHuman intro paragraph.\n\n"
        "<!-- BEGIN GENERATED:x -->\nOLD\n<!-- END GENERATED:x -->\n\n"
        "Human outro paragraph.\n"
    )
    out = inject(doc, name="x", body="NEW")
    assert "NEW" in out
    assert "OLD" not in out
    # Human prose preserved verbatim.
    assert "Human intro paragraph." in out
    assert "Human outro paragraph." in out


def test_inject_is_idempotent():
    doc = "<!-- BEGIN GENERATED:x -->\nA\n<!-- END GENERATED:x -->\n"
    once = inject(doc, name="x", body="B")
    twice = inject(once, name="x", body="B")
    assert once == twice


def test_inject_appends_when_marker_absent():
    doc = "# Title\n\nSome prose.\n"
    out = inject(doc, name="newsec", body="GEN")
    assert "Some prose." in out
    assert "<!-- BEGIN GENERATED:newsec -->" in out
    assert "GEN" in out


def test_inject_preserves_other_sections():
    doc = (
        "<!-- BEGIN GENERATED:a -->\nAA\n<!-- END GENERATED:a -->\n"
        "<!-- BEGIN GENERATED:b -->\nBB\n<!-- END GENERATED:b -->\n"
    )
    out = inject(doc, name="a", body="A2")
    assert "A2" in out
    assert "BB" in out  # b section untouched
    assert "AA" not in out


def test_bootstrap_document_has_all_markers():
    doc = bootstrap_document(title="MyRepo", sections=["system_overview", "findings_summary"])
    assert "# MyRepo" in doc
    assert sections_in(doc) == ["system_overview", "findings_summary"]


# ── full document round-trip ────────────────────────────────────────────


def test_render_document_bootstrap_then_update(tmp_path: Path):
    r = _repo(tmp_path)
    doc_path = tmp_path / "SYSTEM.md"
    assert not doc_path.exists()

    first = r.render_document(doc_path)
    assert doc_path.exists()
    # All five sections present.
    assert set(sections_in(first)) == set(registry())
    assert "modules" in first  # system_overview rendered

    # Inject human prose, then re-render — prose must survive.
    edited = first.replace(
        "# " + r.name,
        "# " + r.name + "\n\nHUMAN NOTE: do not delete me.",
    )
    doc_path.write_text(edited, encoding="utf-8")
    second = r.render_document(doc_path)
    assert "HUMAN NOTE: do not delete me." in second
    assert set(sections_in(second)) == set(registry())
    r.close()


def test_render_document_custom_sections(tmp_path: Path):
    r = _repo(tmp_path)
    doc_path = tmp_path / "DOC.md"
    out = r.render_document(doc_path, title="Custom", sections=["system_overview"])
    assert sections_in(out) == ["system_overview"]
    assert "# Custom" in out
    r.close()
