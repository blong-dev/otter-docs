"""OD-6 — surface the Harness grade.

The grade is computed but was invisible; now Repo.grade() exposes it, a `grade`
renderer puts it in the document, and onboarding writes the letter into
status.json (printed by `status --manifest`).
"""

from __future__ import annotations

from pathlib import Path

from otter_docs import Repo
from otter_docs.agent.schemas import GradeReport
from otter_docs.backends import SqliteBackend
from otter_docs.render import registry


def _repo(tmp_path: Path) -> Repo:
    (tmp_path / "a.py").write_text(
        'def used(x):\n    """Add one to x."""\n    return x + 1\n\n'
        "def caller():\n    return used(3)\n"
    )
    r = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    r.scan()
    return r


def test_repo_grade_returns_report(tmp_path: Path):
    r = _repo(tmp_path)
    report = r.grade()
    assert isinstance(report, GradeReport)
    assert report.overall_letter in {"A", "B", "C", "D", "F"}
    assert 0.0 <= report.overall_score <= 100.0
    assert report.grades  # per-dimension breakdown present
    r.close()


def test_grade_renderer_registered():
    assert "grade" in list(registry())


def test_grade_renderer_output_shape(tmp_path: Path):
    r = _repo(tmp_path)
    out = r.render("grade")
    assert out.startswith("**Overall grade:")
    assert "| dimension | grade | score | findings |" in out
    # unassessed dimensions (need enrich) render as n/a, not a fake A
    assert "n/a" in out
    r.close()


def test_grade_appears_in_rendered_document(tmp_path: Path):
    r = _repo(tmp_path)
    doc = tmp_path / "OTTER.md"
    r.render_document(doc)
    body = doc.read_text()
    assert "BEGIN GENERATED:grade" in body
    assert "Overall grade:" in body
    r.close()
