"""Tests for ReadmeDetector."""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.readme import ReadmeDetector


def _detect(tmp_path: Path):
    return ReadmeDetector().detect(repo="t", repo_root=tmp_path)


def test_no_readme_returns_none(tmp_path: Path):
    assert _detect(tmp_path) is None


def test_readme_md_with_h2(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "# my-project\n\n"
        "A short summary of what this does.\n"
        "Continued on the next line.\n\n"
        "## Install\n\n`pip install my-project`\n\n"
        "## Usage\n\nSee the docs.\n\n"
        "### Advanced\n\nDeep mode.\n"
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.path == "README.md"
    assert "short summary" in surface.summary
    assert "Continued on the next line" in surface.summary
    assert surface.h2_outline == ["Install", "Usage"]


def test_readme_in_docs_dir(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("# x\n\nfallback summary.\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.path == "docs/README.md"
    assert "fallback summary" in surface.summary


def test_root_readme_beats_docs_readme(tmp_path: Path):
    (tmp_path / "README.md").write_text("# root\n\nroot summary.\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("# x\n\ndocs summary.\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.path == "README.md"
    assert "root summary" in surface.summary


def test_summary_capped(tmp_path: Path):
    long_para = "word " * 300  # ~1500 chars
    (tmp_path / "README.md").write_text(f"# x\n\n{long_para}\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert len(surface.summary) <= 500
    assert surface.summary.endswith("…")


def test_readme_with_no_summary_only_heading(tmp_path: Path):
    (tmp_path / "README.md").write_text("# only title\n\n## section\n\nsection content.\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.summary == ""
    assert surface.h2_outline == ["section"]


def test_monorepo_subtree_readmes_require_manifest(tmp_path: Path):
    """No root README. Subtree READMEs are picked up only when the
    directory ALSO has a dependency manifest — that's the signal
    'this directory is a package, not just a doc folder.'

    Without this filter, repos pick up archive/backup READMEs as
    noise (the v3 validation surfaced this)."""
    # Real package (has manifest + README) — should be reported.
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text('{"dependencies": {}}')
    (tmp_path / "web" / "README.md").write_text("# web\n\nweb summary.\n")
    # Documentation-only README (no manifest) — should be skipped.
    (tmp_path / "philosophy").mkdir()
    (tmp_path / "philosophy" / "README.md").write_text("# musings\n\nthoughts.\n")
    # Archive README (no manifest) — should be skipped.
    (tmp_path / "_retired").mkdir()
    (tmp_path / "_retired" / "README.md").write_text("# old\n\nretired.\n")
    result = _detect(tmp_path)
    assert isinstance(result, list)
    paths = {r.path for r in result}
    assert paths == {"web/README.md"}


def test_subtree_skipped_when_root_readme_exists(tmp_path: Path):
    """If a root README is present, return it as a single record
    (don't walk subtrees)."""
    (tmp_path / "README.md").write_text("# root\n\nroot summary.\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text("{}")
    (tmp_path / "web" / "README.md").write_text("# web\n\nweb summary.\n")
    result = _detect(tmp_path)
    assert not isinstance(result, list)
    assert result.path == "README.md"
