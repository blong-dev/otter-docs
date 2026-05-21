"""Tests for SourceLayoutDetector."""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.source_layout import SourceLayoutDetector


def _detect(tmp_path: Path):
    return SourceLayoutDetector().detect(repo="t", repo_root=tmp_path)


def test_annotates_known_dirs(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "README.md").write_text("# x\n")
    surface = _detect(tmp_path)
    assert surface is not None
    by_name = {e.name: e for e in surface.entries}
    assert by_name["src"].annotation == "source root"
    assert by_name["tests"].annotation == "tests"
    assert by_name["docs"].annotation == "documentation"
    assert by_name["scripts"].annotation == "utility scripts"
    assert by_name["README.md"].annotation is None
    assert by_name["src"].is_dir is True
    assert by_name["README.md"].is_dir is False


def test_filters_build_artifacts(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "target").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / "dist").mkdir()
    (tmp_path / "src").mkdir()
    surface = _detect(tmp_path)
    assert surface is not None
    names = {e.name for e in surface.entries}
    assert names == {"src"}


def test_filters_hidden_dirs_unless_known(tmp_path: Path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "workflows").mkdir()
    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".idea").mkdir()
    (tmp_path / "src").mkdir()
    surface = _detect(tmp_path)
    assert surface is not None
    names = {e.name for e in surface.entries}
    assert ".github" in names  # known dot-dir
    assert ".vscode" not in names  # excluded
    assert ".idea" not in names
    assert "src" in names


def test_files_listed_with_no_annotation_by_default(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "LICENSE").write_text("MIT License\n")
    (tmp_path / "Makefile").write_text("all:\n\techo ok\n")
    surface = _detect(tmp_path)
    assert surface is not None
    by_name = {e.name: e for e in surface.entries}
    for name in ("pyproject.toml", "LICENSE", "Makefile"):
        assert by_name[name].annotation is None
        assert by_name[name].is_dir is False


def test_empty_repo_returns_none(tmp_path: Path):
    assert _detect(tmp_path) is None


def test_monorepo_apps_packages(tmp_path: Path):
    (tmp_path / "apps").mkdir()
    (tmp_path / "packages").mkdir()
    (tmp_path / "services").mkdir()
    surface = _detect(tmp_path)
    assert surface is not None
    by_name = {e.name: e for e in surface.entries}
    assert by_name["apps"].annotation == "application code (monorepo)"
    assert by_name["packages"].annotation == "library packages (monorepo)"
    assert by_name["services"].annotation == "services (monorepo)"
