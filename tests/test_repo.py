"""Tests for the Repo skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend


def test_repo_resolves_root(tmp_path: Path):
    with Repo(tmp_path) as repo:
        assert repo.root == tmp_path.resolve()


def test_repo_default_name_is_basename(tmp_path: Path):
    nested = tmp_path / "myproject"
    nested.mkdir()
    with Repo(nested) as repo:
        assert repo.name == "myproject"


def test_repo_explicit_name(tmp_path: Path):
    with Repo(tmp_path, name="otherwise") as repo:
        assert repo.name == "otherwise"


def test_repo_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Repo(tmp_path / "does_not_exist")


def test_repo_creates_data_dir(tmp_path: Path):
    with Repo(tmp_path) as repo:
        assert (repo.root / ".otter-docs").exists()
        assert (repo.root / ".otter-docs").is_dir()


def test_repo_default_backend_is_sqlite(tmp_path: Path):
    with Repo(tmp_path) as repo:
        assert isinstance(repo.graph, SqliteBackend)


def test_repo_accepts_custom_backend(tmp_path: Path):
    custom = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, backend=custom) as repo:
        assert repo.graph is custom
    # closing the repo also closes the backend it owns


def test_repo_scan_empty_repo(tmp_path: Path):
    """scan() over an empty directory should return a ScanReport with zeros."""
    with Repo(tmp_path) as repo:
        report = repo.scan()
        assert report.files_seen == 0
        assert report.files_parsed == 0
        assert report.modules == 0


def test_repo_scan_parses_python_file(tmp_path: Path):
    (tmp_path / "hello.py").write_text(
        '"""mod docs"""\n'
        "import os\n\n"
        "def greet(name):\n"
        "    return 'hi ' + name\n"
    )
    with Repo(tmp_path) as repo:
        report = repo.scan()
        assert report.files_parsed == 1
        assert report.modules == 1
        assert report.functions == 1
        # IMPORTS edge: hello.py → os
        assert report.edges >= 1
        mod = repo.graph.get_module(repo.name, "hello.py")
        assert mod is not None
        assert mod.docstring == "mod docs"
        assert mod.imports == ["os"]


def test_repo_scan_skips_excluded_dirs(tmp_path: Path):
    """node_modules, .git, .venv, etc. should be pruned from the walk."""
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("def x(): pass\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("def y(): pass\n")
    (tmp_path / "real.py").write_text("def real(): pass\n")
    with Repo(tmp_path) as repo:
        report = repo.scan()
        assert report.files_parsed == 1
        # Only the real.py module should have been indexed.
        modules = [m.path for m in repo.graph.list_modules(repo.name)]
        assert modules == ["real.py"]


def test_repo_scan_reset_clears_stale(tmp_path: Path):
    (tmp_path / "a.py").write_text("def f(): pass\n")
    with Repo(tmp_path) as repo:
        repo.scan()
        assert len(list(repo.graph.list_modules(repo.name))) == 1
        # Delete the file, rescan with reset=True — the module should be gone.
        (tmp_path / "a.py").unlink()
        repo.scan(reset=True)
        assert list(repo.graph.list_modules(repo.name)) == []


def test_repo_findings_empty_returns_empty_list(tmp_path: Path):
    with Repo(tmp_path) as repo:
        assert repo.findings() == []


def test_repo_render_stub_raises(tmp_path: Path):
    with Repo(tmp_path) as repo:
        with pytest.raises(NotImplementedError, match="phase"):
            repo.render("system_overview")
