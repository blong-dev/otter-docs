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


def test_repo_scan_stub_raises(tmp_path: Path):
    with Repo(tmp_path) as repo:
        with pytest.raises(NotImplementedError, match="phase"):
            repo.scan()


def test_repo_findings_stub_raises(tmp_path: Path):
    with Repo(tmp_path) as repo:
        with pytest.raises(NotImplementedError, match="phase"):
            repo.findings()


def test_repo_render_stub_raises(tmp_path: Path):
    with Repo(tmp_path) as repo:
        with pytest.raises(NotImplementedError, match="phase"):
            repo.render("system_overview")
