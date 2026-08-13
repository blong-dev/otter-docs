"""OD-2 — test code is excluded from production-focused renders by default.

system_overview's "largest modules" and architecture_smells' functions/hubs
should not be dominated by test fixtures (which naturally carry huge function
counts and high fan-in). Opt back in with OTTER_INCLUDE_TESTS=1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.discovery import is_test_path


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_app.py",
        "gnosis/tests/conftest.py",
        "pkg/foo_test.go",
        "web/src/App.test.tsx",
        "web/src/util.spec.ts",
        "src/__tests__/thing.js",
        "test/helpers.py",
    ],
)
def test_is_test_path_true(path: str):
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "gnosis/shared/permissions.py",  # 'shared', not 'tests'
        "docs/specs/plan.md",  # specs are design docs, NOT tests
        "web/src/App.tsx",
        "pkg/server.go",
        "src/latest.py",  # 'test' substring but not a test file
        "contest.py",  # not conftest
    ],
)
def test_is_test_path_false(path: str):
    assert is_test_path(path) is False


def _repo_with_test_and_prod(tmp_path: Path) -> Repo:
    (tmp_path / "app.py").write_text(
        "def a():\n    return 1\n\n"
        "def b():\n    return 2\n\n"
        "def c():\n    return 3\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "def test_a():\n    assert 1\n\n"
        "def test_b():\n    assert 2\n\n"
        "def test_c():\n    assert 3\n\n"
        "def test_d():\n    assert 4\n"
    )
    r = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    r.scan()
    return r


def test_system_overview_excludes_test_modules_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OTTER_INCLUDE_TESTS", raising=False)
    r = _repo_with_test_and_prod(tmp_path)
    out = r.render("system_overview")
    assert "Largest production modules" in out
    assert "`app.py`" in out
    assert "tests/test_app.py" not in out
    assert "test module(s) excluded" in out
    r.close()


def test_system_overview_includes_tests_with_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OTTER_INCLUDE_TESTS", "1")
    r = _repo_with_test_and_prod(tmp_path)
    out = r.render("system_overview")
    assert "tests/test_app.py" in out
    assert "Largest modules by function count" in out  # non-"production" label
    r.close()
