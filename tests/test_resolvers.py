"""Cross-file resolver tests.

Mostly Python today via jedi. Go and TS resolvers land in 2.4.5 and
will add their own per-resolver tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# jedi is an optional dep — skip cleanly if it isn't installed.
jedi = pytest.importorskip("jedi")

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.models import Language
from otter_docs.resolvers import registry


def _two_file_repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        "def helper():\n    return 1\n\n"
        "def unused():\n    return 99\n"
    )
    (tmp_path / "b.py").write_text(
        "from a import helper\n\n"
        "def caller():\n    return helper() + 1\n"
    )
    return tmp_path


def test_python_resolver_registered():
    assert Language.PYTHON in registry()


def test_resolve_emits_cross_file_call_edge(tmp_path: Path):
    _two_file_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        # Before resolve(): no cross-file edges.
        helper = next(f for f in repo.graph.list_functions(repo.name) if f.name == "helper")
        assert repo.graph.callers_of(repo.name, helper.guid) == []
        # After resolve(): helper has caller.
        reports = repo.resolve()
        assert reports[Language.PYTHON].edges_emitted >= 1
        callers = repo.graph.callers_of(repo.name, helper.guid)
        assert len(callers) == 1
        caller = next(f for f in repo.graph.list_functions(repo.name) if f.name == "caller")
        assert callers == [caller.guid]


def test_resolve_doesnt_resurrect_truly_unused(tmp_path: Path):
    """`unused` is never called — must stay flagged after resolve()."""
    _two_file_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        repo.resolve()
        dead = repo.findings(kinds={"dead_code"})
        names = {f.evidence["function_name"] for f in dead}
        assert "unused" in names
        # `helper` should NOT be flagged anymore (has a caller now).
        assert "helper" not in names


def test_resolve_is_idempotent(tmp_path: Path):
    _two_file_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        repo.resolve()
        # Count edges via callers_of for each function.
        first_pass: dict[str, list[str]] = {}
        for fn in repo.graph.list_functions(repo.name):
            first_pass[fn.name] = list(repo.graph.callers_of(repo.name, fn.guid))
        repo.resolve()
        for fn in repo.graph.list_functions(repo.name):
            assert (
                first_pass[fn.name]
                == list(repo.graph.callers_of(repo.name, fn.guid))
            )


def test_resolve_handles_no_python_in_repo(tmp_path: Path):
    """A non-Python repo just has nothing to resolve."""
    (tmp_path / "main.go").write_text(
        "package main\n\nfunc main() {}\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        reports = repo.resolve()
        # Python resolver ran but found no Python modules; emits 0 edges.
        assert reports[Language.PYTHON].edges_emitted == 0


def test_resolve_can_filter_by_language(tmp_path: Path):
    """languages={} disables every resolver."""
    _two_file_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        reports = repo.resolve(languages=set())  # explicit empty set
        assert reports == {}


def test_jedi_project_root_walks_through_init_py(tmp_path: Path):
    """If repo_root is a package, jedi project root must be its parent.

    Reproduces the gnosis-specific bug where a repo that's both a
    Python project (pyproject.toml) and a package (__init__.py) was
    feeding the wrong root to jedi, causing in-repo imports to fail.
    """
    from otter_docs.resolvers.python import _find_jedi_project_root

    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "pyproject.toml").write_text("[project]\nname='mypkg'\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("")

    # repo_root at the package top — walk up one to tmp_path.
    assert _find_jedi_project_root(pkg) == tmp_path
    # repo_root deeper inside a nested package — walk all the way up.
    assert _find_jedi_project_root(sub) == tmp_path
    # repo_root that isn't a package — stays put.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _find_jedi_project_root(plain) == plain


def test_resolve_method_call_finds_class_method(tmp_path: Path):
    """obj.method() across files should resolve to the class's method."""
    (tmp_path / "lib.py").write_text(
        "class Counter:\n"
        "    def increment(self):\n"
        "        return 1\n"
    )
    (tmp_path / "app.py").write_text(
        "from lib import Counter\n\n"
        "def use():\n"
        "    c = Counter()\n"
        "    return c.increment()\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        repo.resolve()
        # The method `increment` should have `use` as a caller.
        fns = {f.name: f for f in repo.graph.list_functions(repo.name)}
        # parser names the method as "increment" (PythonParser) — confirm via lookup
        method_name_candidates = [n for n in fns if "increment" in n]
        assert method_name_candidates, f"no `increment` in {list(fns)}"
        # At least one of these should now have a caller after resolve().
        any_with_caller = any(
            repo.graph.callers_of(repo.name, fns[n].guid)
            for n in method_name_candidates
        )
        assert any_with_caller


# ── loud-skip: missing resolver should produce a warning ────────────


def _install_pyfile(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")


def test_resolve_warns_when_seen_language_has_no_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If scan sees `.py` files but no Python resolver is registered,
    `resolve()` returns a synthesized report with a warning and the
    install hint — this is the failure mode we don't want silent."""
    _install_pyfile(tmp_path)
    # Force Python out of the registry for this test, regardless of
    # whether jedi happens to be installed in the test env.
    from otter_docs.resolvers import base as resolver_base
    monkeypatch.setitem(resolver_base._registry, Language.PYTHON, None)
    monkeypatch.delitem(resolver_base._registry, Language.PYTHON)
    monkeypatch.delenv("OTTER_RESOLVER_QUIET", raising=False)

    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        reports = repo.resolve()
        assert Language.PYTHON in reports
        warnings = reports[Language.PYTHON].warnings
        assert warnings, "expected a missing-resolver warning for python"
        # The warning carries the install hint so the user can act on it.
        assert "pip install otter-docs[python-resolver]" in warnings[0]
        # And it surfaces the env-var escape hatch.
        assert "OTTER_RESOLVER_QUIET=python" in warnings[0]


def test_resolve_silenced_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OTTER_RESOLVER_QUIET=<lang> drops the warning for that language."""
    _install_pyfile(tmp_path)
    from otter_docs.resolvers import base as resolver_base
    monkeypatch.delitem(resolver_base._registry, Language.PYTHON, raising=False)
    monkeypatch.setenv("OTTER_RESOLVER_QUIET", "python")

    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="rtest", backend=backend) as repo:
        repo.scan()
        reports = repo.resolve()
        # No entry at all for Python — it's neither registered nor warned.
        assert Language.PYTHON not in reports


def test_install_hint_declared_for_every_supported_language() -> None:
    """Each shipped resolver module declares its install hint at import
    time — even when its tooling is absent — so the missing-resolver
    warning can render an actionable command."""
    from otter_docs.resolvers.base import install_hint_for

    py = install_hint_for(Language.PYTHON)
    go = install_hint_for(Language.GO)
    ts = install_hint_for(Language.TYPESCRIPT)
    assert py and "pip install otter-docs[python-resolver]" in py
    assert go and "gopls" in go
    assert ts and "typescript-language-server" in ts
