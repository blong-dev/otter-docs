"""TypeScript resolver tests.

Marked `integration` and skipped unless `typescript-language-server`
is on PATH AND `--run-integration` is passed. Most CI environments
won't have it; install with:

    npm install -g typescript typescript-language-server
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.models import Language
from otter_docs.resolvers import registry


pytestmark = pytest.mark.integration


def _tsserver_available() -> bool:
    return shutil.which("typescript-language-server") is not None


def _write_ts_repo(tmp_path: Path) -> None:
    (tmp_path / "tsconfig.json").write_text(
        '{"compilerOptions":{"target":"es2020","module":"esnext","moduleResolution":"node"},'
        '"include":["**/*.ts"]}'
    )
    (tmp_path / "a.ts").write_text(
        "export function helper(): number { return 1; }\n"
        "export function unused(): number { return 99; }\n"
    )
    (tmp_path / "b.ts").write_text(
        "import { helper } from './a';\n\n"
        "export function caller() { return helper() + 1; }\n"
    )


def test_typescript_resolver_registered_when_binary_present():
    if not _tsserver_available():
        pytest.skip("typescript-language-server not on PATH")
    assert Language.TYPESCRIPT in registry()


def test_typescript_resolver_emits_cross_file_call(tmp_path: Path):
    if not _tsserver_available():
        pytest.skip("typescript-language-server not on PATH")
    _write_ts_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="tstest", backend=backend) as repo:
        repo.scan()
        helper = next(
            f for f in repo.graph.list_functions(repo.name) if f.name == "helper"
        )
        # No callers before resolve.
        assert repo.graph.callers_of(repo.name, helper.guid) == []
        reports = repo.resolve()
        # At least one edge from the TS resolver.
        ts_report = reports.get(Language.TYPESCRIPT)
        assert ts_report is not None and ts_report.edges_emitted >= 1
        callers = repo.graph.callers_of(repo.name, helper.guid)
        assert len(callers) == 1
        caller_fn = next(
            f for f in repo.graph.list_functions(repo.name) if f.name == "caller"
        )
        assert callers == [caller_fn.guid]


def test_typescript_resolver_dead_code_drops_after_resolve(tmp_path: Path):
    if not _tsserver_available():
        pytest.skip("typescript-language-server not on PATH")
    _write_ts_repo(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="tstest", backend=backend) as repo:
        repo.scan()
        repo.resolve()
        dead = repo.findings(kinds={"dead_code"})
        names = {f.evidence["function_name"] for f in dead}
        # helper has a cross-file caller now.
        assert "helper" not in names
        # unused stays flagged.
        assert "unused" in names
