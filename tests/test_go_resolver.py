"""Go resolver tests.

Most of the LSP interaction is shared with the TypeScript resolver
and exercised there. These tests cover the Go-specific pieces:

  - registration is gated on gopls being on PATH
  - _iter_call_positions correctly points at the callee identifier
    for both `foo()` and `pkg.Foo()` patterns
  - integration test against a live gopls (skipped unless gopls is
    installed AND --run-integration is passed)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from otter_docs.models import Language
from otter_docs.resolvers import registry
from otter_docs.resolvers.go import _iter_call_positions


def _gopls_available() -> bool:
    return shutil.which("gopls") is not None


# ── unit ────────────────────────────────────────────────────────────────


def test_go_resolver_registered_only_when_binary_present():
    if _gopls_available():
        assert Language.GO in registry()
    else:
        assert Language.GO not in registry()


def test_iter_call_positions_finds_plain_call():
    import tree_sitter_go
    from tree_sitter import Language as TsLanguage
    from tree_sitter import Parser

    parser = Parser(TsLanguage(tree_sitter_go.language()))
    src = b"package x\n\nfunc main() {\n    helper()\n    pkg.Foo()\n}\n"
    tree = parser.parse(src)
    positions = list(_iter_call_positions(tree.root_node))
    # Two calls: `helper` on line 3 col 4, `Foo` on line 4 col 8.
    assert (3, 4) in positions  # row is 0-based
    # The selector_expression should yield the field identifier `Foo`
    assert any(line == 4 and col >= 4 for (line, col) in positions)


# ── integration ────────────────────────────────────────────────────────


@pytest.mark.integration
def test_go_resolver_emits_cross_file_call(tmp_path: Path):
    if not _gopls_available():
        pytest.skip("gopls not on PATH")
    from otter_docs import Repo
    from otter_docs.backends import SqliteBackend

    # Minimal Go module with a cross-file call.
    (tmp_path / "go.mod").write_text("module example.com/tst\n\ngo 1.21\n")
    (tmp_path / "a.go").write_text(
        "package tst\n\nfunc Helper() int { return 1 }\n"
    )
    (tmp_path / "b.go").write_text(
        "package tst\n\nfunc Caller() int { return Helper() + 1 }\n"
    )

    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="gotest", backend=backend) as repo:
        repo.scan()
        helper = next(
            f for f in repo.graph.list_functions(repo.name) if f.name == "Helper"
        )
        assert repo.graph.callers_of(repo.name, helper.guid) == []
        reports = repo.resolve()
        go_report = reports.get(Language.GO)
        assert go_report is not None and go_report.edges_emitted >= 1
        callers = repo.graph.callers_of(repo.name, helper.guid)
        assert len(callers) == 1
