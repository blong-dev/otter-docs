"""Phase 10 — CLI + hook installer tests.

Drive the CLI via main(argv) (not subprocess) so failures surface
with full tracebacks. The CLI opens a Repo with the default
SqliteBackend at <path>/.otter-docs/graph.db, so tests use tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from otter_docs.cli import main
from otter_docs.hooks import install_hooks


def _py_repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text(
        "import os\n\n"
        "def used():\n    return 1\n\n"
        "def caller():\n    return used()\n\n"
        "def orphan():\n    return 99\n"
    )
    return tmp_path


# ── scan ────────────────────────────────────────────────────────────────


def test_cli_scan(tmp_path: Path, capsys):
    _py_repo(tmp_path)
    rc = main(["scan", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scan:" in out
    assert "functions" in out


def test_cli_scan_no_resolve(tmp_path: Path, capsys):
    _py_repo(tmp_path)
    rc = main(["scan", str(tmp_path), "--no-resolve"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolve[" not in out


# ── find ────────────────────────────────────────────────────────────────


def test_cli_find_text(tmp_path: Path, capsys):
    _py_repo(tmp_path)
    rc = main(["find", str(tmp_path), "--kind", "dead_code", "--no-resolve"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dead_code" in out
    assert "orphan" in out


def test_cli_find_json(tmp_path: Path, capsys):
    _py_repo(tmp_path)
    rc = main(["find", str(tmp_path), "--kind", "dead_code", "--json", "--no-resolve"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(item["kind"] == "dead_code" for item in payload)


# ── render / init ───────────────────────────────────────────────────────


def test_cli_render_section(tmp_path: Path, capsys):
    _py_repo(tmp_path)
    rc = main(["render", str(tmp_path), "--section", "system_overview", "--no-resolve"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "modules" in out


def test_cli_render_document(tmp_path: Path):
    _py_repo(tmp_path)
    rc = main(["render", str(tmp_path), "--no-resolve", "--out", "DOC.md"])
    assert rc == 0
    doc = tmp_path / "DOC.md"
    assert doc.exists()
    assert "BEGIN GENERATED:system_overview" in doc.read_text()


def test_cli_init_bootstraps(tmp_path: Path):
    _py_repo(tmp_path)
    rc = main(["init", str(tmp_path), "--out", "SYSTEM.md"])
    assert rc == 0
    sysmd = tmp_path / "SYSTEM.md"
    assert sysmd.exists()
    text = sysmd.read_text()
    assert "BEGIN GENERATED:findings_summary" in text


def test_cli_init_preserves_human_prose_on_rerun(tmp_path: Path):
    _py_repo(tmp_path)
    main(["init", str(tmp_path)])
    sysmd = tmp_path / "SYSTEM.md"
    edited = sysmd.read_text() + "\n\nHUMAN: keep this.\n"
    sysmd.write_text(edited)
    main(["render", str(tmp_path), "--no-resolve"])
    assert "HUMAN: keep this." in sysmd.read_text()


# ── install-hooks ───────────────────────────────────────────────────────


def test_install_hooks_writes_executable_scripts(tmp_path: Path):
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    written = install_hooks(tmp_path)
    assert {p.name for p in written} == {"pre-commit", "pre-push"}
    for p in written:
        assert p.exists()
        # Executable bit set.
        assert p.stat().st_mode & 0o111
        body = p.read_text()
        assert "otter-docs" in body
        # Never amends history or blanket-stages.
        assert "--amend" not in body
        assert "git add -A" not in body


def test_install_hooks_no_git_dir_returns_empty(tmp_path: Path):
    assert install_hooks(tmp_path) == []


def test_cli_install_hooks_reports_missing_git(tmp_path: Path, capsys):
    rc = main(["install-hooks", str(tmp_path)])
    assert rc == 1
    assert "no .git" in capsys.readouterr().err.lower()


def test_cli_install_hooks_success(tmp_path: Path, capsys):
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    _py_repo(tmp_path)
    rc = main(["install-hooks", str(tmp_path)])
    assert rc == 0
    assert "Installed" in capsys.readouterr().out


# ── serve (no mcp extra) ────────────────────────────────────────────────


def test_cli_serve_without_mcp_extra_errors_cleanly(tmp_path: Path, capsys, monkeypatch):
    # Simulate `mcp` not being importable.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "otter_docs.mcp" or name.startswith("mcp"):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    rc = main(["serve", str(tmp_path)])
    assert rc == 1
    assert "mcp" in capsys.readouterr().err.lower()
