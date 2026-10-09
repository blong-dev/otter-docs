"""`otter_docs.delta` — the per-commit symbol witness.

A throwaway git repo with a handful of commits exercises every change
kind and the two identity rules (marker = stable; unmarked = matched by
name, line shifts ignored). The CLI is driven through main(argv).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from otter_docs.cli import main
from otter_docs.delta import EMPTY_TREE, commit_delta, symbol_delta

GUID_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
GUID_K = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(root: Path, msg: str) -> str:
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(
        f"# guid:{GUID_A}\n"
        "def marked():\n    return 1\n\n"
        "def plain():\n    return 2\n\n"
        f"# guid:{GUID_K}\n"
        "class Keeper:\n    def method(self):\n        return 3\n"
    )
    (tmp_path / "notes.md").write_text("docs only\n")
    _commit(tmp_path, "c1")
    return tmp_path


def _by_name(delta) -> dict[str, object]:
    return {c.name: c for c in delta.changes}


# ── root commit: everything is added ────────────────────────────────────


def test_root_commit_is_all_added(repo: Path):
    d = commit_delta(repo, "HEAD")
    assert d.base == EMPTY_TREE
    names = _by_name(d)
    assert names["marked"].change == "added" and names["marked"].marked
    assert names["marked"].guid == GUID_A
    assert names["plain"].change == "added" and not names["plain"].marked
    assert names["Keeper"].change == "added" and names["Keeper"].kind == "class"
    assert "notes.md" in d.files and "notes.md" not in d.source_files
    assert d.counts()["added"] == 4  # marked, plain, Keeper, Keeper.method


# ── modified / added / removed, with a line shift on an untouched symbol ──


def test_modify_add_remove_ignores_line_shift(repo: Path):
    (repo / "a.py").write_text(
        "import os\n\n"  # shifts every line below by two
        f"# guid:{GUID_A}\n"
        "def marked():\n    return 100\n\n"  # body changed
        "def plain():\n    return 2\n\n"  # untouched, only shifted
        "def fresh():\n    return 4\n"  # new, unmarked
        # Keeper removed
    )
    sha = _commit(repo, "c2")
    d = commit_delta(repo, sha)
    names = _by_name(d)
    assert names["marked"].change == "modified" and names["marked"].marked
    assert names["fresh"].change == "added" and not names["fresh"].marked
    assert names["Keeper"].change == "removed" and names["Keeper"].guid == GUID_K
    assert names["method"].change == "removed"
    assert "plain" not in names, "a line shift on an unmarked symbol is not a change"
    # unmarked = fresh (added) + Keeper.method (removed)
    assert d.counts() == {"added": 1, "modified": 1, "removed": 2, "moved": 0, "unmarked": 2}


# ── moved: same marker, different file ──────────────────────────────────


def test_marked_symbol_moved_across_files(repo: Path):
    (repo / "a.py").write_text("def plain():\n    return 2\n")
    (repo / "b.py").write_text(
        f"# guid:{GUID_A}\n"
        "def marked():\n    return 1\n\n"
        f"# guid:{GUID_K}\n"
        "class Keeper:\n    def method(self):\n        return 3\n"
    )
    sha = _commit(repo, "c2")
    d = commit_delta(repo, sha)
    names = _by_name(d)
    assert names["marked"].change == "moved"
    assert names["marked"].path == "b.py" and names["marked"].from_path == "a.py"
    assert names["Keeper"].change == "moved"
    # Keeper.method is unmarked and changed file: name key differs → removed + added.
    kinds = sorted((c.name, c.change) for c in d.changes if c.name == "method")
    assert kinds == [("method", "added"), ("method", "removed")]


def test_moved_and_modified_reports_modified_with_from_path(repo: Path):
    (repo / "a.py").write_text("def plain():\n    return 2\n")
    (repo / "b.py").write_text(
        f"# guid:{GUID_A}\n"
        "def marked():\n    return 999\n\n"
        f"# guid:{GUID_K}\n"
        "class Keeper:\n    def method(self):\n        return 3\n"
    )
    sha = _commit(repo, "c2")
    names = _by_name(commit_delta(repo, sha))
    assert names["marked"].change == "modified"
    assert names["marked"].from_path == "a.py"


# ── unmarked modified: by derived guid when its line held, by name when it shifted


def test_unmarked_modified_in_place_keeps_derived_guid(repo: Path):
    (repo / "a.py").write_text(
        f"# guid:{GUID_A}\n"
        "def marked():\n    return 1\n\n"
        "def plain():\n    return 22\n\n"
        f"# guid:{GUID_K}\n"
        "class Keeper:\n    def method(self):\n        return 3\n"
    )
    sha = _commit(repo, "c2")
    names = _by_name(commit_delta(repo, sha))
    assert set(names) == {"plain"}
    assert names["plain"].change == "modified" and not names["plain"].marked
    assert names["plain"].from_guid is None  # same line → same derived guid


def test_unmarked_modified_after_shift_is_matched_by_name(repo: Path):
    (repo / "a.py").write_text(
        "import os\n\n"
        f"# guid:{GUID_A}\n"
        "def marked():\n    return 1\n\n"
        "def plain():\n    return 22\n\n"
        f"# guid:{GUID_K}\n"
        "class Keeper:\n    def method(self):\n        return 3\n"
    )
    sha = _commit(repo, "c2")
    names = _by_name(commit_delta(repo, sha))
    assert set(names) == {"plain"}
    assert names["plain"].change == "modified" and not names["plain"].marked
    assert names["plain"].from_guid and names["plain"].from_guid != names["plain"].guid


# ── ranges, resolution, determinism ─────────────────────────────────────


def test_symbol_delta_range_and_resolved_shas(repo: Path):
    c1 = _git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text(
        f"# guid:{GUID_A}\ndef marked():\n    return 5\n\n"
        "def plain():\n    return 2\n\n"
        f"# guid:{GUID_K}\nclass Keeper:\n    def method(self):\n        return 3\n"
    )
    c2 = _commit(repo, "c2")
    (repo / "a.py").write_text(
        f"# guid:{GUID_A}\ndef marked():\n    return 6\n\n"
        "def plain():\n    return 2\n\n"
        f"# guid:{GUID_K}\nclass Keeper:\n    def method(self):\n        return 3\n"
    )
    c3 = _commit(repo, "c3")
    d = symbol_delta(repo, base=c1, head="HEAD", repo="named")
    assert (d.base, d.head, d.repo) == (c1, c3, "named")
    assert [c.name for c in d.changes] == ["marked"]
    again = symbol_delta(repo, base=c1, head=c3, repo="named")
    assert again.to_dict() == d.to_dict()
    # Two commits in the range, one symbol touched: exactly one change row.
    assert c2 != c3


def test_docs_only_commit_has_no_symbol_changes(repo: Path):
    (repo / "notes.md").write_text("changed docs\n")
    sha = _commit(repo, "docs")
    d = commit_delta(repo, sha)
    assert d.files == ["notes.md"] and d.source_files == [] and d.changes == []


# ── CLI ─────────────────────────────────────────────────────────────────


def test_cli_delta_text_and_json(repo: Path, capsys):
    (repo / "a.py").write_text(
        f"# guid:{GUID_A}\ndef marked():\n    return 7\n\n"
        "def plain():\n    return 2\n\n"
        f"# guid:{GUID_K}\nclass Keeper:\n    def method(self):\n        return 3\n"
    )
    _commit(repo, "c2")
    assert main(["delta", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "~1" in out and "marked" in out and GUID_A in out

    assert main(["delta", str(repo), "--commit", "HEAD", "--json", "--repo", "x"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo"] == "x"
    assert payload["counts"]["modified"] == 1
    assert payload["changes"][0]["guid"] == GUID_A


def test_cli_delta_bad_revision_is_a_clean_error(repo: Path, capsys):
    assert main(["delta", str(repo), "--from", "nope"]) == 2
    assert "git failed" in capsys.readouterr().err
