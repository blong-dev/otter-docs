"""Tests for the scan() per-file orphan purge (spec: ts-parser-incomplete-slices.md).

Validates the fix for the silent recall gap where a function moving
between scans created a new hash-anchored guid without removing the
old row. Re-scanning a file should leave exactly the records the
parser emitted on that scan — no ghosts.
"""

from __future__ import annotations

from pathlib import Path

from otter_docs import Repo
from otter_docs.backends import SqliteBackend


def test_rescan_purges_stale_function_record_when_function_moves(tmp_path: Path):
    """Move a function down a few lines between scans; the old hash
    guid would otherwise stay around. Verify only one record exists
    for the (renamed-by-line) function after re-scan."""
    src = tmp_path / "core.py"
    src.write_text(
        "def alpha(x):\n"
        "    return x + 1\n"
        "\n"
        "def beta(x):\n"
        "    return x * 2\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="t", backend=backend) as r:
        r.scan()
        # Add some imports above `beta` so it moves down by 3 lines.
        src.write_text(
            "import math\n"
            "import json\n"
            "import os\n"
            "\n"
            "def alpha(x):\n"
            "    return x + 1\n"
            "\n"
            "def beta(x):\n"
            "    return x * 2\n"
        )
        r.scan()
        beta_records = [
            fn for fn in r.graph.list_functions(r.name) if fn.name == "beta"
        ]
        assert len(beta_records) == 1, (
            "expected exactly 1 beta record after re-scan; "
            f"got {len(beta_records)} → stale records present"
        )
        # New record's line reflects the post-move position.
        assert beta_records[0].line == 8


def test_rescan_purges_renamed_function(tmp_path: Path):
    """A function rename creates a new guid (different name → different
    hash). The old guid should disappear, not coexist."""
    src = tmp_path / "core.py"
    src.write_text("def helper(x):\n    return x\n")
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="t", backend=backend) as r:
        r.scan()
        assert len([f for f in r.graph.list_functions(r.name)]) == 1
        # Rename the function.
        src.write_text("def utility(x):\n    return x\n")
        r.scan()
        all_funcs = list(r.graph.list_functions(r.name))
        names = {f.name for f in all_funcs}
        assert names == {"utility"}, (
            f"expected only `utility` after rename; got {names}"
        )


def test_rescan_purges_deleted_function(tmp_path: Path):
    """A function removed from source should leave the graph."""
    src = tmp_path / "core.py"
    src.write_text(
        "def keep(): return 1\n\n"
        "def drop(): return 2\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="t", backend=backend) as r:
        r.scan()
        assert {f.name for f in r.graph.list_functions(r.name)} == {"keep", "drop"}
        src.write_text("def keep(): return 1\n")
        r.scan()
        assert {f.name for f in r.graph.list_functions(r.name)} == {"keep"}


def test_rescan_purges_orphan_class(tmp_path: Path):
    """Class records purge the same way function records do."""
    src = tmp_path / "core.py"
    src.write_text("class Foo:\n    pass\n")
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="t", backend=backend) as r:
        r.scan()
        assert {c.name for c in r.graph.list_classes(r.name)} == {"Foo"}
        src.write_text("class Bar:\n    pass\n")
        r.scan()
        assert {c.name for c in r.graph.list_classes(r.name)} == {"Bar"}


def test_unchanged_file_no_purge_churn(tmp_path: Path):
    """Re-scanning an unchanged file should leave guids stable
    (idempotent — no purge required)."""
    src = tmp_path / "core.py"
    src.write_text("def f(): return 1\n")
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="t", backend=backend) as r:
        r.scan()
        before = {f.guid for f in r.graph.list_functions(r.name)}
        r.scan()
        after = {f.guid for f in r.graph.list_functions(r.name)}
        assert before == after


def test_rescan_preserves_other_files(tmp_path: Path):
    """Purging file A's orphans must NOT touch records from file B."""
    (tmp_path / "a.py").write_text("def in_a(): return 1\n")
    (tmp_path / "b.py").write_text("def in_b(): return 2\n")
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="t", backend=backend) as r:
        r.scan()
        assert {f.name for f in r.graph.list_functions(r.name)} == {"in_a", "in_b"}
        # Modify only a.py.
        (tmp_path / "a.py").write_text("def in_a_renamed(): return 1\n")
        r.scan()
        assert {f.name for f in r.graph.list_functions(r.name)} == {
            "in_a_renamed", "in_b",
        }
