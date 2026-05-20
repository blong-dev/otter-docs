"""Inline `# guid:` / `// guid:` marker tests — read + write paths.

The marker is the cross-tool primary key (otter-docs ↔ v3 Neo4j graph ↔
REFACTOR/Kanban diff-walker). Backwards compatibility with v3's
`gnosis/scripts/assign_guids.py` format is non-negotiable: the 3130
markers already in v3 source must keep validating with no churn.
"""

from __future__ import annotations

import re
from pathlib import Path

from otter_docs.guids import (
    GUID_MARKER_RE,
    assign_missing_markers,
    assign_missing_markers_in_file,
    assign_missing_markers_in_repo,
    find_marker_guid,
    new_marker_uuid,
    resolve_guid,
)
from otter_docs.models import Language
from otter_docs.parsers import parse_file
from otter_docs.parsers.typescript import TSX_PARSER

# A stable uuid we can assert on in fixtures.
GUID_A = "11111111-2222-3333-4444-555555555555"
GUID_B = "99999999-8888-7777-6666-555555555555"


# ── unit: regex + uuid helper ───────────────────────────────────────────


def test_guid_marker_regex_matches_both_syntaxes():
    assert GUID_MARKER_RE.match(f"# guid:{GUID_A}")
    assert GUID_MARKER_RE.match(f"// guid:{GUID_A}")
    assert GUID_MARKER_RE.match(f"    # guid:{GUID_A}")  # indented
    assert GUID_MARKER_RE.match(f"\t// guid:{GUID_A}")   # tab indent
    # Trailing whitespace tolerated.
    assert GUID_MARKER_RE.match(f"# guid:{GUID_A}   ")
    # Body content after guid is not.
    assert not GUID_MARKER_RE.match(f"# guid:{GUID_A} extra")
    # Wrong shape uuid.
    assert not GUID_MARKER_RE.match("# guid:not-a-uuid")


def test_new_marker_uuid_is_valid_uuid4_shape():
    u = new_marker_uuid()
    assert re.fullmatch(r"[0-9a-f\-]{36}", u)
    # Stable produces distinct values.
    assert u != new_marker_uuid()


# ── read-side: parsers honor markers ────────────────────────────────────


def test_python_marker_above_def_is_used_as_guid():
    src = (
        f"# guid:{GUID_A}\n"
        "def hello():\n"
        "    return 1\n"
    ).encode()
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.functions[0].guid == GUID_A


def test_python_marker_above_decorators_still_resolves():
    src = (
        f"# guid:{GUID_A}\n"
        "@property\n"
        "@cached\n"
        "def hello():\n"
        "    return 1\n"
    ).encode()
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.functions[0].guid == GUID_A


def test_python_marker_between_decorator_and_def_resolves():
    """v3 actually writes this layout: decorators above marker, marker
    immediately above def. Reader must handle it (it's the fast path)."""
    src = (
        "@property\n"
        f"# guid:{GUID_A}\n"
        "def hello():\n"
        "    return 1\n"
    ).encode()
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.functions[0].guid == GUID_A


def test_python_class_marker():
    src = (
        f"# guid:{GUID_A}\n"
        "class Foo:\n"
        "    pass\n"
    ).encode()
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.classes[0].guid == GUID_A


def test_python_no_marker_falls_back_to_derived():
    src = b"def hello():\n    return 1\n"
    r1 = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    r2 = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    # Derived guid stable + non-empty.
    assert r1.functions[0].guid == r2.functions[0].guid
    assert r1.functions[0].guid != GUID_A
    assert re.fullmatch(r"[0-9a-f]+", r1.functions[0].guid)


def test_python_method_marker_inside_class():
    src = (
        "class Foo:\n"
        f"    # guid:{GUID_A}\n"
        "    def bar(self):\n"
        "        return 1\n"
    ).encode()
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    bar = next(f for f in r.functions if f.name == "bar")
    assert bar.guid == GUID_A


def test_go_function_marker():
    src = (
        "package x\n"
        "\n"
        f"// guid:{GUID_A}\n"
        "func hello() int { return 1 }\n"
    ).encode()
    r = parse_file(repo="r", path="m.go", source=src, language=Language.GO)
    assert r.functions[0].guid == GUID_A


def test_typescript_function_marker():
    src = (
        f"// guid:{GUID_A}\n"
        "function hello() { return 1 }\n"
    ).encode()
    r = parse_file(
        repo="r", path="m.ts", source=src, language=Language.TYPESCRIPT
    )
    assert r.functions[0].guid == GUID_A


def test_typescript_arrow_anchors_on_lexical_declaration():
    """Marker placed above `const foo = …` resolves for the arrow."""
    src = (
        f"// guid:{GUID_A}\n"
        "const hello = (x) => x + 1;\n"
    ).encode()
    r = parse_file(
        repo="r", path="m.ts", source=src, language=Language.TYPESCRIPT
    )
    fn = next(f for f in r.functions if f.name == "hello")
    assert fn.guid == GUID_A


def test_resolve_guid_helper_directly():
    src = (
        f"# guid:{GUID_A}\n"
        "def f():\n    pass\n"
    ).encode()
    # def is on row 1 (0-indexed).
    assert find_marker_guid(src, 1) == GUID_A
    assert resolve_guid(src, 1, derived="DERIVED") == GUID_A
    # No marker → fallback.
    assert resolve_guid(b"def f():\n    pass\n", 0, derived="DERIVED") == "DERIVED"


# ── write-side: assigner ────────────────────────────────────────────────


def test_assign_python_inserts_marker_above_def():
    src = b"def hello():\n    return 1\n"
    new, n = assign_missing_markers(src, suffix=".py")
    assert n == 1
    out = new.decode()
    assert out.startswith("# guid:")
    # And the parser now reads that guid.
    r = parse_file(
        repo="r", path="m.py", source=new, language=Language.PYTHON
    )
    assert GUID_MARKER_RE.match(out.splitlines()[0])
    assert r.functions[0].guid == out.splitlines()[0].removeprefix("# guid:").strip()


def test_assign_python_is_idempotent():
    src = b"def hello():\n    return 1\n"
    new1, n1 = assign_missing_markers(src, suffix=".py")
    new2, n2 = assign_missing_markers(new1, suffix=".py")
    assert n1 == 1
    assert n2 == 0
    assert new2 == new1


def test_assign_python_preserves_v3_layout_with_decorator():
    """When decorators exist, marker should land BETWEEN decorators
    and def — matching v3's insertion (insert at def's lineno)."""
    src = (
        b"@property\n"
        b"def hello():\n"
        b"    return 1\n"
    )
    new, n = assign_missing_markers(src, suffix=".py")
    assert n == 1
    lines = new.decode().splitlines()
    assert lines[0] == "@property"
    assert GUID_MARKER_RE.match(lines[1])
    assert lines[2] == "def hello():"


def test_assign_python_preserves_method_indent():
    src = (
        b"class Foo:\n"
        b"    def bar(self):\n"
        b"        return 1\n"
    )
    new, n = assign_missing_markers(src, suffix=".py")
    # 2 markers: class + method.
    assert n == 2
    lines = new.decode().splitlines()
    # Top-level class marker, no indent.
    assert lines[0].startswith("# guid:")
    # Method marker, 4-space indent.
    method_marker = next(
        i for i, ln in enumerate(lines) if ln.startswith("    # guid:")
    )
    assert lines[method_marker + 1] == "    def bar(self):"


def test_assign_python_skips_already_marked():
    src = (
        f"# guid:{GUID_A}\n"
        "def keep():\n    return 1\n"
        "\n"
        "def add(): return 2\n"
    ).encode()
    new, n = assign_missing_markers(src, suffix=".py")
    assert n == 1  # only the second function needed one
    out = new.decode()
    # Original GUID_A line untouched.
    assert f"# guid:{GUID_A}" in out


def test_assign_go_function():
    src = (
        b"package x\n\n"
        b"func hello() int { return 1 }\n"
    )
    new, n = assign_missing_markers(src, suffix=".go")
    assert n == 1
    assert "// guid:" in new.decode()


def test_assign_typescript_arrow_marker_on_const_line():
    src = b"const hello = (x) => x + 1;\n"
    new, n = assign_missing_markers(src, suffix=".ts")
    assert n == 1
    lines = new.decode().splitlines()
    assert lines[0].startswith("// guid:")
    assert lines[1].startswith("const hello")


def test_assign_unknown_extension_is_noop():
    src = b"whatever\n"
    new, n = assign_missing_markers(src, suffix=".rs")
    assert n == 0
    assert new == src


def test_assign_round_trip_parser_uses_assigned_guid():
    """Assign then parse: the FunctionRecord.guid must be the just-inserted
    marker uuid — the cross-tool primary-key invariant."""
    src = b"def a(): return 1\ndef b(): return 2\n"
    new, _ = assign_missing_markers(src, suffix=".py")
    r = parse_file(
        repo="r", path="m.py", source=new, language=Language.PYTHON
    )
    # Extract markers from the source in order.
    markers = [
        m.group(1) for m in (
            GUID_MARKER_RE.match(line) for line in new.decode().splitlines()
        ) if m is not None
    ]
    assert len(markers) == 2
    # Each function's guid is one of the assigned markers.
    fn_guids = {fn.guid for fn in r.functions}
    assert set(markers) == fn_guids


def test_assign_in_file_writes_in_place(tmp_path: Path):
    p = tmp_path / "m.py"
    p.write_text("def hello(): return 1\n")
    count, changed = assign_missing_markers_in_file(p)
    assert count == 1
    assert changed is True
    assert "# guid:" in p.read_text()
    # Second run is a no-op.
    count2, changed2 = assign_missing_markers_in_file(p)
    assert count2 == 0 and changed2 is False


def test_assign_in_repo_walks(tmp_path: Path):
    (tmp_path / "a.py").write_text("def a(): return 1\n")
    (tmp_path / "b.py").write_text("def b(): return 2\n")
    (tmp_path / "README.md").write_text("not source\n")
    report = assign_missing_markers_in_repo(tmp_path)
    assert report.files_changed == 2
    assert report.markers_inserted == 2
    assert report.errors == []


# ── v3 compatibility ────────────────────────────────────────────────────


def test_v3_style_layout_no_churn():
    """An lda.py-style snippet (class marker, then method marker) must
    parse with both markers honored AND survive an assign pass with
    zero new insertions."""
    src = (
        f"# guid:{GUID_A}\n"
        "class LDAAdapter:\n"
        "    \"\"\"docstring\"\"\"\n"
        "\n"
        f"    # guid:{GUID_B}\n"
        "    def _backfill_overrides(self):\n"
        "        return 1\n"
    ).encode()
    # Read-side honors both.
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.classes[0].guid == GUID_A
    bf = next(f for f in r.functions if f.name == "_backfill_overrides")
    assert bf.guid == GUID_B
    # Write-side: no churn.
    new, n = assign_missing_markers(src, suffix=".py")
    assert n == 0
    assert new == src


def test_tsx_parser_directly_honors_marker():
    """TSX_PARSER instance (used by Repo.scan() for .tsx) honors markers."""
    src = (
        f"// guid:{GUID_A}\n"
        "function Hello() { return null; }\n"
    ).encode()
    r = TSX_PARSER.parse(repo="r", path="App.tsx", source=src)
    assert r.functions[0].guid == GUID_A


# ── CLI smoke ───────────────────────────────────────────────────────────


def test_cli_assign_guids_writes_and_reports(tmp_path: Path, capsys):
    from otter_docs.cli import main

    (tmp_path / "a.py").write_text("def a(): return 1\n")
    rc = main(["assign-guids", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "assigned 1 markers" in out
    assert "# guid:" in (tmp_path / "a.py").read_text()


def test_cli_assign_guids_check_flag_nonzero_when_inserting(
    tmp_path: Path, capsys,
):
    from otter_docs.cli import main

    (tmp_path / "a.py").write_text("def a(): return 1\n")
    rc = main(["assign-guids", "--check", str(tmp_path)])
    # Markers were missing → --check exits 1.
    assert rc == 1
    # Re-run: now they're present → exit 0.
    rc2 = main(["assign-guids", "--check", str(tmp_path)])
    assert rc2 == 0


def test_cli_assign_guids_accepts_multiple_files(tmp_path: Path, capsys):
    """Pre-commit hook path: pass individual file paths."""
    from otter_docs.cli import main

    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def a(): return 1\n")
    b.write_text("def b(): return 2\n")
    rc = main(["assign-guids", str(a), str(b)])
    assert rc == 0
    assert "# guid:" in a.read_text()
    assert "# guid:" in b.read_text()
