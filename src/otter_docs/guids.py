"""Inline GUID markers — the cross-tool primary key for code symbols.

A symbol's guid can live in source as a comment line directly above
its definition (above any decorators):

    Python:    `# guid:<uuid4>`
    Go/TS/JS:  `// guid:<uuid4>`

When present, parsers use the marker as the symbol's `guid`; otherwise
they fall back to a derived (path+name+line) hash. The marker is the
canonical identity: it survives renames/moves (the comment travels with
the symbol), and the REFACTOR/Kanban plan walks commit diffs for these
exact markers to attribute work to symbols.

Compatible with v3's `gnosis/scripts/assign_guids.py` (the originating
convention). The markers already in v3 source continue to validate
unchanged — that's a hard requirement: changing the format would churn
thousands of marker lines across the repo.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# One regex covers both comment syntaxes. Lines preceding a symbol are
# always either Python `#` comments or Go/TS/JS `//` comments — never
# the other language's syntax in the same file — so this is unambiguous.
GUID_MARKER_RE = re.compile(
    r"^\s*(?:#|//)\s*guid:([0-9a-f\-]{36})\s*$"
)

# Prefixes that mean "keep walking backwards looking for the marker":
# decorators (`@…`), comments (`#…`, `//…`). Blank lines are also
# skipped. Anything else stops the walk. Matches v3's algorithm exactly.
_SKIPPABLE_PREFIXES = ("@", "#", "//")


def _find_marker_in_str_lines(
    lines: list[str], start_row_0idx: int
) -> str | None:
    """Same logic as find_marker_guid but on pre-split string lines.

    Used by the writer (where we work in str throughout) and shared
    with the bytes-flavored entry point below.
    """
    i = start_row_0idx - 1
    while i >= 0:
        if i >= len(lines):
            i -= 1
            continue
        line = lines[i]
        m = GUID_MARKER_RE.match(line)
        if m is not None:
            return m.group(1)
        stripped = line.strip()
        if not stripped:
            i -= 1
            continue
        if stripped.startswith(_SKIPPABLE_PREFIXES):
            i -= 1
            continue
        break
    return None


def find_marker_guid(source: bytes, start_row_0idx: int) -> str | None:
    """Walk backwards from above row `start_row_0idx` looking for a guid
    marker, past decorators / comment lines / blanks. Returns the uuid
    if found, else None.

    `start_row_0idx` is the 0-indexed row of the symbol's def line
    (matching tree-sitter's `node.start_point.row`).
    """
    lines = [
        ln.decode("utf-8", errors="replace") for ln in source.splitlines()
    ]
    return _find_marker_in_str_lines(lines, start_row_0idx)


def resolve_guid(
    source: bytes,
    start_row_0idx: int,
    derived: str,
) -> str:
    """Marker if present, otherwise the derived fallback.

    This is the single seam between "stable cross-tool identity"
    (marker) and "best-effort identity for unmarked code" (derived).
    """
    marker = find_marker_guid(source, start_row_0idx)
    return marker if marker is not None else derived


def new_marker_uuid() -> str:
    """Mint a fresh marker uuid in the v3-compatible format."""
    return str(uuid.uuid4())


# ── write-side: assign markers in source ──────────────────────────────


@dataclass
class AssignReport:
    """Result of an assign pass."""

    files_scanned: int = 0
    files_changed: int = 0
    markers_inserted: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _python_anchors(source: bytes) -> list[int]:
    """0-indexed def lines for every function/class (incl. nested, methods).

    Matches v3's `assign_guids.py` symbol set exactly: FunctionDef +
    AsyncFunctionDef + ClassDef. No module-level marker.
    """
    from otter_docs.parsers import python as _py
    tree = _py._PARSER.parse(source)
    out: list[int] = []

    def walk(node) -> None:
        if node.type in ("function_definition", "class_definition"):
            out.append(node.start_point.row)
        for c in node.named_children:
            walk(c)

    walk(tree.root_node)
    return out


def _go_anchors(source: bytes) -> list[int]:
    """0-indexed def lines for Go: functions, methods, top-level types."""
    from otter_docs.parsers import go as _go
    tree = _go._PARSER.parse(source)
    out: list[int] = []
    for child in tree.root_node.named_children:
        if child.type in (
            "function_declaration",
            "method_declaration",
            "type_declaration",
        ):
            out.append(child.start_point.row)
    return out


def _ts_anchors(source: bytes, *, is_tsx: bool = False) -> list[int]:
    """0-indexed def lines for TS/JS: function decls, class decls, class
    methods, and any `const/let/var x = (…) => {…}` / `function () {…}`
    declarations. Arrows anchor on the outer `lexical_declaration` line
    so the marker sits where a developer would naturally write it."""
    from otter_docs.parsers import typescript as _ts
    parser = _ts._PARSER_TSX if is_tsx else _ts._PARSER_TS
    tree = parser.parse(source)
    out: list[int] = []

    def walk(node) -> None:
        t = node.type
        if t == "function_declaration":
            out.append(node.start_point.row)
            return
        if t == "class_declaration":
            out.append(node.start_point.row)
            body = node.child_by_field_name("body")
            if body is not None:
                for member in body.named_children:
                    if member.type == "method_definition":
                        out.append(member.start_point.row)
            return
        if t in ("lexical_declaration", "variable_declaration"):
            # Only emit if at least one declarator is an arrow/function
            # expression. One marker per decl statement (line-level).
            has_fn = False
            for decl in node.named_children:
                if decl.type != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                if value is not None and value.type in (
                    "arrow_function", "function_expression"
                ):
                    has_fn = True
                    break
            if has_fn:
                out.append(node.start_point.row)
            return
        for c in node.named_children:
            walk(c)

    walk(tree.root_node)
    return out


def _anchors_for(source: bytes, suffix: str) -> tuple[list[int], str]:
    """Return (anchor_rows, comment_prefix) for a file by extension.

    suffix is the dotted file extension (`.py`, `.go`, `.ts`, `.tsx`,
    `.js`, `.jsx`). Unknown extensions return ([], "") and the caller
    treats them as "nothing to assign."
    """
    s = suffix.lower()
    if s == ".py":
        return _python_anchors(source), "#"
    if s == ".go":
        return _go_anchors(source), "//"
    if s == ".tsx":
        return _ts_anchors(source, is_tsx=True), "//"
    if s in (".ts", ".js", ".jsx"):
        return _ts_anchors(source), "//"
    return [], ""


def assign_missing_markers(
    source: bytes, *, suffix: str,
) -> tuple[bytes, int]:
    """Insert a marker comment above each symbol that doesn't already
    have one. Returns (new_source, count_inserted). Idempotent: a
    second call on the output inserts zero markers."""
    anchors, comment = _anchors_for(source, suffix)
    if not anchors or not comment:
        return source, 0
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    # Insert bottom-up so earlier (smaller-row) anchors don't shift.
    inserted = 0
    for row in sorted(set(anchors), reverse=True):
        if row < 0 or row >= len(lines):
            continue
        # Marker-check against current line state (rows above `row`
        # are untouched because we process bottom-up).
        bare = [ln.rstrip("\r\n") for ln in lines]
        if _find_marker_in_str_lines(bare, row) is not None:
            continue
        def_line = lines[row]
        indent_len = len(def_line) - len(def_line.lstrip(" \t"))
        indent = def_line[:indent_len]
        line_end = "\r\n" if def_line.endswith("\r\n") else "\n"
        marker_line = (
            f"{indent}{comment} guid:{new_marker_uuid()}{line_end}"
        )
        lines.insert(row, marker_line)
        inserted += 1
    if inserted == 0:
        return source, 0
    return ("".join(lines)).encode("utf-8"), inserted


def assign_missing_markers_in_file(path: Path) -> tuple[int, bool]:
    """Apply `assign_missing_markers` to a single file in place.

    Returns (count_inserted, file_changed). Files with unknown
    extensions or no defs are no-ops. Never modifies a file that
    already has markers for every def (idempotent).
    """
    try:
        source = path.read_bytes()
    except OSError:
        return 0, False
    new_source, count = assign_missing_markers(source, suffix=path.suffix)
    if count == 0 or new_source == source:
        return 0, False
    path.write_bytes(new_source)
    return count, True


def assign_missing_markers_in_repo(root: Path) -> AssignReport:
    """Walk every supported source file under `root` and assign missing
    markers. Uses the same iter_source_files discovery as scan() so the
    write surface matches what otter-docs reads."""
    from otter_docs.discovery import iter_source_files
    report = AssignReport()
    for abs_path, _lang in iter_source_files(root):
        report.files_scanned += 1
        try:
            count, changed = assign_missing_markers_in_file(abs_path)
        except Exception as e:
            report.errors.append(
                (str(abs_path), f"{type(e).__name__}: {e}")
            )
            continue
        if changed:
            report.files_changed += 1
            report.markers_inserted += count
    return report
