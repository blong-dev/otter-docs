"""Symbol-level delta between two git revisions — the per-commit witness.

`scan()` records what a repo *is*. `symbol_delta()` records what one
commit *did* to it, at the symbol level, keyed by guid: which
functions and classes were added, modified, removed or moved between
two revisions. It parses the two blobs of every changed source file
straight out of git (no checkout, no working tree, no graph.db), so it
is cheap enough to run from a post-commit hook and deterministic
enough to re-run years later and get the same answer.

The delta is the join between code and intent. otter-docs knows what
changed; a host that tracks *why* work happened (a card, a ticket, a
prompt) attaches that to the commit, and the two together say, for any
symbol, which piece of intent last touched it. otter-docs never learns
what the host's intent objects are — it only emits the delta.

Identity rules:

  - A symbol carrying an inline `# guid:` / `// guid:` marker has a
    stable identity: the same guid on both sides is the same symbol,
    wherever it now lives (a path change is reported as `moved`).
  - An unmarked symbol gets the parser's derived guid, which includes
    its line number. Line shifts would make every unmarked symbol look
    removed-and-added, so unmarked symbols are matched by
    (path, kind, name) when that is unambiguous on both sides. Such a
    match is reported `modified` only if its body changed. Ambiguous
    unmarked symbols (two same-named defs in one file) fall back to
    the derived guid and may be reported as removed + added.

`marked` on every change says which rule applied, so a consumer can
weight the two kinds of identity differently.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from otter_docs.discovery import is_tsx, language_for_path
from otter_docs.models import Language
from otter_docs.parsers import parse_file

# The empty tree: diffing against it makes a root commit's delta "everything added".
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

_MARKER_GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

CHANGE_KINDS = ("added", "modified", "removed", "moved")


def _tool_version() -> str | None:
    """otter-docs' own version, recorded with every delta so a consumer can
    tell which parser produced a row years later."""
    try:
        from importlib.metadata import version
        return version("otter-docs")
    except Exception:
        return None


@dataclass
class SymbolChange:
    """One symbol's fate across the delta."""

    guid: str
    kind: str  # "function" | "class"
    name: str
    path: str  # path on the `to` side (on the `from` side for `removed`)
    change: str  # one of CHANGE_KINDS
    marked: bool  # identity came from an inline marker (stable) vs derived
    line: int | None = None
    end_line: int | None = None
    from_path: str | None = None  # set when the symbol's path differs across the delta
    from_guid: str | None = None  # set when an unmarked symbol was matched by name


@dataclass
class SymbolDelta:
    """Everything one revision range did to the repo's symbols."""

    repo: str
    base: str  # resolved full sha (or EMPTY_TREE)
    head: str  # resolved full sha
    files: list[str] = field(default_factory=list)  # every changed path, any type
    source_files: list[str] = field(default_factory=list)  # the parsed subset
    changes: list[SymbolChange] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["errors"] = [list(e) for e in self.errors]
        d["counts"] = self.counts()
        d["tool_version"] = _tool_version()
        return d

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in CHANGE_KINDS}
        for c in self.changes:
            out[c.change] += 1
        out["unmarked"] = sum(1 for c in self.changes if not c.marked)
        return out


@dataclass
class _Symbol:
    guid: str
    kind: str
    name: str
    path: str
    line: int
    end_line: int
    body_hash: str
    marked: bool


# ── git plumbing ────────────────────────────────────────────────────────


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True,
    ).stdout


def resolve_rev(root: str | Path, rev: str) -> str:
    """Full sha for `rev` (any git revision expression)."""
    return _git(Path(root), "rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()


def _blob(root: Path, rev: str, path: str) -> bytes | None:
    try:
        return _git(root, "show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return None


def _changed_paths(root: Path, base: str, head: str) -> list[tuple[str, str | None, str | None]]:
    """(status, from_path, to_path) per changed file, renames detected.

    status is git's letter: A M D R C T. For A from_path is None; for
    D to_path is None; for R/C both are set.
    """
    raw = _git(root, "diff", "--name-status", "-M", "-z", base, head).decode("utf-8", "replace")
    parts = [p for p in raw.split("\0")]
    out: list[tuple[str, str | None, str | None]] = []
    i = 0
    while i < len(parts):
        status = parts[i]
        if not status:
            i += 1
            continue
        letter = status[0]
        if letter in ("R", "C"):
            out.append((letter, parts[i + 1], parts[i + 2]))
            i += 3
        elif letter == "A":
            out.append((letter, None, parts[i + 1]))
            i += 2
        elif letter == "D":
            out.append((letter, parts[i + 1], None))
            i += 2
        else:  # M, T, and anything else: same path both sides
            out.append((letter, parts[i + 1], parts[i + 1]))
            i += 2
    return out


# ── parsing one side ────────────────────────────────────────────────────


def _body_hash(source: bytes, line: int, end_line: int) -> str:
    lines = source.splitlines()
    chunk = b"\n".join(lines[max(line - 1, 0):end_line])
    return hashlib.sha256(chunk).hexdigest()


def _parse_side(repo: str, path: str, source: bytes) -> list[_Symbol]:
    language = language_for_path(path)
    if language is Language.UNKNOWN:
        return []
    if language is Language.TYPESCRIPT and is_tsx(path):
        from otter_docs.parsers.typescript import TSX_PARSER
        result = TSX_PARSER.parse(repo=repo, path=path, source=source)
    else:
        result = parse_file(repo=repo, path=path, source=source, language=language)
    if result is None:
        return []
    out: list[_Symbol] = []
    for fn in result.functions:
        out.append(_Symbol(
            guid=fn.guid, kind="function", name=fn.name, path=path,
            line=fn.line, end_line=fn.end_line,
            body_hash=_body_hash(source, fn.line, fn.end_line),
            marked=bool(_MARKER_GUID_RE.match(fn.guid)),
        ))
    for cls in result.classes:
        out.append(_Symbol(
            guid=cls.guid, kind="class", name=cls.name, path=path,
            line=cls.line, end_line=cls.end_line,
            body_hash=_body_hash(source, cls.line, cls.end_line),
            marked=bool(_MARKER_GUID_RE.match(cls.guid)),
        ))
    return out


# ── the diff ────────────────────────────────────────────────────────────


def _name_key(s: _Symbol) -> str:
    return f"{s.path}|{s.kind}|{s.name}"


def _diff_tables(base: list[_Symbol], head: list[_Symbol]) -> list[SymbolChange]:
    base_by_guid = {s.guid: s for s in base}
    head_by_guid = {s.guid: s for s in head}
    changes: list[SymbolChange] = []
    consumed_base: set[str] = set()
    consumed_head: set[str] = set()

    # Pass 1 — stable identity: same guid on both sides.
    for guid, h in head_by_guid.items():
        b = base_by_guid.get(guid)
        if b is None:
            continue
        consumed_base.add(guid)
        consumed_head.add(guid)
        moved = b.path != h.path
        if b.body_hash != h.body_hash:
            changes.append(SymbolChange(
                guid=guid, kind=h.kind, name=h.name, path=h.path, change="modified",
                marked=h.marked, line=h.line, end_line=h.end_line,
                from_path=b.path if moved else None,
            ))
        elif moved:
            changes.append(SymbolChange(
                guid=guid, kind=h.kind, name=h.name, path=h.path, change="moved",
                marked=h.marked, line=h.line, end_line=h.end_line, from_path=b.path,
            ))
        # same guid, same body, same path: unchanged (line shifts only)

    # Pass 2 — unmarked symbols matched by (path, kind, name) when unique on
    # both sides. Their derived guid embeds the line number, so a shift
    # alone would otherwise read as removed + added.
    def _unique_unmarked(symbols: list[_Symbol], consumed: set[str]) -> dict[str, _Symbol]:
        seen: dict[str, list[_Symbol]] = {}
        for s in symbols:
            if s.marked or s.guid in consumed:
                continue
            seen.setdefault(_name_key(s), []).append(s)
        return {k: v[0] for k, v in seen.items() if len(v) == 1}

    base_named = _unique_unmarked(base, consumed_base)
    head_named = _unique_unmarked(head, consumed_head)
    for key, h in head_named.items():
        b = base_named.get(key)
        if b is None:
            continue
        consumed_base.add(b.guid)
        consumed_head.add(h.guid)
        if b.body_hash != h.body_hash:
            changes.append(SymbolChange(
                guid=h.guid, kind=h.kind, name=h.name, path=h.path, change="modified",
                marked=False, line=h.line, end_line=h.end_line, from_guid=b.guid,
            ))
        # else: unchanged body, only its line moved — not a change

    # Pass 3 — what's left is genuinely added / removed.
    for guid, h in head_by_guid.items():
        if guid in consumed_head:
            continue
        changes.append(SymbolChange(
            guid=guid, kind=h.kind, name=h.name, path=h.path, change="added",
            marked=h.marked, line=h.line, end_line=h.end_line,
        ))
    for guid, b in base_by_guid.items():
        if guid in consumed_base:
            continue
        changes.append(SymbolChange(
            guid=guid, kind=b.kind, name=b.name, path=b.path, change="removed",
            marked=b.marked, line=b.line, end_line=b.end_line,
        ))

    changes.sort(key=lambda c: (c.path, c.line or 0, c.name))
    return changes


def symbol_delta(
    root: str | Path,
    *,
    base: str,
    head: str = "HEAD",
    repo: str | None = None,
) -> SymbolDelta:
    """Compute the symbol delta of `base..head` in the git repo at `root`.

    `base` may be `EMPTY_TREE` for a root commit. Both revisions are
    resolved to full shas; the result is stable for a given pair.
    """
    root = Path(root).resolve()
    repo_name = repo or root.name
    head_sha = resolve_rev(root, head)
    base_sha = base if base == EMPTY_TREE else resolve_rev(root, base)

    delta = SymbolDelta(repo=repo_name, base=base_sha, head=head_sha)
    base_symbols: list[_Symbol] = []
    head_symbols: list[_Symbol] = []

    for _status, from_path, to_path in _changed_paths(root, base_sha, head_sha):
        shown = to_path or from_path or ""
        delta.files.append(shown)
        relevant = language_for_path(shown) is not Language.UNKNOWN or (
            from_path and language_for_path(from_path) is not Language.UNKNOWN
        )
        if not relevant:
            continue
        delta.source_files.append(shown)
        if from_path is not None:
            src = _blob(root, base_sha, from_path)
            if src is not None:
                try:
                    base_symbols.extend(_parse_side(repo_name, from_path, src))
                except Exception as e:  # a parser crash must not hide the rest
                    delta.errors.append((from_path, f"base parse failed: {type(e).__name__}: {e}"))
        if to_path is not None:
            src = _blob(root, head_sha, to_path)
            if src is not None:
                try:
                    head_symbols.extend(_parse_side(repo_name, to_path, src))
                except Exception as e:
                    delta.errors.append((to_path, f"head parse failed: {type(e).__name__}: {e}"))

    delta.changes = _diff_tables(base_symbols, head_symbols)
    return delta


def commit_delta(root: str | Path, sha: str = "HEAD", *, repo: str | None = None) -> SymbolDelta:
    """The delta of a single commit against its first parent (or the empty
    tree for a root commit). Merge commits are diffed against their first
    parent, which is what `git show` does too."""
    root = Path(root).resolve()
    head_sha = resolve_rev(root, sha)
    try:
        parent = resolve_rev(root, f"{head_sha}^")
    except subprocess.CalledProcessError:
        parent = EMPTY_TREE
    return symbol_delta(root, base=parent, head=head_sha, repo=repo)
