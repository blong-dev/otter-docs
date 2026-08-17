"""doc.divergence — hand-written docs (README / CLAUDE / specs) vs the code
they name.

The docstring detector (`description.divergence` + `confirm_description`)
judges a *symbol's* own docstring against its body. Prose docs are different:
a README or a spec makes claims about the codebase and references code by
**explicit mention** — a backtick span (`confirm_redundancy`), a file path
(`gnosis/orchestrator/scheduler.py`), a markdown link. That mention is an
*exact* link, not a cosine guess, so the candidate signal here is
deterministic and high-precision by construction — the opposite of the noisy
embedding pre-filter the docstring path needs.

This module is filesystem-bound (it reads `.md` off disk), so it is **not** a
registered graph detector — detectors only get `(repo, graph)`. It is driven
from `Repo.doc_findings()`, which has `repo_root`. Each emitted
`doc.divergence` Finding is a *candidate* (this doc section names this symbol);
the LLM judge `confirm_doc_description` is what rules accurate/stale/wrong.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from otter_docs.backends.base import GraphBackend
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location

logger = logging.getLogger(__name__)

# Docs we treat as hand-written prose about the codebase. Globs are matched
# relative to repo_root. Kept deliberately small + high-signal; a repo can
# widen it via `Repo.doc_findings(doc_globs=...)`.
DEFAULT_DOC_GLOBS: tuple[str, ...] = (
    "README.md",
    "README.markdown",
    "CLAUDE.md",
    "AGENTS.md",
    "docs/specs/**/*.md",
    "docs/**/*.md",
)

# A backtick token shorter than this is dropped — `run`, `get`, `id` collide
# with too many symbols to judge a specific claim.
MIN_SYMBOL_LEN = 4
# A name that resolves to more than this many symbols is too ambiguous to pin
# a claim on; skip it rather than guess the wrong body.
MAX_NAME_FANOUT = 4
# Per-section cap on emitted candidates (most-specific mentions first).
DEFAULT_MAX_SYMBOLS_PER_CHUNK = 3
# Per-doc cap; if a doc would exceed this we log what we dropped (no silent
# truncation — the OD signal-quality rule).
MAX_CANDIDATES_PER_DOC = 40
# Section text handed to the judge is truncated to keep the prompt bounded.
MAX_SECTION_CHARS = 4000

_SOURCE_EXTS = (".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".rs", ".java")

# Inline `code span` — but NOT inside a fenced ``` block (handled by stripping
# fences first). Non-greedy, single-backtick spans.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Markdown link target: [text](target)
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# Fenced code block (```lang ... ```), stripped before mention extraction so
# example code isn't mistaken for a claim.
_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
# ATX markdown heading: leading #'s + text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class DocSection:
    """A markdown heading and the body beneath it (until the next heading)."""

    __slots__ = ("heading", "level", "start_line", "text")

    def __init__(self, heading: str, level: int, start_line: int, text: str) -> None:
        self.heading = heading
        self.level = level
        self.start_line = start_line
        self.text = text


def split_sections(doc_text: str) -> list[DocSection]:
    """Split a markdown doc into (heading, body) sections.

    Each ATX heading opens a section that runs until the next heading of any
    level. Content before the first heading becomes a synthetic leading
    section (heading ""). A doc with no headings is a single section.
    """
    lines = doc_text.splitlines()
    sections: list[DocSection] = []
    cur_heading = ""
    cur_level = 0
    cur_start = 1
    cur_body: list[str] = []
    in_fence = False

    def _flush() -> None:
        text = "\n".join(cur_body).strip()
        if cur_heading or text:
            sections.append(DocSection(cur_heading, cur_level, cur_start, text))

    for i, line in enumerate(lines, start=1):
        # A ``` toggles fenced-code state; a `#` inside a fence is a code
        # comment, not a heading (e.g. Python `# ...`).
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            cur_body.append(line)
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            _flush()
            cur_level = len(m.group(1))
            cur_heading = m.group(2).strip()
            cur_start = i
            cur_body = []
        else:
            cur_body.append(line)
    _flush()
    return sections


class _MentionIndex:
    """Resolves a doc token (backtick span / path / link) to graph symbols.

    Built once per `build_doc_findings` run from the graph. Ambiguous
    (>MAX_NAME_FANOUT) and short (<MIN_SYMBOL_LEN) names are dropped so a
    mention only resolves when it pins a specific symbol.
    """

    def __init__(self, repo: str, graph: GraphBackend) -> None:
        self._by_name: dict[str, list[tuple[str, str, str, str]]] = {}
        # module path index: full repo-relative path + basename → module path
        self._by_path: dict[str, str] = {}
        self._by_basename: dict[str, list[str]] = {}

        for fn in graph.list_functions(repo):
            self._add_name(fn.name, ("function", fn.guid, fn.name, fn.module_path))
        for cls in graph.list_classes(repo):
            self._add_name(cls.name, ("class", cls.guid, cls.name, cls.module_path))
        for mod in graph.list_modules(repo):
            self._by_path[mod.path] = mod.path
            self._by_basename.setdefault(Path(mod.path).name, []).append(mod.path)

    def _add_name(self, name: str, entry: tuple[str, str, str, str]) -> None:
        if len(name) < MIN_SYMBOL_LEN:
            return
        self._by_name.setdefault(name, []).append(entry)

    def resolve(self, token: str) -> tuple[str, str, str, str] | None:
        """Resolve one token to (kind, guid, name, path), or None.

        kind ∈ {"module", "class", "function"}. Paths win over names (more
        specific). Returns None on miss, ambiguity, or too-short token.
        """
        tok = token.strip().strip("`").strip()
        if not tok:
            return None
        # strip a trailing call `foo()` and a leading language sigil
        tok = tok.rstrip("()")
        # Path-like: contains a slash or ends in a source extension.
        if "/" in tok or tok.endswith(_SOURCE_EXTS):
            path = self._resolve_path(tok)
            if path is not None:
                return ("module", "", Path(path).name, path)
            # Dotted-but-not-path (e.g. Repo.confirm_description) falls through
            if "/" in tok:
                return None
        # Dotted symbol: try last segment as the name (Class.method → method).
        candidates = [tok]
        if "." in tok:
            candidates.append(tok.rsplit(".", 1)[-1])
            candidates.append(tok.split(".", 1)[0])
        for cand in candidates:
            hit = self._resolve_name(cand)
            if hit is not None:
                return hit
        return None

    def _resolve_path(self, tok: str) -> str | None:
        if tok in self._by_path:
            return tok
        base = Path(tok).name
        matches = self._by_basename.get(base, [])
        # If the token is a suffix of exactly one known module path, take it.
        suffixed = [p for p in matches if p == tok or p.endswith("/" + tok)]
        if len(suffixed) == 1:
            return suffixed[0]
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_name(self, name: str) -> tuple[str, str, str, str] | None:
        if len(name) < MIN_SYMBOL_LEN:
            return None
        entries = self._by_name.get(name)
        if not entries or len(entries) > MAX_NAME_FANOUT:
            return None
        # Prefer a class over a function on a tie (docs more often name types),
        # and require the resolution to be unambiguous within its kind.
        classes = [e for e in entries if e[0] == "class"]
        funcs = [e for e in entries if e[0] == "function"]
        if len(classes) == 1:
            return classes[0]
        if not classes and len(funcs) == 1:
            return funcs[0]
        return None


def _extract_tokens(section_text: str) -> list[str]:
    """Ordered, de-duplicated mention tokens from a section (fences removed)."""
    body = _FENCE_RE.sub(" ", section_text)
    tokens: list[str] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(body):
        t = m.group(1).split("#", 1)[0]
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)
    for m in _INLINE_CODE_RE.finditer(body):
        t = m.group(1).strip()
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)
    return tokens


# most-specific first, so the per-section cap keeps the strongest mentions
_KIND_RANK = {"module": 0, "class": 1, "function": 2}


def build_doc_findings(
    repo: str,
    repo_root: Path,
    graph: GraphBackend,
    *,
    doc_globs: tuple[str, ...] = DEFAULT_DOC_GLOBS,
    max_symbols_per_chunk: int = DEFAULT_MAX_SYMBOLS_PER_CHUNK,
) -> list[Finding]:
    """Scan hand-written docs and emit one `doc.divergence` candidate per
    (doc section, resolved symbol). Pure I/O + graph lookups — no LLM.

    See module docstring. The Finding's Location points at the *doc*; the
    referenced symbol travels in `evidence` for `confirm_doc_description`.
    """
    from otter_docs.discovery import is_test_path

    index = _MentionIndex(repo, graph)
    findings: list[Finding] = []

    seen_docs: set[Path] = set()
    for pattern in doc_globs:
        for path in sorted(repo_root.glob(pattern)):
            if path in seen_docs or not path.is_file():
                continue
            seen_docs.add(path)
            rel = path.relative_to(repo_root).as_posix()
            if is_test_path(rel):
                continue
            try:
                doc_text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.debug("doc unreadable, skipping: %s", rel, exc_info=True)
                continue
            findings.extend(
                _findings_for_doc(repo, rel, doc_text, index, max_symbols_per_chunk)
            )
    return findings


def _findings_for_doc(
    repo: str,
    rel: str,
    doc_text: str,
    index: _MentionIndex,
    max_symbols_per_chunk: int,
) -> list[Finding]:
    out: list[Finding] = []
    for section in split_sections(doc_text):
        if not section.text:
            continue
        resolved: list[tuple[str, str, str, str]] = []
        seen_guids: set[str] = set()
        for tok in _extract_tokens(section.text):
            hit = index.resolve(tok)
            if hit is None:
                continue
            key = hit[1] or f"module:{hit[3]}"
            if key in seen_guids:
                continue
            seen_guids.add(key)
            resolved.append(hit)
        if not resolved:
            continue
        resolved.sort(key=lambda e: _KIND_RANK.get(e[0], 9))
        for kind, guid, name, sym_path in resolved[:max_symbols_per_chunk]:
            if len(out) >= MAX_CANDIDATES_PER_DOC:
                logger.info(
                    "doc.divergence: capped candidates for %s at %d (dropping rest)",
                    rel, MAX_CANDIDATES_PER_DOC,
                )
                return out
            out.append(_make_finding(repo, rel, section, kind, guid, name, sym_path))
    return out


def _make_finding(
    repo: str,
    rel: str,
    section: DocSection,
    sym_kind: str,
    guid: str,
    name: str,
    sym_path: str,
) -> Finding:
    heading = section.heading or "(top)"
    return Finding(
        # confidence is a fixed modest recall signal — the deterministic
        # mention is the candidate, the LLM judge is the gate. The publisher
        # bumps confidence to the verdict's on confirm (OD-9 lesson).
        kind="doc.divergence",
        confidence=0.5,
        locations=[Location(repo=repo, path=rel, line=section.start_line)],
        evidence={
            "doc_path": rel,
            "heading": section.heading,
            "section_text": section.text[:MAX_SECTION_CHARS],
            "symbol_kind": sym_kind,
            "symbol_guid": guid,
            "symbol_name": name,
            "symbol_path": sym_path,
        },
        recommendation=Recommendation(
            summary=(
                f"Check doc `{rel}` § {heading} against `{name}` "
                f"({sym_kind}) — it may have drifted"
            ),
            rationale=(
                f"The section “{heading}” in {rel} references "
                f"`{name}` ({sym_path}). Confirm the prose still describes "
                f"what that code does; if the code moved on, update the doc."
            ),
            blast_radius=[guid] if guid else [sym_path],
        ),
        source_detector="doc.divergence",
    )
