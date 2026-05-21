"""ReadmeDetector — find the repo's README and extract a summary.

Looks at the repo root first; falls back to `docs/` for repos that
keep their README there. `summary` is the first non-heading paragraph
(what a reader expects to see right under the title); `h2_outline`
is the `##` heading list in document order.

Markdown-aware (skips H1, recognizes H2). For .rst we'd want a
proper docutils pass; v1 only extracts the H2 outline if the file
uses Markdown-style `##`. RST h2-equivalent (`---` underline) is
deferred.
"""

from __future__ import annotations

import re
from pathlib import Path

from otter_docs.infra.base import register_infra_detector
from otter_docs.infra.models import Readme

_CANDIDATE_NAMES = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "readme.md",
)
_FALLBACK_DIRS = (".", "docs", "doc")
_SUMMARY_MAX_CHARS = 500

# Subtree discovery — same skip-set as the dependency detector for
# consistency. A monorepo with README in each package gets one Readme
# record per package.
_SKIP_DIRS = frozenset({
    ".git", ".github", ".venv", "venv", "env", "node_modules",
    "target", "build", "dist", ".otter-docs", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
    ".tox", ".idea", ".vscode",
})
_MAX_SUBTREE_DEPTH = 3

# Files whose presence marks a directory as a "package root." Subtree
# READMEs are only emitted from such directories — that filters
# documentation-only READMEs that live in /backups, /archive, etc.
# Keep in sync with DependencyManifestDetector._CANDIDATES.
_PACKAGE_MARKERS = (
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "Gemfile",
    "requirements.txt",
)


class ReadmeDetector:
    kind = "readme"

    def detect(
        self, *, repo: str, repo_root: Path
    ) -> Readme | list[Readme] | None:
        # Root / docs fallback first — single Readme is the common case.
        path = _find_readme(repo_root)
        if path is not None:
            return _readme_from_path(path, repo, repo_root)
        # Monorepo fallback: collect subtree READMEs, but ONLY from
        # directories that also contain a dependency manifest. A
        # README without a manifest is documentation noise (e.g.
        # `backups/README.md`, `philosophy/.../README.md`,
        # `_retired/.../README.md`) — not a package root.
        results: list[Readme] = []
        for d, _depth in _walk_subdirs(repo_root, _MAX_SUBTREE_DEPTH):
            if not _is_package_root(d):
                continue
            sub_path = _find_readme_in_dir(d)
            if sub_path is not None:
                rec = _readme_from_path(sub_path, repo, repo_root)
                if rec is not None:
                    results.append(rec)
        return results or None


def _is_package_root(directory: Path) -> bool:
    return any((directory / marker).is_file() for marker in _PACKAGE_MARKERS)


def _readme_from_path(path: Path, repo: str, repo_root: Path) -> Readme | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rel = path.relative_to(repo_root).as_posix()
    return Readme(
        repo=repo,
        path=rel,
        summary=_extract_summary(text),
        h2_outline=_extract_h2(text),
    )


def _find_readme(repo_root: Path) -> Path | None:
    for d in _FALLBACK_DIRS:
        directory = repo_root / d
        if not directory.is_dir():
            continue
        hit = _find_readme_in_dir(directory)
        if hit is not None:
            return hit
    return None


def _find_readme_in_dir(directory: Path) -> Path | None:
    try:
        lower_map = {
            entry.name.lower(): entry
            for entry in directory.iterdir()
            if entry.is_file()
        }
    except OSError:
        return None
    for name in _CANDIDATE_NAMES:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _walk_subdirs(root: Path, max_depth: int):
    """Yield (directory, depth) for every directory under `root` up
    to max_depth, skipping known build / vendored / cache dirs."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name in _SKIP_DIRS:
                continue
            if entry.name.startswith(".") and entry.name not in {".github"}:
                continue
            yield entry, depth + 1
            stack.append((entry, depth + 1))


def _extract_summary(text: str) -> str:
    """Prose between the title (H1) and the first H2.

    By convention a README's "summary" is the paragraph(s) directly
    under the title, before any sectioning starts. If the document
    has an H1 and the first content after it is already an H2
    (or another heading), there is no summary — the title is the
    whole top-of-file. Returns an empty string in that case rather
    than guessing.
    """
    paragraph: list[str] = []
    saw_h1 = False
    for line in text.splitlines():
        stripped = line.strip()
        # H1 boundary: start capturing after we pass the title.
        if stripped.startswith("# ") and not saw_h1:
            saw_h1 = True
            continue
        # Any other heading (##, ###) ends the summary region.
        if stripped.startswith("#"):
            break
        if not stripped:
            # Blank line inside the summary region: end paragraph if
            # we already collected something, otherwise keep looking.
            if paragraph:
                break
            continue
        # Lines before the H1 are also valid prose for repos without
        # an explicit title — but skip leading badges (markdown image
        # / link lines that are common at the top of READMEs).
        if not saw_h1 and (stripped.startswith("![") or stripped.startswith("[!")):
            continue
        paragraph.append(stripped)
    joined = " ".join(paragraph)
    if len(joined) > _SUMMARY_MAX_CHARS:
        return joined[: _SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return joined


_H2_RE = re.compile(r"^##\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _extract_h2(text: str) -> list[str]:
    """Markdown-style `##` headings, in document order. Ignores `###` and deeper."""
    return [m.group(1).strip() for m in _H2_RE.finditer(text)]


register_infra_detector(ReadmeDetector())
