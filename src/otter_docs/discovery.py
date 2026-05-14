"""File discovery for `Repo.scan()`.

Walks a repo root and yields (relative_path, Language) pairs for files
we know how to parse. Excludes common build / vendor / VCS directories
so the cost of scanning a checkout is predictable.

The defaults are conservative — they aim to mirror what a reasonable
developer would expect "scan my repo" to mean. Future phases can layer
.gitignore awareness on top via the optional `respect_gitignore` flag.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from otter_docs.models import Language


# Directories never worth walking. Keep this list short and obvious;
# anything project-specific belongs in the user's .otter-docs config
# (a later phase). This is the floor, not the policy.
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".otter-docs",
    "target",         # Rust / sbt build output — common enough to default
    "vendor",         # Go's vendored deps
    ".next",          # Next.js build cache
    ".turbo",
    "coverage",
})

# Extension → Language. The TSX parser is picked separately by
# `language_for_path` because both .ts and .tsx parse-dispatch to
# Language.TYPESCRIPT but use different underlying grammars.
_EXTENSION_LANGUAGES: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".go": Language.GO,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
}


def language_for_path(path: str | Path) -> Language:
    """Resolve a path to a known Language, or UNKNOWN if we can't parse it."""
    suffix = Path(path).suffix.lower()
    return _EXTENSION_LANGUAGES.get(suffix, Language.UNKNOWN)


def is_tsx(path: str | Path) -> bool:
    """True iff this is a .tsx file (uses the TSX-flavored grammar)."""
    return Path(path).suffix.lower() == ".tsx"


def iter_source_files(
    root: str | Path,
    *,
    excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS,
) -> Iterator[tuple[Path, Language]]:
    """Walk `root` and yield (absolute_path, language) for parseable files.

    Directory exclusions are checked by name only, not by full path, so
    a top-level `node_modules` and a deeply nested one are both skipped.
    Symlinks are followed at the root but not below it, to avoid loops.
    """
    root_path = Path(root).resolve()
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=False):
        # Mutate dirnames in place so os.walk skips the pruned dirs.
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for fname in filenames:
            lang = language_for_path(fname)
            if lang is Language.UNKNOWN:
                continue
            yield Path(dirpath) / fname, lang
