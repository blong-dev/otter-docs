"""SourceLayoutDetector — top-level directory map with annotations.

Lists top-level files + directories at the repo root, annotates ones
whose names match the curated role table, filters build artifacts
and hidden directories.
"""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.base import register_infra_detector
from otter_docs.infra.models import SourceLayout, SourceLayoutEntry

# Curated annotation table per spec §5.4. Lowercased lookup.
_ANNOTATIONS: dict[str, str] = {
    "src": "source root",
    "lib": "source root",
    "cmd": "binaries / entry points",
    "bin": "binaries / entry points",
    "pkg": "Go package layout",
    "internal": "Go internal packages",
    "app": "application code",
    "apps": "application code (monorepo)",
    "packages": "library packages (monorepo)",
    "services": "services (monorepo)",
    "tests": "tests",
    "test": "tests",
    "spec": "tests",
    "__tests__": "tests",
    "docs": "documentation",
    "documentation": "documentation",
    "doc": "documentation",
    "infra": "infrastructure",
    "deploy": "infrastructure",
    "terraform": "infrastructure",
    "k8s": "infrastructure",
    "kubernetes": "infrastructure",
    "migrations": "database schema",
    "db": "database",
    "scripts": "utility scripts",
    "tools": "utility scripts",
    "examples": "examples",
    "example": "examples",
    "samples": "examples",
    "static": "static assets",
    "public": "static assets",
    "assets": "static assets",
    "config": "configuration",
    "configs": "configuration",
    ".github": "GitHub config",
    ".gitlab": "GitLab config",
}

# Dirs to exclude entirely (build artifacts, caches, vendored).
_HIDDEN_DIR_EXCLUDE = frozenset({
    "node_modules", "vendor", "target", "build", "dist",
    "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".otter-docs", ".tox", ".coverage", ".idea", ".vscode",
})

# Top-level files we never list (system noise + otter-docs's own output).
# OTTER.md / SYSTEM.md are excluded so re-renders are bit-identical: if
# the previous render wrote OTTER.md to the root, the next pass would
# otherwise see it as a new top-level file and the listing would
# differ between first and subsequent renders.
_HIDDEN_FILE_EXCLUDE = frozenset({
    ".DS_Store", "Thumbs.db",
    "OTTER.md", "SYSTEM.md",
})


class SourceLayoutDetector:
    kind = "source_layout"

    def detect(self, *, repo: str, repo_root: Path) -> SourceLayout | None:
        if not repo_root.is_dir():
            return None
        entries: list[SourceLayoutEntry] = []
        try:
            children = sorted(repo_root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return None
        for child in children:
            name = child.name
            is_dir = child.is_dir()
            if is_dir:
                if name in _HIDDEN_DIR_EXCLUDE:
                    continue
                # Hidden dot-dirs allowed only if explicitly annotated.
                if name.startswith(".") and name.lower() not in _ANNOTATIONS:
                    continue
            else:
                if name in _HIDDEN_FILE_EXCLUDE:
                    continue
                # Skip hidden dot-files.
                if name.startswith("."):
                    continue
            annotation = _ANNOTATIONS.get(name.lower())
            entries.append(SourceLayoutEntry(
                name=name, is_dir=is_dir, annotation=annotation,
            ))
        if not entries:
            return None
        return SourceLayout(repo=repo, entries=entries)


register_infra_detector(SourceLayoutDetector())
