"""DependencyManifestDetector — find and parse the repo's dep manifest.

One detector, many formats. Looks at the repo root for the standard
manifest filenames in priority order; the first found wins
(monorepo subtree manifests are deferred per spec §6.1).

Per-format parsing dispatches to a small per-format function. All
parsers return `(direct, dev)` dependency lists with the same
`Dependency` shape. Formats that don't distinguish dev deps put
everything in `direct` and leave `dev` empty.

Supported in v1:
  - pyproject.toml (PEP 621 + Poetry)
  - package.json (Node)
  - go.mod
  - Cargo.toml
  - Gemfile
  - requirements.txt (fallback — only if no higher-priority manifest)

Deferred for follow-up: pom.xml, build.gradle, Podfile, composer.json,
mix.exs, pubspec.yaml. Pattern is the same; add when a real repo
needs them.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from otter_docs.infra.base import register_infra_detector
from otter_docs.infra.models import Dependency, DependencyManifest, ManifestFormat

# Order matters: first hit wins. Higher-fidelity formats come before
# lower-fidelity siblings (pyproject before requirements.txt).
_CANDIDATES: list[tuple[str, ManifestFormat]] = [
    ("pyproject.toml", "pyproject"),
    ("package.json", "package_json"),
    ("go.mod", "go_mod"),
    ("Cargo.toml", "cargo"),
    ("Gemfile", "gemfile"),
    ("requirements.txt", "requirements"),
]


# Directory names to skip when searching for subtree manifests
# (vendored deps, caches, build artifacts, otter-docs's own data).
_SKIP_DIRS = frozenset({
    ".git", ".github", ".venv", "venv", "env", "node_modules",
    "target", "build", "dist", ".otter-docs", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
    ".tox", ".idea", ".vscode",
})

# How deep we'll recurse looking for subtree manifests in a monorepo.
_MAX_SUBTREE_DEPTH = 4


class DependencyManifestDetector:
    kind = "dependencies"

    def detect(
        self, *, repo: str, repo_root: Path
    ) -> DependencyManifest | list[DependencyManifest] | None:
        # Root-first: the common single-package case. Returns the
        # highest-fidelity root manifest, exactly as before.
        root_hit = _parse_at(repo_root, repo, repo_root)
        if root_hit is not None:
            return root_hit
        # Monorepo / subtree fallback: walk for any manifest. Group by
        # parent dir so a project with `package.json` + `package-lock.json`
        # in the same dir contributes one entry.
        results: list[DependencyManifest] = []
        seen_dirs: set[Path] = set()
        for d, _depth in _walk_subdirs(repo_root, _MAX_SUBTREE_DEPTH):
            if d in seen_dirs:
                continue
            hit = _parse_at(d, repo, repo_root)
            if hit is not None:
                results.append(hit)
                seen_dirs.add(d)
        return results or None


def _parse_at(
    directory: Path, repo: str, repo_root: Path
) -> DependencyManifest | None:
    """Try each known manifest filename in `directory`; return the
    first that parses cleanly. Path on the record is repo-relative."""
    for filename, fmt in _CANDIDATES:
        path = directory / filename
        if not path.is_file():
            continue
        try:
            direct, dev, resolved_fmt = _PARSERS[fmt](path)
        except Exception:
            continue
        rel = path.relative_to(repo_root).as_posix()
        return DependencyManifest(
            repo=repo, path=rel, format=resolved_fmt,
            direct=direct, dev=dev,
        )
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


# ── per-format parsers ─────────────────────────────────────────────────


def _parse_pyproject(path: Path) -> tuple[list[Dependency], list[Dependency], ManifestFormat]:
    """PEP 621 (project.dependencies + project.optional-dependencies.dev)
    OR Poetry (tool.poetry.dependencies + tool.poetry.group.dev.dependencies).
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "project" in data and "dependencies" in data.get("project", {}):
        proj = data["project"]
        direct = [_dep_from_pep508(s) for s in proj.get("dependencies", [])]
        # optional-dependencies is dict[str, list[str]] in PEP 621; treat any
        # group whose name looks dev-ish as dev. Most repos use "dev" or "test".
        dev: list[Dependency] = []
        for group_name, deps in (proj.get("optional-dependencies") or {}).items():
            if _looks_dev_group(group_name):
                dev.extend(_dep_from_pep508(s) for s in deps)
        return direct, dev, "pyproject"
    if "tool" in data and "poetry" in data.get("tool", {}):
        poet = data["tool"]["poetry"]
        direct = [
            Dependency(name=k, version_spec=_version_of(v))
            for k, v in (poet.get("dependencies") or {}).items()
            if k != "python"  # python interpreter constraint, not a dep
        ]
        dev: list[Dependency] = []
        for group_name, group in (poet.get("group") or {}).items():
            if _looks_dev_group(group_name):
                for k, v in (group.get("dependencies") or {}).items():
                    dev.append(Dependency(name=k, version_spec=_version_of(v)))
        # Legacy: tool.poetry.dev-dependencies (pre-1.2)
        for k, v in (poet.get("dev-dependencies") or {}).items():
            dev.append(Dependency(name=k, version_spec=_version_of(v)))
        return direct, dev, "poetry"
    return [], [], "pyproject"


def _looks_dev_group(name: str) -> bool:
    return name.lower() in {"dev", "test", "tests", "lint", "ci", "docs"}


def _version_of(v: object) -> str | None:
    """Poetry dep values can be a string ("^1.2") or a dict with `version`."""
    if isinstance(v, str):
        return v or None
    if isinstance(v, dict):
        ver = v.get("version")
        return str(ver) if ver else None
    return None


def _dep_from_pep508(spec: str) -> Dependency:
    """Parse `name[extras]>=1.2,<2 ; marker` into a Dependency.

    Best-effort: we want name + version_spec, not full PEP 508 fidelity.
    The renderer just shows the strings; round-tripping isn't required.
    """
    # Strip environment marker (everything after ';').
    head = spec.split(";", 1)[0].strip()
    # Strip extras: `name[a,b]>=1` -> `name>=1`
    head = re.sub(r"\[[^\]]*\]", "", head)
    # Split name from spec at first comparator character.
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", head)
    if not m:
        return Dependency(name=head)
    name = m.group(1)
    rest = m.group(2).strip()
    return Dependency(name=name, version_spec=rest or None)


def _parse_package_json(path: Path) -> tuple[list[Dependency], list[Dependency], ManifestFormat]:
    data = json.loads(path.read_text(encoding="utf-8"))
    direct = [Dependency(name=k, version_spec=str(v)) for k, v in (data.get("dependencies") or {}).items()]
    dev = [Dependency(name=k, version_spec=str(v)) for k, v in (data.get("devDependencies") or {}).items()]
    return direct, dev, "package_json"


def _parse_go_mod(path: Path) -> tuple[list[Dependency], list[Dependency], ManifestFormat]:
    text = path.read_text(encoding="utf-8")
    direct: list[Dependency] = []
    # Block form: `require (\n  pkg v1.2.3\n  ...\n)`
    block_pat = re.compile(r"require\s*\(([^)]*)\)", re.DOTALL)
    for block in block_pat.findall(text):
        for line in block.splitlines():
            dep = _parse_go_require_line(line)
            if dep is not None:
                direct.append(dep)
    # Line form: `require pkg v1.2.3`
    line_pat = re.compile(r"^require\s+([^\s/]+\S*)\s+(\S+)", re.MULTILINE)
    for name, version in line_pat.findall(text):
        if "(" in name or ")" in name:
            continue
        direct.append(Dependency(name=name, version_spec=version))
    return direct, [], "go_mod"


def _parse_go_require_line(line: str) -> Dependency | None:
    # Strip comment (`// indirect` etc.)
    line = line.split("//", 1)[0].strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    return Dependency(name=parts[0], version_spec=parts[1])


def _parse_cargo(path: Path) -> tuple[list[Dependency], list[Dependency], ManifestFormat]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    direct = [
        Dependency(name=k, version_spec=_version_of(v))
        for k, v in (data.get("dependencies") or {}).items()
    ]
    dev = [
        Dependency(name=k, version_spec=_version_of(v))
        for k, v in (data.get("dev-dependencies") or {}).items()
    ]
    return direct, dev, "cargo"


def _parse_gemfile(path: Path) -> tuple[list[Dependency], list[Dependency], ManifestFormat]:
    """Gemfile is Ruby DSL. We pattern-match `gem "name"[, "spec"]`."""
    text = path.read_text(encoding="utf-8")
    pat = re.compile(
        r"^\s*gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?",
        re.MULTILINE,
    )
    direct = [
        Dependency(name=name, version_spec=ver or None)
        for name, ver in pat.findall(text)
    ]
    return direct, [], "gemfile"


def _parse_requirements(path: Path) -> tuple[list[Dependency], list[Dependency], ManifestFormat]:
    direct: list[Dependency] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Drop env markers / extras / comments
        line = line.split(";", 1)[0].strip()
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        direct.append(_dep_from_pep508(line))
    return direct, [], "requirements"


_PARSERS = {
    "pyproject": _parse_pyproject,
    "package_json": _parse_package_json,
    "go_mod": _parse_go_mod,
    "cargo": _parse_cargo,
    "gemfile": _parse_gemfile,
    "requirements": _parse_requirements,
}


register_infra_detector(DependencyManifestDetector())
