"""TestsDetector — locate test directories + infer the test runners.

Walks for the standard test directory names (`tests/`, `__tests__/`,
`test/`, `spec/`) at any depth up to a small ceiling, plus root-level
single-file conventions (`test_*.py`, `*.test.ts`, `*.spec.ts`,
`*_test.go`). Counts test files; emits inferred runner set.

Runner inference looks at filename pattern AND nearby config files
(pytest.ini, jest.config.*, vitest.config.*, etc.). When multiple
runners coexist (common in monorepos) all of them are reported.
"""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.base import register_infra_detector
from otter_docs.infra.models import TestsLayout

# Directory names that conventionally hold tests.
_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__", "spec", "specs"})

# Source-skip set: don't descend into these even searching for tests.
_SKIP_DIR_NAMES = frozenset({
    ".git", ".github", ".venv", "venv", "env", "node_modules",
    "target", "build", "dist", ".otter-docs", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "vendor",
})

# Max walk depth from repo root — tests usually live shallow; deep walks
# would slow down on giant monorepos.
_MAX_DEPTH = 6

# Filename patterns → runner id. Both directory tests AND root-level
# stragglers get matched.
_PATTERN_RUNNERS: list[tuple[tuple[str, str], str]] = [
    # (filename startswith, filename endswith) → runner
    (("test_", ".py"), "pytest"),
    (("", "_test.py"), "pytest"),
    (("", ".test.ts"), "jest-or-vitest"),
    (("", ".test.tsx"), "jest-or-vitest"),
    (("", ".test.js"), "jest-or-vitest"),
    (("", ".test.jsx"), "jest-or-vitest"),
    (("", ".spec.ts"), "jest-or-vitest"),
    (("", ".spec.tsx"), "jest-or-vitest"),
    (("", ".spec.js"), "jest-or-vitest"),
    (("", "_test.go"), "go test"),
    (("", "_spec.rb"), "rspec"),
    (("", "_test.rb"), "minitest"),
    (("test_", ".rs"), "cargo test"),
]

# Config files that disambiguate jest-or-vitest, or that announce a
# runner even without test files yet (rare).
_CONFIG_RUNNER_HINTS: dict[str, str] = {
    "pytest.ini": "pytest",
    "pyproject.toml": "",  # checked separately for [tool.pytest.ini_options]
    "jest.config.js": "jest",
    "jest.config.ts": "jest",
    "jest.config.cjs": "jest",
    "jest.config.mjs": "jest",
    "vitest.config.js": "vitest",
    "vitest.config.ts": "vitest",
    "vitest.config.mjs": "vitest",
    ".rspec": "rspec",
}


class TestsDetector:
    kind = "tests"

    def detect(self, *, repo: str, repo_root: Path) -> TestsLayout | None:
        if not repo_root.is_dir():
            return None
        # Provisionally collect candidate test dirs (by name) and the set
        # of paths to actual matched test files. A directory only
        # graduates from candidate to a reported test dir if it contains
        # at least one matched file — that filters false positives like
        # `docs/specs/` (a specs-by-name dir holding markdown, not tests).
        candidate_dirs: list[Path] = []
        matched_files: list[Path] = []
        runners: set[str] = set()

        for path, _depth in _walk_with_depth(repo_root, _MAX_DEPTH):
            if path.is_dir():
                if path.name in _TEST_DIR_NAMES and path != repo_root:
                    candidate_dirs.append(path)
                continue
            runner = _match_runner(path.name)
            if runner is not None:
                matched_files.append(path)
                runners.add(runner)

        # A candidate dir survives only when at least one matched file
        # lives under it.
        test_dirs: list[str] = []
        for d in candidate_dirs:
            d_str = str(d)
            if any(str(f).startswith(d_str + "/") for f in matched_files):
                test_dirs.append(d.relative_to(repo_root).as_posix())
        file_count = len(matched_files)

        # Resolve `jest-or-vitest` ambiguity via config hints.
        if "jest-or-vitest" in runners:
            disambig = _disambiguate_jest_vitest(repo_root)
            runners.discard("jest-or-vitest")
            runners.add(disambig)

        # Config-only hints (no test files yet): emit anyway.
        for fname, hint in _CONFIG_RUNNER_HINTS.items():
            if not hint:
                continue
            if (repo_root / fname).is_file():
                runners.add(hint)

        # pytest pyproject.toml: only count if [tool.pytest.ini_options] present.
        if (repo_root / "pyproject.toml").is_file() and _pyproject_has_pytest_config(repo_root / "pyproject.toml"):
            runners.add("pytest")

        if not test_dirs and file_count == 0 and not runners:
            return None
        return TestsLayout(
            repo=repo,
            dirs=sorted(set(test_dirs)),
            file_count=file_count,
            runners=sorted(runners),
        )


def _walk_with_depth(root: Path, max_depth: int):
    """Yield (path, depth) entries up to max_depth, skipping _SKIP_DIR_NAMES."""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _SKIP_DIR_NAMES:
                    continue
                yield entry, depth + 1
                if depth + 1 < max_depth:
                    stack.append((entry, depth + 1))
            else:
                yield entry, depth + 1


def _match_runner(filename: str) -> str | None:
    for (prefix, suffix), runner in _PATTERN_RUNNERS:
        if prefix and not filename.startswith(prefix):
            continue
        if suffix and not filename.endswith(suffix):
            continue
        if not prefix and not suffix:
            continue
        return runner
    return None


def _disambiguate_jest_vitest(repo_root: Path) -> str:
    for name in ("vitest.config.js", "vitest.config.ts", "vitest.config.mjs"):
        if (repo_root / name).is_file():
            return "vitest"
    for name in ("jest.config.js", "jest.config.ts", "jest.config.cjs", "jest.config.mjs"):
        if (repo_root / name).is_file():
            return "jest"
    return "jest-or-vitest"  # leave generic if neither is found


def _pyproject_has_pytest_config(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "[tool.pytest" in text or ("pytest" in text.lower() and "[tool.pytest.ini_options]" in text)


register_infra_detector(TestsDetector())
