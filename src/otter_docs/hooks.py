"""Git hook installer.

Writes pre-commit and pre-push hooks that call the otter-docs CLI.
The hooks are deliberately thin shell scripts that shell out to
`otter-docs` — no Python logic embedded — so they stay debuggable
and the library can evolve without rewriting installed hooks.

Design choices learned from the v3 doc-automation audit:
  - pre-commit re-stages ONLY the generated doc, never `git add -A`.
  - hooks never `--amend` or rewrite history.
  - a hook failure prints but does not block the commit/push (doc
    generation is best-effort; a broken scan shouldn't wedge a commit).
"""

from __future__ import annotations

import stat
from pathlib import Path

_PRE_COMMIT = """\
#!/bin/sh
# otter-docs pre-commit:
#   1. assign inline `# guid:` / `// guid:` markers to staged source
#      files that are missing them (idempotent, mutates source)
#   2. re-stage ONLY those touched files so markers land in the commit
#   3. regenerate the doc and stage it
# Best-effort: never blocks the commit.
#
# Re-staging is scoped to files already staged by the developer — we
# never blanket-add, never sweep unstaged work. Post-P7 safer-hooks
# pattern carried forward.
set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 0
if ! command -v otter-docs >/dev/null 2>&1; then
  exit 0
fi
# Staged source files (added/modified/renamed). Word-split is the
# portable read; paths with whitespace are out of scope for this hook.
FILES=$(git diff --cached --name-only --diff-filter=ACMR -- \\
  '*.py' '*.go' '*.ts' '*.tsx' '*.js' '*.jsx' 2>/dev/null)
if [ -n "$FILES" ]; then
  # shellcheck disable=SC2086
  otter-docs assign-guids $FILES >/dev/null 2>&1 || \\
    echo "otter-docs: assign-guids failed (commit proceeds)" >&2
  # shellcheck disable=SC2086
  git add -- $FILES 2>/dev/null || true
fi
otter-docs render "$ROOT" --out "{out}" --no-resolve >/dev/null 2>&1 || \\
  echo "otter-docs: render failed (commit proceeds)" >&2
git add -- "{out}" 2>/dev/null || true
exit 0
"""

_PRE_PUSH = """\
#!/bin/sh
# otter-docs pre-push — full scan + resolve + doc refresh.
# Best-effort: never blocks the push, never amends history.
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 0
if command -v otter-docs >/dev/null 2>&1; then
  otter-docs render "$ROOT" --out "{out}" >/dev/null 2>&1 || \\
    echo "otter-docs: render failed (push proceeds)" >&2
fi
exit 0
"""


def install_hooks(repo_root: Path, *, out: str = "SYSTEM.md") -> list[Path]:
    """Install pre-commit + pre-push hooks. Returns the paths written.

    Returns [] if `repo_root` has no `.git` directory (caller decides
    whether that's an error). Existing hooks are overwritten — we own
    these filenames by convention; a project mixing otter-docs hooks
    with others should use a hook multiplexer.
    """
    git_dir = repo_root / ".git"
    if not git_dir.is_dir():
        return []
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, template in (("pre-commit", _PRE_COMMIT), ("pre-push", _PRE_PUSH)):
        path = hooks_dir / name
        path.write_text(template.format(out=out), encoding="utf-8")
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(path)
    return written
