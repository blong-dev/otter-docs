"""Python cross-file resolver via jedi.

For every Python module the graph knows about, ask jedi to resolve
each call site. When the target is a function defined in another
module we *also* track, emit a CALLS edge with src=caller_guid and
dst=target_guid.

The function-GUID lookup uses the (path, line) tuple as a fingerprint
— our parser assigns guids deterministically from (repo, path, name,
line), but jedi only gives us (path, line) for the target, so we
build an index over Function/Class records keyed by (path, line) and
match against it.

Calls into stdlib, into third-party packages, or to symbols we don't
index (e.g. methods on imported classes whose source we don't have)
are dropped — those edges would be noise without an "external" node
type, which we don't have yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from otter_docs.backends.base import GraphBackend
from otter_docs.models import Edge, Language
from otter_docs.resolvers.base import register

if TYPE_CHECKING:
    pass  # jedi imported lazily inside resolve() so it stays an opt-in dep


class JediResolver:
    language = Language.PYTHON

    def resolve(
        self,
        *,
        repo: str,
        repo_root: Path,
        graph: GraphBackend,
    ) -> Iterable[Edge]:
        try:
            import jedi
        except ImportError as e:
            raise ImportError(
                "JediResolver requires the `jedi` package. Install via:\n"
                "    pip install otter-docs[python-resolver]\n"
                "or directly:\n"
                "    pip install jedi"
            ) from e

        # Build a (path, line) → guid index over every function and
        # class in the repo. Used twice per call site: once to find
        # the caller (which function contains this line), once to
        # find the callee (which function does jedi resolve to).
        line_index = self._build_line_index(repo, graph)

        # If the repo root is itself a package (has __init__.py), jedi
        # needs to see the directory *above* it to resolve in-repo
        # imports like `from gnosis.adapters import X`. Walk up until
        # we find a non-package directory.
        project_root = _find_jedi_project_root(repo_root)
        project = jedi.Project(str(project_root))
        # Track which modules we've already walked so we don't read
        # the same file twice in the rare case list_modules yields dupes.
        seen_paths: set[str] = set()

        for module in graph.list_modules(repo):
            if module.language is not Language.PYTHON:
                continue
            if module.path in seen_paths:
                continue
            seen_paths.add(module.path)
            abs_path = repo_root / module.path
            try:
                source = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                script = jedi.Script(source, project=project, path=str(abs_path))
                names = script.get_names(
                    all_scopes=True, definitions=False, references=True
                )
            except Exception:  # noqa: BLE001 — jedi can be flaky on bad source
                continue
            yield from self._edges_from_names(
                names, module_path=module.path, line_index=line_index,
            )

    @staticmethod
    def _build_line_index(
        repo: str, graph: GraphBackend
    ) -> dict[tuple[str, int], str]:
        idx: dict[tuple[str, int], str] = {}
        for fn in graph.list_functions(repo):
            idx[(fn.module_path, fn.line)] = fn.guid
        # Classes get tracked too, so calls that resolve to `__init__`
        # of a class still light up a CALLS edge to *something* — but
        # only when the resolution lands precisely on the class's line.
        for cls in graph.list_classes(repo):
            idx.setdefault((cls.module_path, cls.line), cls.guid)
        return idx

    def _edges_from_names(
        self,
        names,
        *,
        module_path: str,
        line_index: dict[tuple[str, int], str],
    ) -> Iterator[Edge]:
        for name in names:
            if name.is_definition():
                continue
            # Find which function in *this* file contains the call.
            caller_guid = _enclosing_function_guid(
                module_path=module_path, call_line=name.line, line_index=line_index,
            )
            if caller_guid is None:
                # The call sits at module level — emit no edge for
                # v0.1. Module-level call edges would need a "module
                # calls function" type we don't model.
                continue

            try:
                defs = name.goto(follow_imports=True)
            except Exception:  # noqa: BLE001
                continue
            for d in defs:
                if d.type not in ("function", "class"):
                    continue
                if d.module_path is None or d.line is None:
                    continue
                # jedi gives us an absolute path; convert back to the
                # repo-relative form we use as keys.
                rel = self._rel_path(d.module_path, line_index)
                if rel is None:
                    continue
                dst = line_index.get((rel, d.line))
                if dst is None:
                    continue
                if dst == caller_guid:
                    continue  # self-recursion; not interesting as an edge
                yield Edge(kind="CALLS", src_id=caller_guid, dst_id=dst)

    @staticmethod
    def _rel_path(
        abs_path, line_index: dict[tuple[str, int], str]
    ) -> str | None:
        """Try to recover the repo-relative path jedi-style.

        We don't have repo_root in scope here, so we match against the
        known module paths in the index. A given absolute path will
        end with the relative path of one of our modules; pick the
        longest match.
        """
        candidate = str(abs_path)
        best: str | None = None
        for path, _line in line_index.keys():
            if candidate.endswith(path) and (best is None or len(path) > len(best)):
                best = path
        return best


def _find_jedi_project_root(repo_root: Path) -> Path:
    """Find the directory jedi should use as its project root.

    Many Python repos are both a project *and* a package — they live at
    e.g. `gnosis/`, have a top-level `pyproject.toml`, and import as
    `gnosis.x.y`. For jedi to resolve those imports, the project root
    must be the parent of the outermost `__init__.py`-bearing dir.

    Algorithm: walk up from repo_root while the current dir has
    `__init__.py`. Stop when we reach a non-package dir; return that.
    For repos with no top-level `__init__.py`, return repo_root as-is.

    Capped at 8 levels of walk-up to prevent surprises with deeply-
    nested monorepos.
    """
    current = repo_root.resolve()
    for _ in range(8):
        if not (current / "__init__.py").exists():
            return current
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def _enclosing_function_guid(
    *,
    module_path: str,
    call_line: int,
    line_index: dict[tuple[str, int], str],
) -> str | None:
    """Find the function whose start_line ≤ call_line and whose body covers it.

    The line_index only carries start lines. For v0.1 we approximate by
    picking the largest function-start ≤ call_line within this module.
    A more precise version would carry end_line too — TODO for 2.4.3
    when we revisit dead_code accuracy.
    """
    best_line = -1
    best_guid: str | None = None
    for (path, line), guid in line_index.items():
        if path != module_path:
            continue
        if line <= call_line and line > best_line:
            best_line = line
            best_guid = guid
    return best_guid


register(JediResolver())
