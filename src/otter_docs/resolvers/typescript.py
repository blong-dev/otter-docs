"""TypeScript / TSX cross-file resolver via typescript-language-server.

Pattern mirrors JediResolver: walk each file, find call sites with
tree-sitter (we already do this in the parser, so we re-derive call
positions here from the TS source), ask the LSP server for the
definition, map back to indexed function/class guids.

Requires `typescript-language-server` and `typescript` on PATH. The
resolver does NOT auto-register if the binary is missing — that way
`Repo.resolve()` quietly skips TypeScript when the tooling isn't
available, instead of crashing.

Performance notes: tsserver indexes the project on first didOpen,
then handles definition queries near-instantly. We open every TS
file once at start, then batch all definition requests. For repos
with thousands of files this trades latency at the start for fast
per-query cost.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import tree_sitter_typescript
from tree_sitter import Language as TsLanguage, Node, Parser

from otter_docs.backends.base import GraphBackend
from otter_docs.models import Edge, Language
from otter_docs.resolvers.base import register
from otter_docs.resolvers.lsp import LspClient, LspError

if TYPE_CHECKING:
    pass


_LANGUAGE_TS = TsLanguage(tree_sitter_typescript.language_typescript())
_LANGUAGE_TSX = TsLanguage(tree_sitter_typescript.language_tsx())
_PARSER_TS = Parser(_LANGUAGE_TS)
_PARSER_TSX = Parser(_LANGUAGE_TSX)


# Seconds to wait after didOpen before issuing definition requests.
# tsserver indexes asynchronously; queries issued too early return
# bad results (e.g. resolving to the enclosing function instead of
# the actual target). Empirically 1.5s is enough for small repos and
# 3s covers larger ones. We use 2s as a middle ground.
INDEX_WAIT_SECONDS = 2.0


# Heuristic: when more than this many TS files are in the repo, the
# resolver gives up rather than holding the LSP session open for too
# long. Real-world TS repos with thousands of files need a smarter
# batching strategy that's out of scope for v0.1.
MAX_FILES = 1000


def _binary_available() -> bool:
    return shutil.which("typescript-language-server") is not None


class TypeScriptResolver:
    language = Language.TYPESCRIPT

    def resolve(
        self,
        *,
        repo: str,
        repo_root: Path,
        graph: GraphBackend,
    ) -> Iterable[Edge]:
        if not _binary_available():
            return
        # Index every TS/TSX function + class by (path, line) so we can
        # map jedi-style — i.e. when tsserver tells us a definition is
        # at b.ts line 0, we look up the function/class there.
        line_index = self._build_line_index(repo, graph)
        ts_modules = [
            m for m in graph.list_modules(repo)
            if m.language is Language.TYPESCRIPT
        ]
        if not ts_modules:
            return
        if len(ts_modules) > MAX_FILES:
            # Defer to a future smarter resolver — bailing rather than
            # spending minutes is honest about the limit.
            return

        with LspClient(
            ["typescript-language-server", "--stdio"],
            cwd=str(repo_root),
            timeout=30.0,
        ) as client:
            try:
                client.initialize(root_uri=f"file://{repo_root.resolve()}")
                client.notify("initialized", {})
            except LspError:
                return

            # Open every file so tsserver builds its full project graph.
            sources: dict[str, str] = {}
            for module in ts_modules:
                abs_path = (repo_root / module.path).resolve()
                try:
                    text = abs_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                sources[module.path] = text
                lang_id = "typescriptreact" if module.path.endswith(".tsx") else "typescript"
                try:
                    client.did_open(
                        uri=f"file://{abs_path}", language_id=lang_id, text=text,
                    )
                except LspError:
                    continue

            # Let tsserver index before we start asking questions.
            time.sleep(INDEX_WAIT_SECONDS)

            yield from self._edges_for_modules(
                client=client, ts_modules=ts_modules, repo_root=repo_root,
                sources=sources, line_index=line_index,
            )

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_line_index(
        repo: str, graph: GraphBackend,
    ) -> dict[tuple[str, int], str]:
        idx: dict[tuple[str, int], str] = {}
        for fn in graph.list_functions(repo):
            idx[(fn.module_path, fn.line)] = fn.guid
        for cls in graph.list_classes(repo):
            idx.setdefault((cls.module_path, cls.line), cls.guid)
        return idx

    def _edges_for_modules(
        self,
        *,
        client: LspClient,
        ts_modules,
        repo_root: Path,
        sources: dict[str, str],
        line_index: dict[tuple[str, int], str],
    ) -> Iterator[Edge]:
        for module in ts_modules:
            text = sources.get(module.path)
            if text is None:
                continue
            abs_uri = f"file://{(repo_root / module.path).resolve()}"
            parser = _PARSER_TSX if module.path.endswith(".tsx") else _PARSER_TS
            try:
                tree = parser.parse(text.encode("utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for line, character in _iter_call_positions(tree.root_node):
                caller_guid = _enclosing_function_guid(
                    module_path=module.path,
                    call_line=line + 1,  # tree-sitter is 0-based; our guids use 1-based
                    line_index=line_index,
                )
                if caller_guid is None:
                    continue
                try:
                    defs = client.definition(
                        uri=abs_uri, line=line, character=character,
                    )
                except LspError:
                    continue
                for d in defs:
                    target = self._map_definition(d, repo_root, line_index)
                    if target is None or target == caller_guid:
                        continue
                    yield Edge(kind="CALLS", src_id=caller_guid, dst_id=target)

    @staticmethod
    def _map_definition(
        d: dict, repo_root: Path, line_index: dict[tuple[str, int], str],
    ) -> str | None:
        uri = d.get("uri", "")
        rng = d.get("range") or {}
        start = rng.get("start") or {}
        if not isinstance(uri, str) or not uri.startswith("file://"):
            return None
        target_line = start.get("line")
        if target_line is None:
            return None
        # uri is absolute — convert back to repo-relative.
        abs_path = Path(uri[len("file://"):])
        try:
            rel = abs_path.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return None
        # tsserver returns 0-based; our guids are 1-based.
        return _nearest_definition(rel, target_line + 1, line_index)


def _iter_call_positions(root: Node) -> Iterator[tuple[int, int]]:
    """Yield (line, column) for the *callee identifier* of each call site.

    A call_expression's `function` field is the callee — for `helper()`
    it's the identifier `helper`; for `obj.method()` it's the
    member_expression, and we point at the property identifier so
    tsserver resolves it to the method on the type.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if func is not None:
                target = _target_position(func)
                if target is not None:
                    yield target
        for child in node.named_children:
            stack.append(child)


def _target_position(func: Node) -> tuple[int, int] | None:
    if func.type == "identifier":
        return (func.start_point.row, func.start_point.column)
    if func.type == "member_expression":
        prop = func.child_by_field_name("property")
        if prop is not None:
            return (prop.start_point.row, prop.start_point.column)
    return None


def _enclosing_function_guid(
    *,
    module_path: str,
    call_line: int,
    line_index: dict[tuple[str, int], str],
) -> str | None:
    best_line = -1
    best_guid: str | None = None
    for (path, line), guid in line_index.items():
        if path != module_path:
            continue
        if line <= call_line and line > best_line:
            best_line = line
            best_guid = guid
    return best_guid


def _nearest_definition(
    path: str,
    target_line: int,
    line_index: dict[tuple[str, int], str],
) -> str | None:
    """Find the function/class whose start line ≤ target_line.

    tsserver returns the start of the symbol's identifier, which may
    not exactly match our parser's "line of the def keyword" — for
    `export function foo(...)` they happen to agree; for arrow
    functions assigned to const, our line is the `const` line. The
    nearest-not-after rule handles both.
    """
    best_line = -1
    best_guid: str | None = None
    for (p, line), guid in line_index.items():
        if p != path:
            continue
        if line <= target_line and line > best_line:
            best_line = line
            best_guid = guid
    return best_guid


# Register only when the binary is on PATH so a repo with no TS tooling
# doesn't see a crash when it calls Repo.resolve().
if _binary_available():
    register(TypeScriptResolver())
