"""Go cross-file resolver via gopls.

Mirrors TypeScriptResolver: walk each .go file, use tree-sitter to
find call sites, ask gopls for definitions, map to indexed function/
class GUIDs.

Requires `gopls` (and Go itself) on PATH. Install:

    # Ubuntu/Debian
    sudo apt install golang-go
    go install golang.org/x/tools/gopls@latest
    # add ~/go/bin to PATH

Status: code is in place but **not yet validated against a live gopls**
because the build machine doesn't have Go installed. The LSP wire
shape is identical to the TypeScript path (which IS validated end-
to-end), so the bulk of the risk is in Go-specific quirks: how gopls
resolves method-on-receiver-type, whether `go.mod` discovery needs
extra plumbing, and how packages outside the workspace are surfaced.
The first user pointing otter-docs at a real Go repo will surface
those. Until then this is on-paper coverage.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import tree_sitter_go
from tree_sitter import Language as TsLanguage, Node, Parser

from otter_docs.backends.base import GraphBackend
from otter_docs.models import Edge, Language
from otter_docs.resolvers.base import register
from otter_docs.resolvers.lsp import LspClient, LspError

if TYPE_CHECKING:
    pass


_GO_LANGUAGE = TsLanguage(tree_sitter_go.language())
_GO_PARSER = Parser(_GO_LANGUAGE)


INDEX_WAIT_SECONDS = 2.0
MAX_FILES = 2000  # gopls handles big repos better than tsserver


def _binary_available() -> bool:
    return shutil.which("gopls") is not None


class GoResolver:
    language = Language.GO

    def resolve(
        self,
        *,
        repo: str,
        repo_root: Path,
        graph: GraphBackend,
    ) -> Iterable[Edge]:
        if not _binary_available():
            return
        line_index = self._build_line_index(repo, graph)
        go_modules = [
            m for m in graph.list_modules(repo)
            if m.language is Language.GO
        ]
        if not go_modules or len(go_modules) > MAX_FILES:
            return

        with LspClient(["gopls", "serve"], cwd=str(repo_root), timeout=30.0) as client:
            try:
                client.initialize(root_uri=f"file://{repo_root.resolve()}")
                client.notify("initialized", {})
            except LspError:
                return

            sources: dict[str, str] = {}
            for module in go_modules:
                abs_path = (repo_root / module.path).resolve()
                try:
                    text = abs_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                sources[module.path] = text
                try:
                    client.did_open(
                        uri=f"file://{abs_path}", language_id="go", text=text,
                    )
                except LspError:
                    continue

            time.sleep(INDEX_WAIT_SECONDS)
            yield from self._edges_for_modules(
                client=client, modules=go_modules, repo_root=repo_root,
                sources=sources, line_index=line_index,
            )

    # ── helpers (mirror TypeScript resolver) ─────────────────────────

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
        modules,
        repo_root: Path,
        sources: dict[str, str],
        line_index: dict[tuple[str, int], str],
    ) -> Iterator[Edge]:
        for module in modules:
            text = sources.get(module.path)
            if text is None:
                continue
            abs_uri = f"file://{(repo_root / module.path).resolve()}"
            try:
                tree = _GO_PARSER.parse(text.encode("utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for line, character in _iter_call_positions(tree.root_node):
                caller_guid = _enclosing_function_guid(
                    module_path=module.path, call_line=line + 1,
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
                    target = _map_definition(d, repo_root, line_index)
                    if target is None or target == caller_guid:
                        continue
                    yield Edge(kind="CALLS", src_id=caller_guid, dst_id=target)


def _iter_call_positions(root: Node) -> Iterator[tuple[int, int]]:
    """Yield (line, column) for the *callee identifier* of each call site.

    For `foo()`: the identifier `foo`. For `pkg.Foo()`: the field
    identifier `Foo` in the selector_expression — gopls resolves it
    to the exported function/method.
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
    if func.type == "selector_expression":
        field = func.child_by_field_name("field")
        if field is not None:
            return (field.start_point.row, field.start_point.column)
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
    abs_path = Path(uri[len("file://"):])
    try:
        rel = abs_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None
    return _nearest_definition(rel, target_line + 1, line_index)


def _nearest_definition(
    path: str,
    target_line: int,
    line_index: dict[tuple[str, int], str],
) -> str | None:
    best_line = -1
    best_guid: str | None = None
    for (p, line), guid in line_index.items():
        if p != path:
            continue
        if line <= target_line and line > best_line:
            best_line = line
            best_guid = guid
    return best_guid


if _binary_available():
    register(GoResolver())
