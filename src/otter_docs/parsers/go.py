"""Go source parser via tree-sitter.

Extracts:
  - ModuleRecord (one per .go file)
  - FunctionRecord for `func X(...)` and `func (r R) X(...)` methods
  - ClassRecord for each top-level `type T struct/interface` declaration
  - IMPORTS edges from `import "..."` blocks
  - Intra-file CALLS edges by callee identifier match

Receiver methods carry the receiver type in their name as "R.X" so
downstream tools can group method sets without an extra edge type.
"""

from __future__ import annotations

import hashlib

import tree_sitter_go
from tree_sitter import Language, Node, Parser

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language as Lang,
    ModuleRecord,
)
from otter_docs.parsers.base import ParseResult, register


_LANGUAGE = Language(tree_sitter_go.language())
_PARSER = Parser(_LANGUAGE)


def _guid(repo: str, path: str, name: str, line: int) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{repo}|{path}|{name}|{line}".encode())
    return h.hexdigest()


def _name(node: Node, field: str = "name") -> str:
    n = node.child_by_field_name(field)
    return n.text.decode("utf-8", errors="replace") if n else ""


def _imports(root: Node) -> list[str]:
    out: list[str] = []
    for child in root.named_children:
        if child.type != "import_declaration":
            continue
        for c in child.named_children:
            if c.type == "import_spec":
                path = c.child_by_field_name("path")
                if path is not None:
                    txt = path.text.decode("utf-8", errors="replace").strip('"`')
                    out.append(txt)
            elif c.type == "import_spec_list":
                for spec in c.named_children:
                    if spec.type == "import_spec":
                        path = spec.child_by_field_name("path")
                        if path is not None:
                            out.append(
                                path.text.decode("utf-8", errors="replace").strip('"`')
                            )
    return out


def _receiver_type(fn_node: Node) -> str:
    rec = fn_node.child_by_field_name("receiver")
    if rec is None:
        return ""
    # rec is a parameter_list with one parameter_declaration.
    for pd in rec.named_children:
        if pd.type == "parameter_declaration":
            t = pd.child_by_field_name("type")
            if t is None:
                continue
            txt = t.text.decode("utf-8", errors="replace").lstrip("*")
            return txt
    return ""


def _args(fn_node: Node) -> list[str]:
    params = fn_node.child_by_field_name("parameters")
    if params is None:
        return []
    out: list[str] = []
    for p in params.named_children:
        if p.type != "parameter_declaration":
            continue
        for c in p.named_children:
            if c.type == "identifier":
                out.append(c.text.decode("utf-8", errors="replace"))
    return out


def _collect_calls(body: Node, sink: list[str]) -> None:
    if body is None:
        return
    for c in body.named_children:
        if c.type == "call_expression":
            func = c.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    sink.append(func.text.decode("utf-8", errors="replace"))
                elif func.type == "selector_expression":
                    field = func.child_by_field_name("field")
                    if field is not None:
                        sink.append(field.text.decode("utf-8", errors="replace"))
        _collect_calls(c, sink)


class GoParser:
    language = Lang.GO

    def parse(self, *, repo: str, path: str, source: bytes) -> ParseResult:
        tree = _PARSER.parse(source)
        root = tree.root_node
        module = ModuleRecord(
            repo=repo, path=path, language=Lang.GO, imports=_imports(root)
        )

        functions: list[FunctionRecord] = []
        classes: list[ClassRecord] = []
        edges: list[Edge] = []
        pending_calls: list[tuple[str, str]] = []

        for child in root.named_children:
            if child.type == "function_declaration":
                name = _name(child)
                line = child.start_point.row + 1
                end = child.end_point.row + 1
                guid = _guid(repo, path, name, line)
                fn = FunctionRecord(
                    repo=repo, guid=guid, name=name, module_path=path,
                    line=line, end_line=end, args=_args(child),
                )
                functions.append(fn)
                # Capture intra-file calls within this function's body.
                body = child.child_by_field_name("body")
                if body is not None:
                    call_names: list[str] = []
                    _collect_calls(body, call_names)
                    for cn in call_names:
                        pending_calls.append((guid, cn))

            elif child.type == "method_declaration":
                name = _name(child)
                recv = _receiver_type(child)
                qualified = f"{recv}.{name}" if recv else name
                line = child.start_point.row + 1
                end = child.end_point.row + 1
                guid = _guid(repo, path, qualified, line)
                fn = FunctionRecord(
                    repo=repo, guid=guid, name=qualified, module_path=path,
                    line=line, end_line=end, args=_args(child),
                )
                functions.append(fn)
                body = child.child_by_field_name("body")
                if body is not None:
                    call_names: list[str] = []
                    _collect_calls(body, call_names)
                    for cn in call_names:
                        pending_calls.append((guid, cn))

            elif child.type == "type_declaration":
                # `type T struct {…}` / `type T interface {…}` — emit class
                # records so downstream tooling can browse types uniformly
                # with Python/TS classes.
                for spec in child.named_children:
                    if spec.type == "type_spec":
                        type_name = _name(spec)
                        line = spec.start_point.row + 1
                        end = spec.end_point.row + 1
                        guid = _guid(repo, path, f"class:{type_name}", line)
                        classes.append(ClassRecord(
                            repo=repo, guid=guid, name=type_name, module_path=path,
                            line=line, end_line=end,
                        ))

        by_name: dict[str, str] = {}
        for fn in functions:
            # Allow both qualified ("R.X") and bare ("X") name lookups.
            by_name.setdefault(fn.name, fn.guid)
            if "." in fn.name:
                bare = fn.name.split(".", 1)[1]
                by_name.setdefault(bare, fn.guid)
        for caller, callee_name in pending_calls:
            target = by_name.get(callee_name)
            if target is not None and target != caller:
                edges.append(Edge(kind="CALLS", src_id=caller, dst_id=target))

        for imp in module.imports:
            edges.append(Edge(kind="IMPORTS", src_id=path, dst_id=imp))

        return ParseResult(module=module, functions=functions, classes=classes, edges=edges)


register(GoParser())
