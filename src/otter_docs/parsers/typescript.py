"""TypeScript / TSX source parser via tree-sitter.

Extracts:
  - ModuleRecord (one per file)
  - FunctionRecord for `function`, arrow functions assigned to const,
    and class methods
  - ClassRecord for each `class X { ... }`
  - IMPORTS edges from `import ... from "..."` and `import "..."`
  - Intra-file CALLS edges by callee identifier match

We register the same parser instance under both PYTHON / GO / TS path
dispatchers via the .ts/.tsx file extensions handled in scan(). The
extension's TypeScript-vs-TSX choice is decided by the dispatcher.
"""

from __future__ import annotations

import hashlib

import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    ModuleRecord,
)
from otter_docs.models import (
    Language as Lang,
)
from otter_docs.parsers.base import ParseResult, register

_LANGUAGE_TS = Language(tree_sitter_typescript.language_typescript())
_LANGUAGE_TSX = Language(tree_sitter_typescript.language_tsx())
_PARSER_TS = Parser(_LANGUAGE_TS)
_PARSER_TSX = Parser(_LANGUAGE_TSX)


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
        if child.type == "import_statement":
            src = child.child_by_field_name("source")
            if src is not None:
                out.append(src.text.decode("utf-8", errors="replace").strip("'\""))
    return out


def _args_from_params(params: Node | None) -> list[str]:
    if params is None:
        return []
    out: list[str] = []
    for p in params.named_children:
        # Various: required_parameter, optional_parameter, rest_pattern
        n = p.child_by_field_name("pattern") or p.child_by_field_name("name")
        if n is None and p.type == "identifier":
            n = p
        if n is None:
            for c in p.named_children:
                if c.type == "identifier":
                    n = c
                    break
        if n is not None:
            out.append(n.text.decode("utf-8", errors="replace"))
    return out


def _collect_calls(node: Node, sink: list[str]) -> None:
    for c in node.named_children:
        if c.type == "call_expression":
            func = c.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    sink.append(func.text.decode("utf-8", errors="replace"))
                elif func.type == "member_expression":
                    prop = func.child_by_field_name("property")
                    if prop is not None:
                        sink.append(prop.text.decode("utf-8", errors="replace"))
        _collect_calls(c, sink)


class _TSParserBase:
    def __init__(self, language: Lang, parser: Parser) -> None:
        self.language = language
        self._parser = parser

    def parse(self, *, repo: str, path: str, source: bytes) -> ParseResult:
        tree = self._parser.parse(source)
        root = tree.root_node
        module = ModuleRecord(
            repo=repo, path=path, language=self.language, imports=_imports(root)
        )

        functions: list[FunctionRecord] = []
        classes: list[ClassRecord] = []
        edges: list[Edge] = []
        pending_calls: list[tuple[str, str]] = []

        def emit_fn(name: str, node: Node, params: Node | None) -> str:
            line = node.start_point.row + 1
            end = node.end_point.row + 1
            guid = _guid(repo, path, name, line)
            is_async = any(
                c.type == "async" or (c.is_named and c.type == "async") for c in node.children
            )
            functions.append(FunctionRecord(
                repo=repo, guid=guid, name=name, module_path=path,
                line=line, end_line=end, args=_args_from_params(params),
                is_async=is_async,
            ))
            return guid

        def walk(node: Node) -> None:
            t = node.type
            if t == "function_declaration":
                fn_name = _name(node) or "<anonymous>"
                guid = emit_fn(fn_name, node, node.child_by_field_name("parameters"))
                body = node.child_by_field_name("body")
                if body is not None:
                    names: list[str] = []
                    _collect_calls(body, names)
                    for n in names:
                        pending_calls.append((guid, n))
                return
            if t == "lexical_declaration" or t == "variable_declaration":
                # const foo = (x) => …  /  const bar = function (x) {…}
                for decl in node.named_children:
                    if decl.type != "variable_declarator":
                        continue
                    name_node = decl.child_by_field_name("name")
                    value = decl.child_by_field_name("value")
                    if name_node is None or value is None:
                        continue
                    if value.type in ("arrow_function", "function_expression"):
                        guid = emit_fn(
                            name_node.text.decode("utf-8", errors="replace"),
                            value,
                            value.child_by_field_name("parameters"),
                        )
                        body = value.child_by_field_name("body")
                        if body is not None:
                            names = []
                            _collect_calls(body, names)
                            for n in names:
                                pending_calls.append((guid, n))
                return
            if t == "class_declaration":
                cls_name = _name(node) or "<anonymous>"
                line = node.start_point.row + 1
                end = node.end_point.row + 1
                cls_guid = _guid(repo, path, f"class:{cls_name}", line)
                classes.append(ClassRecord(
                    repo=repo, guid=cls_guid, name=cls_name, module_path=path,
                    line=line, end_line=end,
                ))
                body = node.child_by_field_name("body")
                if body is not None:
                    for member in body.named_children:
                        if member.type == "method_definition":
                            m_name = _name(member) or "<anonymous>"
                            qualified = f"{cls_name}.{m_name}"
                            guid = emit_fn(
                                qualified,
                                member,
                                member.child_by_field_name("parameters"),
                            )
                            edges.append(
                                Edge(kind="CONTAINS", src_id=cls_guid, dst_id=guid)
                            )
                            m_body = member.child_by_field_name("body")
                            if m_body is not None:
                                names = []
                                _collect_calls(m_body, names)
                                for n in names:
                                    pending_calls.append((guid, n))
                return
            # Recurse so we catch top-level `export` wrappers etc.
            for c in node.named_children:
                walk(c)

        for child in root.named_children:
            walk(child)

        by_name: dict[str, str] = {}
        for fn in functions:
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


register(_TSParserBase(Lang.TYPESCRIPT, _PARSER_TS))
register(_TSParserBase(Lang.JAVASCRIPT, _PARSER_TS))


# Exposed for scan() to pick the TSX parser when the extension is .tsx.
TSX_PARSER = _TSParserBase(Lang.TYPESCRIPT, _PARSER_TSX)
