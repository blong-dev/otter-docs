"""Rust source parser via tree-sitter.

Extracts:
  - ModuleRecord (one per .rs file)
  - FunctionRecord for `fn X(...)` and `impl T { fn X(...) }` methods
  - ClassRecord for each `struct`, `enum`, and `trait` declaration
  - IMPORTS edges from `use ...;` declarations (module path)
  - Intra-file CALLS edges by callee identifier match

Impl-block methods carry the impl type in their name as "T.X" so
downstream tools can group method sets without an extra edge type,
mirroring how Go's receiver methods are namespaced.
"""

from __future__ import annotations

import hashlib

import tree_sitter_rust
from tree_sitter import Language, Node, Parser

from otter_docs.guids import resolve_guid
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

_LANGUAGE = Language(tree_sitter_rust.language())
_PARSER = Parser(_LANGUAGE)


def _guid(repo: str, path: str, name: str, line: int) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{repo}|{path}|{name}|{line}".encode())
    return h.hexdigest()


def _name(node: Node, field: str = "name") -> str:
    n = node.child_by_field_name(field)
    return n.text.decode("utf-8", errors="replace") if n else ""


def _imports(root: Node) -> list[str]:
    """`use foo::bar::baz;` → "foo::bar::baz" (the leaf path)."""
    out: list[str] = []
    for child in root.named_children:
        if child.type != "use_declaration":
            continue
        # Take the path child as a flat string; brace lists yield the
        # parent path. Good enough for graph IMPORTS edges.
        for c in child.named_children:
            if c.type in (
                "scoped_identifier",
                "identifier",
                "use_wildcard",
                "use_list",
                "use_as_clause",
                "scoped_use_list",
            ):
                txt = c.text.decode("utf-8", errors="replace").strip()
                if txt:
                    out.append(txt)
                break
    return out


def _args(fn_node: Node) -> list[str]:
    """Extract parameter names from a function's `parameters` field.

    Rust parameters can be `&self`, `mut x: T`, patterns, etc. We
    surface a best-effort identifier list — the leftmost identifier
    inside each `parameter` child.
    """
    params = fn_node.child_by_field_name("parameters")
    if params is None:
        return []
    out: list[str] = []
    for p in params.named_children:
        if p.type == "self_parameter":
            out.append("self")
            continue
        if p.type != "parameter":
            continue
        pat = p.child_by_field_name("pattern")
        if pat is None:
            continue
        # `pat` may be an identifier or a richer pattern; either way
        # the first identifier descendant is the binding name.
        ident = _first_identifier(pat)
        if ident:
            out.append(ident)
    return out


def _first_identifier(node: Node) -> str:
    if node.type == "identifier":
        return node.text.decode("utf-8", errors="replace")
    for c in node.named_children:
        got = _first_identifier(c)
        if got:
            return got
    return ""


def _collect_calls(body: Node | None, sink: list[str]) -> None:
    """Walk a function body and append callee identifier names."""
    if body is None:
        return
    for c in body.named_children:
        if c.type == "call_expression":
            func = c.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    sink.append(func.text.decode("utf-8", errors="replace"))
                elif func.type == "scoped_identifier":
                    name = func.child_by_field_name("name")
                    if name is not None:
                        sink.append(name.text.decode("utf-8", errors="replace"))
                elif func.type == "field_expression":
                    field = func.child_by_field_name("field")
                    if field is not None:
                        sink.append(field.text.decode("utf-8", errors="replace"))
        _collect_calls(c, sink)


def _impl_type_name(impl_node: Node) -> str:
    """`impl Foo { ... }` or `impl Trait for Foo { ... }` — return "Foo"."""
    type_node = impl_node.child_by_field_name("type")
    if type_node is None:
        return ""
    return type_node.text.decode("utf-8", errors="replace")


class RustParser:
    language = Lang.RUST

    def parse(self, *, repo: str, path: str, source: bytes) -> ParseResult:
        tree = _PARSER.parse(source)
        root = tree.root_node
        module = ModuleRecord(
            repo=repo, path=path, language=Lang.RUST, imports=_imports(root)
        )

        functions: list[FunctionRecord] = []
        classes: list[ClassRecord] = []
        edges: list[Edge] = []
        pending_calls: list[tuple[str, str]] = []

        def emit_fn(node: Node, name: str, qualified: str | None = None) -> None:
            qname = qualified or name
            line = node.start_point.row + 1
            end = node.end_point.row + 1
            guid = resolve_guid(
                source, node.start_point.row,
                _guid(repo, path, qname, line),
            )
            functions.append(FunctionRecord(
                repo=repo, guid=guid, name=qname, module_path=path,
                line=line, end_line=end, args=_args(node),
            ))
            body = node.child_by_field_name("body")
            if body is not None:
                call_names: list[str] = []
                _collect_calls(body, call_names)
                for cn in call_names:
                    pending_calls.append((guid, cn))

        def emit_class(node: Node, name: str, line: int, end: int) -> None:
            guid = resolve_guid(
                source, node.start_point.row,
                _guid(repo, path, f"class:{name}", line),
            )
            classes.append(ClassRecord(
                repo=repo, guid=guid, name=name, module_path=path,
                line=line, end_line=end,
            ))

        for child in root.named_children:
            t = child.type
            if t == "function_item":
                emit_fn(child, _name(child))
            elif t == "impl_item":
                impl_type = _impl_type_name(child)
                body = child.child_by_field_name("body")
                if body is None:
                    continue
                for member in body.named_children:
                    if member.type == "function_item":
                        method_name = _name(member)
                        qualified = (
                            f"{impl_type}.{method_name}" if impl_type else method_name
                        )
                        emit_fn(member, method_name, qualified=qualified)
            elif t in ("struct_item", "enum_item", "trait_item", "union_item"):
                name = _name(child)
                if name:
                    emit_class(
                        child, name,
                        child.start_point.row + 1,
                        child.end_point.row + 1,
                    )
                if t == "trait_item":
                    # Trait method signatures register as functions too —
                    # they describe the contract surface.
                    body = child.child_by_field_name("body")
                    if body is not None:
                        for member in body.named_children:
                            if member.type in (
                                "function_item", "function_signature_item",
                            ):
                                method_name = _name(member)
                                qualified = (
                                    f"{name}.{method_name}" if name else method_name
                                )
                                emit_fn(member, method_name, qualified=qualified)

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

        return ParseResult(
            module=module, functions=functions, classes=classes, edges=edges,
        )


register(RustParser())
