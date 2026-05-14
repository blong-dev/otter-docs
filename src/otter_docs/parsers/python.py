"""Python source parser via tree-sitter.

Extracts:
  - ModuleRecord (one per file, with docstring + imports)
  - FunctionRecord for every `def` / `async def` at any nesting depth
  - ClassRecord for every `class`
  - Intra-file CALLS edges (name-based; cross-file resolution deferred)
  - IMPORTS edges from `import` / `from X import Y` statements
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language as Lang,
    ModuleRecord,
)
from otter_docs.parsers.base import LanguageParser, ParseResult, register


_LANGUAGE = Language(tree_sitter_python.language())
_PARSER = Parser(_LANGUAGE)


def _guid(repo: str, path: str, name: str, line: int) -> str:
    """Stable identifier for a function/class node.

    Hashing (repo, path, name, line) gives us a deterministic id we
    can rebuild on every scan without storing extra state. Callers
    that need a fancier scheme (e.g. canonicalized AST hashes) can
    plug into Phase 3 enrichment later.
    """
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{repo}|{path}|{name}|{line}".encode())
    return h.hexdigest()


def _docstring(body: Node | None) -> str:
    """Pull the leading string-expression out of a function/class body."""
    if body is None:
        return ""
    for child in body.named_children:
        if child.type == "expression_statement":
            inner = child.named_children[0] if child.named_children else None
            if inner is not None and inner.type == "string":
                # Strip surrounding triple-quotes / quotes.
                raw = inner.text.decode("utf-8", errors="replace")
                return _strip_pystring(raw)
        # First non-string statement aborts the search.
        break
    return ""


def _strip_pystring(raw: str) -> str:
    for q in ('"""', "'''", '"', "'"):
        if raw.startswith(q) and raw.endswith(q):
            return raw[len(q) : -len(q)]
    return raw


def _name(node: Node, field: str = "name") -> str:
    target = node.child_by_field_name(field)
    return target.text.decode("utf-8", errors="replace") if target else ""


@dataclass
class _Walker:
    repo: str
    path: str
    functions: list[FunctionRecord]
    classes: list[ClassRecord]
    edges: list[Edge]
    # call_targets is the set of intra-file CALLS edges to emit. Keyed
    # by (caller_guid, callee_name); we resolve the callee_name to a
    # guid in a second pass once all functions are known.
    pending_calls: list[tuple[str, str]]

    def visit_function(self, node: Node, parent_class_guid: str | None) -> None:
        name = _name(node)
        line = node.start_point.row + 1
        end = node.end_point.row + 1
        guid = _guid(self.repo, self.path, name, line)
        body = node.child_by_field_name("body")
        is_async = node.type == "function_definition" and any(
            c.type == "async" for c in node.children
        )
        # async def is its own grammar node in tree-sitter-python.
        if not is_async and node.parent and node.parent.type == "decorated_definition":
            pass  # decorators don't affect async-ness
        is_async = is_async or node.type == "function_definition" and node.children and node.children[0].type == "async"
        args = self._args(node)
        record = FunctionRecord(
            repo=self.repo,
            guid=guid,
            name=name,
            module_path=self.path,
            line=line,
            end_line=end,
            docstring=_docstring(body),
            args=args,
            is_async=is_async,
        )
        self.functions.append(record)
        if parent_class_guid is not None:
            self.edges.append(Edge(kind="CONTAINS", src_id=parent_class_guid, dst_id=guid))
        # Walk body for nested defs/classes and call expressions.
        if body is not None:
            self._scan_body(body, current_function_guid=guid, parent_class_guid=None)

    def _args(self, fn_node: Node) -> list[str]:
        params = fn_node.child_by_field_name("parameters")
        if params is None:
            return []
        out: list[str] = []
        for p in params.named_children:
            # Various parameter node types: identifier, typed_parameter,
            # default_parameter, list_splat_pattern, etc.
            n = p.child_by_field_name("name")
            if n is None and p.type == "identifier":
                n = p
            if n is None:
                # Splats and complex patterns — fall back to first identifier child.
                for c in p.named_children:
                    if c.type == "identifier":
                        n = c
                        break
            if n is not None:
                out.append(n.text.decode("utf-8", errors="replace"))
        return out

    def visit_class(self, node: Node) -> None:
        name = _name(node)
        line = node.start_point.row + 1
        end = node.end_point.row + 1
        guid = _guid(self.repo, self.path, f"class:{name}", line)
        body = node.child_by_field_name("body")
        record = ClassRecord(
            repo=self.repo,
            guid=guid,
            name=name,
            module_path=self.path,
            line=line,
            end_line=end,
            docstring=_docstring(body),
        )
        self.classes.append(record)
        if body is not None:
            self._scan_body(body, current_function_guid=None, parent_class_guid=guid)

    def _scan_body(
        self,
        body: Node,
        *,
        current_function_guid: str | None,
        parent_class_guid: str | None,
    ) -> None:
        for child in body.named_children:
            self._visit(
                child,
                current_function_guid=current_function_guid,
                parent_class_guid=parent_class_guid,
            )

    def _visit(
        self,
        node: Node,
        *,
        current_function_guid: str | None,
        parent_class_guid: str | None,
    ) -> None:
        if node.type == "function_definition":
            self.visit_function(node, parent_class_guid=parent_class_guid)
            return
        if node.type == "class_definition":
            self.visit_class(node)
            return
        if node.type == "decorated_definition":
            # The wrapped def/class is in the `definition` field.
            inner = node.child_by_field_name("definition")
            if inner is not None:
                self._visit(
                    inner,
                    current_function_guid=current_function_guid,
                    parent_class_guid=parent_class_guid,
                )
            return
        if node.type == "call" and current_function_guid is not None:
            callee_name = self._call_target_name(node)
            if callee_name:
                self.pending_calls.append((current_function_guid, callee_name))
        # Recurse so we catch nested calls inside `if`/`for`/`with`/etc.
        for c in node.named_children:
            self._visit(
                c,
                current_function_guid=current_function_guid,
                parent_class_guid=parent_class_guid,
            )

    def _call_target_name(self, call_node: Node) -> str:
        func = call_node.child_by_field_name("function")
        if func is None:
            return ""
        if func.type == "identifier":
            return func.text.decode("utf-8", errors="replace")
        if func.type == "attribute":
            # foo.bar() — use the rightmost attribute name.
            attr = func.child_by_field_name("attribute")
            if attr is not None:
                return attr.text.decode("utf-8", errors="replace")
        return ""


def _module_docstring(root: Node) -> str:
    """Module docstring = first top-level string expression."""
    for child in root.named_children:
        if child.type == "expression_statement":
            inner = child.named_children[0] if child.named_children else None
            if inner is not None and inner.type == "string":
                return _strip_pystring(inner.text.decode("utf-8", errors="replace"))
        break
    return ""


def _imports(root: Node) -> list[str]:
    """Flat list of imported module names."""
    out: list[str] = []
    for child in root.named_children:
        if child.type == "import_statement":
            for c in child.named_children:
                if c.type == "dotted_name":
                    out.append(c.text.decode("utf-8", errors="replace"))
                elif c.type == "aliased_import":
                    name = c.child_by_field_name("name")
                    if name is not None:
                        out.append(name.text.decode("utf-8", errors="replace"))
        elif child.type == "import_from_statement":
            module = child.child_by_field_name("module_name")
            if module is not None:
                out.append(module.text.decode("utf-8", errors="replace"))
    return out


class PythonParser:
    language = Lang.PYTHON

    def parse(self, *, repo: str, path: str, source: bytes) -> ParseResult:
        tree = _PARSER.parse(source)
        root = tree.root_node
        module = ModuleRecord(
            repo=repo,
            path=path,
            language=Lang.PYTHON,
            docstring=_module_docstring(root),
            imports=_imports(root),
        )
        walker = _Walker(
            repo=repo, path=path, functions=[], classes=[], edges=[], pending_calls=[]
        )
        for child in root.named_children:
            walker._visit(child, current_function_guid=None, parent_class_guid=None)

        # Resolve intra-file calls: callee name → guid (first match wins;
        # ambiguous names just drop their edge for now — we'll lean on
        # stack-graphs later for proper resolution).
        by_name: dict[str, str] = {}
        for fn in walker.functions:
            by_name.setdefault(fn.name, fn.guid)
        for caller_guid, callee_name in walker.pending_calls:
            callee_guid = by_name.get(callee_name)
            if callee_guid is not None and callee_guid != caller_guid:
                walker.edges.append(
                    Edge(kind="CALLS", src_id=caller_guid, dst_id=callee_guid)
                )

        # IMPORTS edges are emitted with dst_id = imported module name.
        # They're cross-file by nature and stay as string-ids; the
        # backend doesn't require dst_id to refer to a known node.
        module_id = path  # modules are keyed by (repo, path)
        for imp in module.imports:
            walker.edges.append(Edge(kind="IMPORTS", src_id=module_id, dst_id=imp))

        return ParseResult(
            module=module,
            functions=walker.functions,
            classes=walker.classes,
            edges=walker.edges,
        )


register(PythonParser())
