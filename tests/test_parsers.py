"""Parser-level tests — direct AST extraction without going through Repo.scan().

These exercise the language-specific parsers in isolation so regressions
can be localized to a single grammar.
"""

from __future__ import annotations

import pytest

from otter_docs.models import Language
from otter_docs.parsers import parse_file
from otter_docs.parsers.typescript import TSX_PARSER

# ── Python ──────────────────────────────────────────────────────────────


def test_python_module_doc_and_imports():
    src = b'"""mod doc"""\nimport os\nfrom pathlib import Path\n'
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.module.docstring == "mod doc"
    assert r.module.imports == ["os", "pathlib"]


def test_python_function_extraction():
    src = b"def hello(x, y=1):\n    return x + y\n"
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert len(r.functions) == 1
    fn = r.functions[0]
    assert fn.name == "hello"
    assert fn.line == 1
    assert fn.args == ["x", "y"]
    assert fn.is_async is False


def test_python_async_function():
    src = b"async def fetch():\n    pass\n"
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert r.functions[0].is_async is True


def test_python_class_with_method_and_contains_edge():
    src = b'class Foo:\n    """class doc"""\n    def bar(self):\n        return 1\n'
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert len(r.classes) == 1
    assert r.classes[0].name == "Foo"
    assert r.classes[0].docstring == "class doc"
    assert any(e.kind == "CONTAINS" for e in r.edges)


def test_python_intra_file_call_edge():
    src = b"def helper(): return 1\n\ndef caller(): return helper()\n"
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    call_edges = [e for e in r.edges if e.kind == "CALLS"]
    assert len(call_edges) == 1
    caller = next(f for f in r.functions if f.name == "caller")
    helper = next(f for f in r.functions if f.name == "helper")
    assert call_edges[0].src_id == caller.guid
    assert call_edges[0].dst_id == helper.guid


def test_python_decorated_function_is_extracted():
    src = b"@staticmethod\ndef wrapped(): pass\n"
    r = parse_file(repo="r", path="m.py", source=src, language=Language.PYTHON)
    assert any(f.name == "wrapped" for f in r.functions)


# ── Go ──────────────────────────────────────────────────────────────────


def test_go_function_and_method():
    src = b'package x\n\nfunc Plain() int { return 1 }\nfunc (s *S) Method() int { return 2 }\n'
    r = parse_file(repo="r", path="m.go", source=src, language=Language.GO)
    names = sorted(f.name for f in r.functions)
    assert names == ["Plain", "S.Method"]


def test_go_imports():
    src = b'package x\n\nimport (\n    "fmt"\n    "os"\n)\n'
    r = parse_file(repo="r", path="m.go", source=src, language=Language.GO)
    assert r.module.imports == ["fmt", "os"]


def test_go_type_becomes_class():
    src = b'package x\n\ntype Server struct { addr string }\n'
    r = parse_file(repo="r", path="m.go", source=src, language=Language.GO)
    assert len(r.classes) == 1
    assert r.classes[0].name == "Server"


# ── TypeScript ──────────────────────────────────────────────────────────


def test_typescript_function_and_arrow():
    src = (
        b"export function plain(x: number): number { return x; }\n"
        b"const arrow = (y: number) => y * 2;\n"
    )
    r = parse_file(repo="r", path="m.ts", source=src, language=Language.TYPESCRIPT)
    names = sorted(f.name for f in r.functions)
    assert names == ["arrow", "plain"]


def test_typescript_class_methods_get_qualified_names():
    src = b"class Counter { increment() { return 1; } async go() {} }\n"
    r = parse_file(repo="r", path="m.ts", source=src, language=Language.TYPESCRIPT)
    assert "Counter" in [c.name for c in r.classes]
    method_names = {f.name for f in r.functions}
    assert "Counter.increment" in method_names
    assert "Counter.go" in method_names
    go_method = next(f for f in r.functions if f.name == "Counter.go")
    assert go_method.is_async is True


def test_typescript_imports():
    src = b'import {x} from "lib";\nimport "./side.css";\n'
    r = parse_file(repo="r", path="m.ts", source=src, language=Language.TYPESCRIPT)
    assert r.module.imports == ["lib", "./side.css"]


def test_tsx_parser_handles_jsx():
    src = b'export const App = () => <div>hello</div>;\n'
    r = TSX_PARSER.parse(repo="r", path="App.tsx", source=src)
    assert any(f.name == "App" for f in r.functions)


# ── Dispatch ────────────────────────────────────────────────────────────


def test_unknown_language_returns_none():
    r = parse_file(repo="r", path="m.x", source=b"", language=Language.UNKNOWN)
    assert r is None


@pytest.mark.parametrize("lang", [Language.PYTHON, Language.GO, Language.TYPESCRIPT])
def test_empty_source_yields_empty_records(lang: Language):
    r = parse_file(repo="r", path="empty", source=b"", language=lang)
    assert r is not None
    assert r.module.path == "empty"
    assert r.functions == []
    assert r.classes == []
