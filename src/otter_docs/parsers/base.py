"""Parser Protocol + dispatch.

Each language ships a `LanguageParser` whose only job is to turn a
(repo, path, source_bytes) tuple into records + edges. The base
module also exposes the public `parse_file` dispatcher used by
`Repo.scan()` and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language,
    ModuleRecord,
)


@dataclass
class ParseResult:
    """One file's worth of extracted records + edges."""

    module: ModuleRecord
    functions: list[FunctionRecord] = field(default_factory=list)
    classes: list[ClassRecord] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


class LanguageParser(Protocol):
    """Per-language parser interface.

    `parse` must be pure: same input → same output. No network, no
    LLM calls. Vectors stay None at this stage — they're filled in by
    Phase 3/4 enrichment.
    """

    language: Language

    def parse(self, *, repo: str, path: str, source: bytes) -> ParseResult: ...


_REGISTRY: dict[Language, LanguageParser] = {}


def register(parser: LanguageParser) -> None:
    """Register a parser for its declared language.

    Called once at module import time by each language module. Tests
    can also call this with a stub parser to override dispatch.
    """
    _REGISTRY[parser.language] = parser


def parse_file(
    *, repo: str, path: str, source: bytes, language: Language
) -> ParseResult | None:
    """Dispatch to the registered parser for `language`.

    Returns None for unknown / unregistered languages so the caller
    (typically scan()) can skip the file without erroring.
    """
    parser = _REGISTRY.get(language)
    if parser is None:
        return None
    return parser.parse(repo=repo, path=path, source=source)


# Eager-register built-in parsers. Imports trigger their `register()` calls.
def _bootstrap() -> None:
    # Local imports keep top-level import time low; if a grammar package
    # is missing the user gets a clear error only when they reach the
    # parser they're trying to use.
    from otter_docs.parsers import go as _go  # noqa: F401
    from otter_docs.parsers import python as _py  # noqa: F401
    from otter_docs.parsers import typescript as _ts  # noqa: F401


_bootstrap()
