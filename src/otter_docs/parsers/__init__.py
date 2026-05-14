"""Language-dispatched source parsers.

Each parser turns a single source file into a `ParseResult` of records
(ModuleRecord + FunctionRecord/ClassRecord) and `Edge`s discoverable
from the file's AST alone. Cross-file resolution (the kind stack-graphs
solves) is deferred to a later phase — for v0.1, edges from a parser
are restricted to what's syntactically visible in that one file:

  IMPORTS   — from import statements
  CALLS     — only intra-file (function-to-function in the same file)
  CONTAINS  — module → class → method nesting

The public entry point is `parse_file(path, source, language)`. The
language enum drives dispatch; unknown languages return an empty
ParseResult so scan() can keep going.
"""

from __future__ import annotations

from otter_docs.parsers.base import LanguageParser, ParseResult, parse_file

__all__ = ["LanguageParser", "ParseResult", "parse_file"]
