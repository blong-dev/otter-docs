"""Cross-file name resolution.

Resolvers fill in the edge we can't see from a single-file AST: the
cross-file CALLS edge. Each resolver is language-specific and uses a
mature upstream solver (jedi for Python, gopls for Go, tsserver for
TypeScript) — better-than-stack-graphs accuracy on the languages we
support, at the cost of needing the upstream tools installed.

Repo.resolve() dispatches by Language; languages without a registered
resolver are skipped silently so a polyglot repo still gets partial
coverage.

For v0.1 we ship JediResolver. Go and TS land in 2.4.5.
"""

from __future__ import annotations

from otter_docs.resolvers.base import Resolver, register, registry, resolve_repo

__all__ = ["Resolver", "register", "registry", "resolve_repo"]


def _bootstrap() -> None:
    # Each resolver module auto-registers if its dependencies are
    # available (jedi importable, typescript-language-server on PATH,
    # gopls on PATH). Missing tooling = silent skip; users opt in by
    # installing the tool.
    from otter_docs.resolvers import go as _go  # noqa: F401
    from otter_docs.resolvers import python as _py  # noqa: F401
    from otter_docs.resolvers import typescript as _ts  # noqa: F401


_bootstrap()
