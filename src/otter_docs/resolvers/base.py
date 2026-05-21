"""Resolver Protocol + registry + dispatch."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from otter_docs.backends.base import GraphBackend
from otter_docs.models import Edge, Language


@dataclass
class ResolveReport:
    """Per-language summary of a resolve() pass.

    edges_emitted is the count *after* the backend deduped — re-running
    over an unchanged repo reports the same total without adding rows.

    `warnings` carries human-readable strings about *missing* resolvers —
    languages the scan saw but no resolver registered for. The actionable
    install hint is included so the user can fix it in one read.
    """

    edges_emitted: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Resolver(Protocol):
    """Per-language cross-file edge resolver.

    `resolve` walks a repo root, asks the language's name-resolver
    where each call site points, and yields Edges. The yielded edges
    carry `src_id` and `dst_id` as function GUIDs when both ends are
    known to the graph; when the call resolves to a symbol the graph
    doesn't track (a stdlib function, a third-party dep), the resolver
    can drop it or emit it with a synthetic dst_id — we default to
    drop for v0.1 since stdlib edges are noise.
    """

    language: Language

    def resolve(
        self,
        *,
        repo: str,
        repo_root: Path,
        graph: GraphBackend,
    ) -> Iterable[Edge]: ...


_registry: dict[Language, Resolver] = {}

# Install hints declared by *every* resolver module at import time —
# even when the module doesn't actually register (because its upstream
# tool is missing). That's the data we need to tell a user "you have
# Python files but no Python resolver; install jedi with X".
_install_hints: dict[Language, str] = {}


def register(resolver: Resolver) -> None:
    """Register a resolver for its declared language."""
    _registry[resolver.language] = resolver


def declare_install_hint(language: Language, hint: str) -> None:
    """Record the install hint for a language's resolver.

    Called by each resolver module at import time, unconditionally — so
    even if the resolver doesn't auto-register (missing tooling), the
    hint is still available for the missing-resolver warning.
    """
    _install_hints[language] = hint


def install_hint_for(language: Language) -> str | None:
    """Return the install hint registered for `language`, or None."""
    return _install_hints.get(language)


def registry() -> dict[Language, Resolver]:
    """Read-only view of registered resolvers."""
    return dict(_registry)


def _quiet_languages() -> set[str]:
    """Languages the user has explicitly opted out of resolving.

    Set via `OTTER_RESOLVER_QUIET` env var as a comma-separated list of
    Language values (e.g. `OTTER_RESOLVER_QUIET=go,typescript`). Names
    are matched case-insensitively against `Language.value`.
    """
    raw = os.environ.get("OTTER_RESOLVER_QUIET", "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def resolve_repo(
    *,
    repo: str,
    repo_root: Path,
    graph: GraphBackend,
    languages: set[Language] | None = None,
    languages_seen: set[Language] | None = None,
) -> dict[Language, ResolveReport]:
    """Run every registered resolver against `repo` and write edges.

    Parameters
    ----------
    repo, repo_root, graph :
        Match the Repo internals.
    languages :
        Optional filter — only run resolvers for these languages. None
        means run every registered resolver.
    languages_seen :
        Optional set of languages present in the scanned graph. When
        provided, `resolve_repo` emits a `warnings` entry on a
        synthesized ResolveReport for each language seen-but-unregistered,
        with an install hint. Silenced per-language via
        `OTTER_RESOLVER_QUIET`.

    Returns
    -------
    Per-language ResolveReport. Languages without a registered resolver
    appear in the map ONLY when `languages_seen` flagged them as missing
    — those entries have `edges_emitted == 0` and a populated `warnings`.
    """
    reports: dict[Language, ResolveReport] = {}
    for lang, resolver in _registry.items():
        if languages is not None and lang not in languages:
            continue
        report = ResolveReport()
        for edge in resolver.resolve(repo=repo, repo_root=repo_root, graph=graph):
            graph._add_edge_with_repo(edge, repo=repo)
            report.edges_emitted += 1
        reports[lang] = report

    if languages_seen:
        quiet = _quiet_languages()
        missing = {
            lang for lang in languages_seen
            if lang not in _registry and lang.value.lower() not in quiet
        }
        for lang in missing:
            hint = _install_hints.get(lang)
            msg = (
                f"no resolver registered for {lang.value} "
                f"(scan saw {lang.value} files); "
                + (f"install: {hint}" if hint else "no install hint available")
                + f". Silence with OTTER_RESOLVER_QUIET={lang.value}."
            )
            reports[lang] = ResolveReport(warnings=[msg])

    return reports
