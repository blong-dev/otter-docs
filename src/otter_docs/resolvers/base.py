"""Resolver Protocol + registry + dispatch."""

from __future__ import annotations

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
    """

    edges_emitted: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


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


def register(resolver: Resolver) -> None:
    """Register a resolver for its declared language."""
    _registry[resolver.language] = resolver


def registry() -> dict[Language, Resolver]:
    """Read-only view of registered resolvers."""
    return dict(_registry)


def resolve_repo(
    *,
    repo: str,
    repo_root: Path,
    graph: GraphBackend,
    languages: set[Language] | None = None,
) -> dict[Language, ResolveReport]:
    """Run every registered resolver against `repo` and write edges.

    Parameters
    ----------
    repo, repo_root, graph :
        Match the Repo internals.
    languages :
        Optional filter — only run resolvers for these languages. None
        means run every registered resolver.

    Returns
    -------
    Per-language ResolveReport. Languages without a registered resolver
    don't appear in the map.
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
    return reports
