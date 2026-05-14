"""Detector Protocol + registry."""

from __future__ import annotations

from typing import Literal, Protocol

from otter_docs.backends.base import GraphBackend
from otter_docs.findings import Finding


CostTier = Literal["static", "embedding", "llm_direct"]


class Detector(Protocol):
    """Detectors emit Findings against an indexed Repo's graph.

    Attributes
    ----------
    kind :
        Canonical finding kind this detector emits (e.g. "dead_code",
        "redundancy.semantic_equivalence"). Used for filtering at the
        `findings()` call site.
    cost_tier :
        How expensive this detector is. Callers can opt out of the
        expensive ones when running on a tight budget.

    Methods
    -------
    run(repo, graph) :
        Return a list of Findings. Must not mutate the graph.
    """

    kind: str
    cost_tier: CostTier

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]: ...


_registry: dict[str, Detector] = {}


def register(detector: Detector) -> None:
    """Add a detector to the global registry, keyed by its `kind`."""
    _registry[detector.kind] = detector


def registry() -> dict[str, Detector]:
    """Read-only view of registered detectors."""
    return dict(_registry)


def run_all(
    repo: str,
    graph: GraphBackend,
    *,
    kinds: set[str] | None = None,
    cost_tiers: set[CostTier] | None = None,
) -> list[Finding]:
    """Run all registered detectors and return their Findings concatenated.

    Filters:
      kinds       run only detectors whose `kind` is in this set
      cost_tiers  run only detectors whose `cost_tier` is in this set

    With no filters, runs every registered detector.
    """
    out: list[Finding] = []
    for det in _registry.values():
        if kinds is not None and det.kind not in kinds:
            continue
        if cost_tiers is not None and det.cost_tier not in cost_tiers:
            continue
        out.extend(det.run(repo, graph))
    return out
