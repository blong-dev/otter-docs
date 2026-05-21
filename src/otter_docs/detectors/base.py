"""Detector Protocol + registry."""

from __future__ import annotations

from collections.abc import Iterator
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


# Tiers that run on a default (unfiltered) findings() call. `llm_direct`
# is excluded: those detectors only produce meaningful output with an
# LLM follow-up pass, so they shouldn't fire on a free scan. Request
# them explicitly via `cost_tiers={"llm_direct"}` (or include it in the
# set). See `description.divergence` for the motivating case — cosine
# divergence alone is too noisy without an LLM judge interpreting it.
_DEFAULT_COST_TIERS: frozenset[CostTier] = frozenset({"static", "embedding"})


def _filtered_detectors(
    kinds: set[str] | None,
    cost_tiers: set[CostTier] | None,
) -> Iterator[Detector]:
    # When the caller names specific kinds, honor exactly those (no
    # tier default) — explicit kind selection is opt-in. Otherwise the
    # tier filter defaults to the cheap tiers so llm_direct stays
    # opt-in.
    effective_tiers = cost_tiers
    if effective_tiers is None and kinds is None:
        effective_tiers = _DEFAULT_COST_TIERS
    for det in _registry.values():
        if kinds is not None and det.kind not in kinds:
            continue
        if effective_tiers is not None and det.cost_tier not in effective_tiers:
            continue
        yield det


def _stream_one(det: Detector, repo: str, graph: GraphBackend) -> Iterator[Finding]:
    """Yield findings from one detector.

    Backward-compatible streaming hook: detectors that opt into true
    incremental emission expose `run_stream(repo, graph) -> Iterator`;
    everything else falls back to materializing via `run()` and yielding
    from the list. Either way the caller sees one finding at a time.
    """
    runner = getattr(det, "run_stream", None)
    if runner is not None:
        yield from runner(repo, graph)
    else:
        yield from det.run(repo, graph)


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
    return list(run_all_stream(repo, graph, kinds=kinds, cost_tiers=cost_tiers))


def run_all_stream(
    repo: str,
    graph: GraphBackend,
    *,
    kinds: set[str] | None = None,
    cost_tiers: set[CostTier] | None = None,
) -> Iterator[Finding]:
    """Generator form of `run_all` — yield findings as detectors produce them.

    Each detector still runs sequentially (we don't parallelize across
    detectors; that's caller's job if they want it), but findings flow
    out detector-by-detector. Consumers wiring otter-docs to a message
    bus can publish each finding the moment it's available instead of
    waiting for every detector to finish.

    Same filter semantics as `run_all`.
    """
    for det in _filtered_detectors(kinds, cost_tiers):
        yield from _stream_one(det, repo, graph)
