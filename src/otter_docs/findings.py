"""Finding + Recommendation: the typed product otter-docs emits.

Detectors return `Finding`s. Each Finding may carry an optional
`Recommendation` describing the right next action — that's what the
agent harness reads, ranks, and acts on. The library itself never
modifies code; the harness does.

The schema matches the spec one-for-one:

  Finding
    kind                "redundancy.semantic_equivalence" / "dead_code" / ...
    confidence          0..1
    edge_confidence     0..1 when a graph traversal produced this finding;
                        None for purely-local findings (e.g. dead_code).
                        Lower confidence means stack-graphs (or the
                        v0.1 AST-only edge extractor) encountered
                        ambiguous or missing bindings on the path.
    locations           where in the codebase the finding applies
    evidence            arbitrary detector-specific data
    recommendation      optional next action

A Finding without a Recommendation is information; with one, it's a
proposal. Both shapes are valid.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from otter_docs.models import Location


class Recommendation(BaseModel):
    """Concrete next action the agent can take to address a Finding."""

    model_config = ConfigDict(frozen=True)

    summary: str
    rationale: str
    # `proposed_diff` is a unified-diff string when an LLM-direct tier
    # detector produced one. Static-tier detectors usually leave it None
    # because they don't know enough to write a patch.
    proposed_diff: str | None = None
    # GUIDs (or module paths for modules) that this change would touch.
    blast_radius: list[str] = Field(default_factory=list)
    suggested_tests: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """A single signal emitted by a detector."""

    model_config = ConfigDict(frozen=True)

    kind: str
    confidence: float = Field(ge=0.0, le=1.0)
    edge_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    locations: list[Location]
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommendation: Recommendation | None = None
    # Tracks which detector produced this. Useful for debugging,
    # filtering, and finding-vs-finding deduplication across detectors
    # that overlap (e.g. our dead_code + vulture's dead_code).
    source_detector: str = ""
