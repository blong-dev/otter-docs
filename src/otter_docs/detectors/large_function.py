"""large_function — flags functions over a configurable line threshold.

Line count is a noisy proxy for "this function is doing too much",
but it's a useful first cut. Threshold defaults to 80 lines, which
catches the long tail without flagging routine 30-50 line functions.

Confidence is moderate (0.6) because long isn't the same as bad —
a parser or state machine may legitimately need the lines. The
recommendation suggests review rather than refactor.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location

DEFAULT_THRESHOLD = 80


class LargeFunctionDetector:
    kind = "large_function"
    cost_tier = "static"

    def __init__(self, threshold: int = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]:
        findings: list[Finding] = []
        for fn in graph.list_functions(repo):
            length = max(0, fn.end_line - fn.line + 1)
            if length < self.threshold:
                continue
            findings.append(Finding(
                kind=self.kind,
                confidence=0.6,
                locations=[Location(
                    repo=repo, path=fn.module_path,
                    line=fn.line, end_line=fn.end_line, guid=fn.guid,
                )],
                evidence={
                    "function_name": fn.name,
                    "lines": length,
                    "threshold": self.threshold,
                },
                recommendation=Recommendation(
                    summary=f"Review `{fn.name}` ({length} lines) for split opportunities",
                    rationale=(
                        f"`{fn.name}` spans {length} lines, above the "
                        f"{self.threshold}-line threshold. Long functions "
                        f"often cluster multiple responsibilities; a review "
                        f"will tell you whether this one is legitimately "
                        f"large (e.g. a state machine) or a refactor candidate."
                    ),
                    blast_radius=[fn.guid],
                ),
                source_detector=self.kind,
            ))
        return findings


register(LargeFunctionDetector())
