"""large_function — flags functions that are long or branchy.

Two metrics, either of which can trip the threshold:

  - Raw line span (`end_line - line + 1`). Default threshold 80.
    Available for every language otter-docs parses.
  - McCabe cyclomatic complexity (`FunctionRecord.cyclomatic_complexity`).
    Default threshold 10 (radon's "C" grade boundary). Today only the
    Python parser populates this; non-Python functions read None and
    only the line-span gate applies.

Confidence (2026-05-21 calibration):
  - Both gates tripped → 0.85 ("long AND branchy" — strong refactor signal)
  - Only one gate tripped → 0.6 (review candidate; might be legitimate)

Long isn't the same as bad — a parser or state machine may legitimately
need the lines AND the branches. The recommendation suggests review,
not refactor, even at the higher confidence.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location

DEFAULT_LINE_THRESHOLD = 80
DEFAULT_CYCLOMATIC_THRESHOLD = 10


class LargeFunctionDetector:
    kind = "large_function"
    cost_tier = "static"

    def __init__(
        self,
        threshold: int = DEFAULT_LINE_THRESHOLD,
        *,
        cyclomatic_threshold: int = DEFAULT_CYCLOMATIC_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.cyclomatic_threshold = cyclomatic_threshold

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]:
        findings: list[Finding] = []
        for fn in graph.list_functions(repo):
            length = max(0, fn.end_line - fn.line + 1)
            cc = fn.cyclomatic_complexity
            long_enough = length >= self.threshold
            branchy_enough = (
                cc is not None and cc >= self.cyclomatic_threshold
            )
            if not (long_enough or branchy_enough):
                continue
            confidence = 0.85 if (long_enough and branchy_enough) else 0.6
            triggered: list[str] = []
            if long_enough:
                triggered.append(f"{length} lines (≥{self.threshold})")
            if branchy_enough:
                triggered.append(
                    f"cyclomatic complexity {cc} (≥{self.cyclomatic_threshold})"
                )
            metric_str = " and ".join(triggered)

            findings.append(Finding(
                kind=self.kind,
                confidence=confidence,
                locations=[Location(
                    repo=repo, path=fn.module_path,
                    line=fn.line, end_line=fn.end_line, guid=fn.guid,
                )],
                evidence={
                    "function_name": fn.name,
                    "lines": length,
                    "threshold": self.threshold,
                    "cyclomatic_complexity": cc,
                    "cyclomatic_threshold": self.cyclomatic_threshold,
                },
                recommendation=Recommendation(
                    summary=(
                        f"Review `{fn.name}` ({metric_str})"
                    ),
                    rationale=(
                        f"`{fn.name}` is {metric_str}. Long or branchy "
                        "functions often cluster multiple responsibilities; "
                        "a review will tell you whether this one is "
                        "legitimately large (e.g. a state machine or "
                        "parser) or a refactor candidate."
                    ),
                    blast_radius=[fn.guid],
                ),
                source_detector=self.kind,
            ))
        return findings


register(LargeFunctionDetector())
