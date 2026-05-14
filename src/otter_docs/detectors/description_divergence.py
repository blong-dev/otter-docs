"""description.divergence — when intent and implementation drift apart.

A function's description vector should look like its code vector if
the description faithfully summarizes what the code does. When they
diverge — cosine similarity below a threshold — one of two things
is true:

  (a) The docstring is stale; the code has evolved past it.
  (b) The function does multiple things and the description only
      captures one of them.

Both are worth surfacing. We don't claim to know which one without an
LLM-direct follow-up (that's Phase 7). Output: a Finding with both
vectors' similarity in evidence and a recommendation to read the
function and either rewrite the docstring or split the function.

Cost-tier: embedding. Free at query time once vectors exist.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location


DEFAULT_THRESHOLD = 0.4
# When cosine similarity between description and code is below this,
# we emit a Finding. Empirically (on nomic-embed-text) functions whose
# descriptions match their code sit above ~0.65; below 0.4 there's
# usually something real going on. The default errs on the side of
# false negatives.


class DescriptionDivergenceDetector:
    kind = "description.divergence"
    cost_tier = "embedding"

    def __init__(self, *, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]:
        findings: list[Finding] = []
        for fn in graph.list_functions(repo):
            if fn.description_vec is None or fn.code_vec is None:
                continue
            sim = _cosine(fn.description_vec, fn.code_vec)
            if sim >= self.threshold:
                continue
            # We emit "confidence" as (threshold - sim)/threshold — i.e.
            # how far below the line this function is. A vec right at
            # 0.0 gets confidence ~1.0; one at threshold-epsilon gets
            # ~0.0. Clamps within [0, 1].
            confidence = max(0.0, min(1.0, (self.threshold - sim) / self.threshold))
            findings.append(Finding(
                kind=self.kind,
                confidence=confidence,
                locations=[Location(
                    repo=repo, path=fn.module_path,
                    line=fn.line, end_line=fn.end_line, guid=fn.guid,
                )],
                evidence={
                    "function_name": fn.name,
                    "description_code_similarity": sim,
                    "threshold": self.threshold,
                },
                recommendation=Recommendation(
                    summary=(
                        f"Review `{fn.name}` — description and code diverge"
                    ),
                    rationale=(
                        f"`{fn.name}`'s description vector and code vector "
                        f"have cosine similarity {sim:.2f}, below the "
                        f"{self.threshold:.2f} threshold. Possible causes: "
                        f"the docstring is stale, or the function is doing "
                        f"more than its description claims. Read both and "
                        f"either update the docstring or split the function."
                    ),
                    blast_radius=[fn.guid],
                ),
                source_detector=self.kind,
            ))
        return findings


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


register(DescriptionDivergenceDetector())
