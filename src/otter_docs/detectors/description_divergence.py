"""description.divergence — when intent and implementation drift apart.

A function's description vector should look like its code vector if
the description faithfully summarizes what the code does. When they
diverge — cosine similarity below a threshold — one of two things
is true:

  (a) The docstring is stale; the code has evolved past it.
  (b) The function does multiple things and the description only
      captures one of them.

**Known limitation (2026-05-21): cosine similarity alone is too noisy
to ship enabled-by-default.** Empirically (v3, 6423 enriched
functions on nomic-embed-text) the signal conflates real divergence
with "terse code + verbose description": a Go one-liner with a
*perfectly accurate* docstring scores the same ~0.53 as one with a
stale docstring. Same number, opposite meaning, undetectable by
distance. The right fix is an LLM judge (planned `confirm_description`,
v0.2) that reads both and rules accurate / partial / stale / wrong —
the same shape as `confirm_redundancy`.

Until then this detector is **cost_tier=`llm_direct`**, so it does NOT
run on a default `findings()` call (only the static + embedding tiers
do). Request it explicitly with `findings(kinds={"description.
divergence"})` or `cost_tiers={"llm_direct"}` if you want the raw
distance signal anyway.
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
    # llm_direct → excluded from the default findings() tier set so the
    # noisy raw-distance signal doesn't surface until the LLM-judge
    # version lands. See module docstring.
    cost_tier = "llm_direct"

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
