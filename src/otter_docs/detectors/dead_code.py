"""dead_code — functions that no one calls inside this repo.

Heuristic: a function is dead if `callers_of(repo, guid)` returns
empty. This is precise for the AST-only edge extractor we ship today
(intra-file CALLS), which means it's actually a "no in-file caller"
signal — a function called only from another file will look dead.

We compensate by:
  - emitting at low confidence (0.5) until stack-graph-equivalent
    cross-file resolution lands
  - setting edge_confidence to reflect "edges are file-local only"
  - excluding entry-point names (main, __main__, test_*, setUp,
    tearDown) since those are typically callee-of-test-runner

The harness/agent can re-weight these against vulture's output and
the user's risk tolerance.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location


# Names we never flag — they're called by external systems (test
# runners, CLIs, frameworks) rather than in-repo code.
_ENTRY_POINTS = frozenset({
    "main",
    "__main__",
    "__init__",
    "__enter__",
    "__exit__",
    "__call__",
    "__str__",
    "__repr__",
    "__eq__",
    "__hash__",
    "setUp",
    "tearDown",
    "setUpClass",
    "tearDownClass",
})


def _is_entry_point(name: str) -> bool:
    if name in _ENTRY_POINTS:
        return True
    # Pytest collects test_* / Test* automatically.
    if name.startswith("test_") or name.startswith("Test"):
        return True
    # Public exports starting with uppercase letter often have external
    # callers (API surfaces). Don't flag them on this heuristic-only tier.
    return False


class DeadCodeDetector:
    kind = "dead_code"
    cost_tier = "static"

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]:
        findings: list[Finding] = []
        for fn in graph.list_functions(repo):
            if _is_entry_point(fn.name) or _is_entry_point(fn.name.split(".")[-1]):
                continue
            callers = graph.callers_of(repo, fn.guid)
            if callers:
                continue
            findings.append(Finding(
                kind=self.kind,
                # Low confidence on purpose — AST-only edges are file-
                # local for v0.1, so cross-file callers are invisible.
                confidence=0.5,
                edge_confidence=0.5,
                locations=[Location(
                    repo=repo, path=fn.module_path,
                    line=fn.line, end_line=fn.end_line, guid=fn.guid,
                )],
                evidence={
                    "function_name": fn.name,
                    "callers_in_repo": 0,
                    "edge_scope": "intra-file",
                },
                recommendation=Recommendation(
                    summary=f"Remove or verify external callers of `{fn.name}`",
                    rationale=(
                        f"`{fn.name}` has no callers in this repo's intra-file "
                        f"call graph. v0.1 doesn't resolve cross-file calls, "
                        f"so confirm by greping for the name before deleting."
                    ),
                    blast_radius=[fn.guid],
                    suggested_tests=[
                        f"grep -rn '\\b{fn.name}\\b' .",
                        "run the full test suite after removal",
                    ],
                ),
                source_detector=self.kind,
            ))
        return findings


register(DeadCodeDetector())
