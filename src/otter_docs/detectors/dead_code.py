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
# runners, CLIs, frameworks) rather than in-repo code. The dunder
# list covers the common entry points the runtime/GC/protocol-machinery
# invokes implicitly; the test/setup names cover pytest/unittest.
_ENTRY_POINTS = frozenset({
    "main",
    "__main__",
    "__init__",
    "__new__",
    "__del__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__call__",
    "__str__",
    "__repr__",
    "__bool__",
    "__len__",
    "__iter__",
    "__next__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__contains__",
    "__eq__",
    "__ne__",
    "__hash__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__getattribute__",
    "__missing__",
    "__copy__",
    "__deepcopy__",
    "__reduce__",
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
        # Determine whether cross-file resolution has run by sampling
        # the call-graph: if any function has a caller from a different
        # module path, resolve() has produced cross-file edges and we
        # can trust the "no callers" signal more.
        resolution_ran = _detect_resolution_signal(repo, graph)
        confidence = 0.75 if resolution_ran else 0.5
        edge_confidence = 0.8 if resolution_ran else 0.5
        edge_scope = "cross-file" if resolution_ran else "intra-file"

        findings: list[Finding] = []
        for fn in graph.list_functions(repo):
            # `name` is sometimes "ClassName.method" — check both forms
            # so entry-point methods on classes don't get flagged.
            if _is_entry_point(fn.name) or _is_entry_point(fn.name.split(".")[-1]):
                continue
            callers = graph.callers_of(repo, fn.guid)
            if callers:
                continue
            findings.append(Finding(
                kind=self.kind,
                confidence=confidence,
                edge_confidence=edge_confidence,
                locations=[Location(
                    repo=repo, path=fn.module_path,
                    line=fn.line, end_line=fn.end_line, guid=fn.guid,
                )],
                evidence={
                    "function_name": fn.name,
                    "callers_in_repo": 0,
                    "edge_scope": edge_scope,
                    "resolution_ran": resolution_ran,
                },
                recommendation=Recommendation(
                    summary=f"Remove or verify external callers of `{fn.name}`",
                    rationale=(
                        f"`{fn.name}` has no callers in the {edge_scope} call "
                        f"graph. "
                        + (
                            "Cross-file resolution has run, so this is a "
                            "stronger signal than pure AST-local analysis — "
                            "but methods reached via dynamic dispatch "
                            "(self.x.method()) can still escape detection."
                            if resolution_ran else
                            "v0.1 hasn't resolved cross-file calls yet; "
                            "run Repo.resolve() first or confirm with a "
                            "grep before deleting."
                        )
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


def _detect_resolution_signal(repo: str, graph: GraphBackend) -> bool:
    """Heuristic: does the call graph contain any cross-module CALLS edge?

    Build a guid → module_path index once (O(N)), then walk a sample
    of edges checking source and target modules (O(sample × edges_per
    _func)). Returning True means resolve() has produced at least one
    cross-file edge — we use this to tune dead_code's confidence.
    """
    guid_to_path: dict[str, str] = {}
    sample_targets: list[str] = []
    for fn in graph.list_functions(repo):
        guid_to_path[fn.guid] = fn.module_path
        if len(sample_targets) < 100:
            sample_targets.append(fn.guid)
    for target_guid in sample_targets:
        target_path = guid_to_path.get(target_guid)
        if target_path is None:
            continue
        for e in graph.edges_to(repo, target_guid):
            if e.kind != "CALLS":
                continue
            src_path = guid_to_path.get(e.src_id)
            if src_path is not None and src_path != target_path:
                return True
    return False


register(DeadCodeDetector())
