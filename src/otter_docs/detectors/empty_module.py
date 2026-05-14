"""empty_module — modules with no functions and no classes.

Often legitimate (re-export shims, conditional imports, type stubs)
but worth surfacing so the user can confirm. Confidence is low (0.3)
because the false-positive rate on this one is high by nature.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location


class EmptyModuleDetector:
    kind = "empty_module"
    cost_tier = "static"

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]:
        # Build a set of (module_path) that have at least one function
        # or class. Easier than a per-module count query.
        non_empty: set[str] = set()
        for fn in graph.list_functions(repo):
            non_empty.add(fn.module_path)
        for cls in graph.list_classes(repo):
            non_empty.add(cls.module_path)

        findings: list[Finding] = []
        for m in graph.list_modules(repo):
            if m.path in non_empty:
                continue
            # If the module imports nothing either, it might be a
            # __init__.py marker — still emit but flag in evidence.
            findings.append(Finding(
                kind=self.kind,
                confidence=0.3,
                locations=[Location(repo=repo, path=m.path)],
                evidence={
                    "imports": m.imports,
                    "is_marker_likely": not m.imports and not m.docstring,
                },
                recommendation=Recommendation(
                    summary=f"Confirm `{m.path}` should exist",
                    rationale=(
                        f"`{m.path}` contains no functions or classes. "
                        f"Common legitimate reasons: re-export shim, "
                        f"`__init__.py` marker, type stub. Confirm before "
                        f"deleting."
                    ),
                ),
                source_detector=self.kind,
            ))
        return findings


register(EmptyModuleDetector())
