"""empty_module — modules with no functions and no classes.

Often legitimate (re-export shims, conditional imports, type stubs)
but worth surfacing so the user can confirm. Confidence is low (0.3)
because the false-positive rate on this one is high by nature.

`__init__.py` package markers — empty modules whose only purpose is
to declare a directory as a Python package — are detected via
`is_marker_likely` (no imports, no docstring) and dropped to
LIKELY_MARKER_CONFIDENCE (0.05). Still emitted so the bus / graph
remains observable, but well below any subscriber threshold so they
never card.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location

DEFAULT_CONFIDENCE = 0.3
LIKELY_MARKER_CONFIDENCE = 0.05


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
            is_marker_likely = not m.imports and not m.docstring
            confidence = (
                LIKELY_MARKER_CONFIDENCE if is_marker_likely else DEFAULT_CONFIDENCE
            )
            findings.append(Finding(
                kind=self.kind,
                confidence=confidence,
                locations=[Location(repo=repo, path=m.path)],
                evidence={
                    "imports": m.imports,
                    "is_marker_likely": is_marker_likely,
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
