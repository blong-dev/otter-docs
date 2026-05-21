"""Auto-docs infrastructure layer.

Where the existing `detectors` package scans the code graph
(functions, classes, edges) for code-quality findings, this package
scans the **filesystem** for infrastructure surfaces: dependency
manifests, licenses, READMEs, test layouts, source-tree maps. The
output feeds the infrastructure sections of OTTER.md alongside the
existing code-graph sections.

Detectors are pure functions of the filesystem — no graph backend,
no LLM, no embeddings. They run cheaply (<10ms each on a real repo)
and can re-run without rescanning code.

See `docs/specs/refactor/auto-docs-infra-layer.md` for the design.
"""

from otter_docs.infra.base import (
    InfraDetector,
    infra_registry,
    register_infra_detector,
)
from otter_docs.infra.models import (
    Dependency,
    DependencyManifest,
    License,
    Readme,
    SourceLayout,
    SourceLayoutEntry,
    TestsLayout,
)


def _bootstrap() -> None:
    """Import detector modules so they self-register via `register_infra_detector`.

    Tolerant of missing modules during incremental development — a
    detector that hasn't been written yet just doesn't register; the
    rest still work. Once the universal-layer set is complete this
    can become a strict import.
    """
    for mod in ("dependencies", "license", "readme", "tests", "source_layout"):
        try:
            __import__(f"otter_docs.infra.{mod}")
        except ImportError:
            pass


_bootstrap()


__all__ = [
    "Dependency",
    "DependencyManifest",
    "InfraDetector",
    "License",
    "Readme",
    "SourceLayout",
    "SourceLayoutEntry",
    "TestsLayout",
    "infra_registry",
    "register_infra_detector",
]
