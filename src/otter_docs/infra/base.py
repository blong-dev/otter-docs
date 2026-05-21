"""InfraDetector Protocol + registry.

Mirrors `otter_docs.detectors.base` — a small Protocol plus a global
registry. Detectors register themselves at import time; consumers
(the renderer integration in `render_document`) iterate the registry
and call each detector's `detect()` against the repo root.

A detector returns either an `InfraSurface` record (the surface is
present in this repo) or None (this repo doesn't have this surface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from otter_docs.infra.models import InfraSurface


class InfraDetector(Protocol):
    """Detects one named infrastructure surface in a repo's filesystem.

    `kind` is the surface name used as both the registry key and the
    OTTER.md marker section id. Detectors do not touch the graph
    backend or the LLM — they read files on disk only.

    Return shape (2026-05-21 monorepo support):
      - `None`        → this surface isn't present
      - InfraSurface  → one instance (the typical single-package case)
      - list          → multiple instances, one per subtree (monorepo)
    The render adapter handles both single and list shapes; renderers
    operate on individual surface records.
    """

    kind: str

    def detect(
        self, *, repo: str, repo_root: Path
    ) -> InfraSurface | list[InfraSurface] | None: ...


_registry: dict[str, InfraDetector] = {}


def register_infra_detector(detector: InfraDetector) -> None:
    """Add a detector to the global registry, keyed by its `kind`."""
    _registry[detector.kind] = detector


def infra_registry() -> dict[str, InfraDetector]:
    """Read-only view of registered infra detectors."""
    return dict(_registry)
