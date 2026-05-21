"""Bridge from `InfraDetector` + per-surface renderer to the
`Renderer` Protocol the existing render_document flow uses.

Each infra section is a Renderer whose `render()` runs the detector
against the repo's root and dispatches to the per-surface renderer
function. If the detector returns None, the renderer emits a single
italics line marking the absence so the marker stays live for
future re-renders.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from otter_docs.infra.base import InfraDetector
from otter_docs.infra.models import InfraSurface
from otter_docs.render.base import register

if TYPE_CHECKING:
    from otter_docs.repo import Repo


NOT_DETECTED = "*(not detected in this repo)*"


@dataclass
class InfraSectionRenderer:
    """Renderer that wraps an InfraDetector + a surface renderer.

    Exposes the `Renderer` Protocol so it slots into the existing
    section registry. `name` is the marker section id in OTTER.md.

    Handles both single-surface and multi-surface (monorepo) detector
    return shapes: a list result is rendered as a sequence of
    per-subtree blocks joined with a blank line.
    """

    name: str
    detector: InfraDetector
    surface_renderer: Callable[[InfraSurface], str]

    def render(self, repo: Repo) -> str:
        result = self.detector.detect(repo=repo.name, repo_root=repo.root)
        if result is None:
            return NOT_DETECTED
        if isinstance(result, list):
            if not result:
                return NOT_DETECTED
            blocks = [self.surface_renderer(s) for s in result]
            return "\n".join(blocks).rstrip() + "\n"
        return self.surface_renderer(result)


def register_infra_sections() -> None:
    """Pair each registered InfraDetector with its renderer and
    register the resulting `InfraSectionRenderer` on the existing
    Renderer registry. Idempotent — re-registering with the same
    `name` just replaces the entry.
    """
    from otter_docs.infra import infra_registry
    from otter_docs.render.infra import dependencies as r_deps
    from otter_docs.render.infra import license as r_license
    from otter_docs.render.infra import readme as r_readme
    from otter_docs.render.infra import source_layout as r_source
    from otter_docs.render.infra import tests as r_tests

    pairings: dict[str, Callable[[InfraSurface], str]] = {
        "dependencies": r_deps.render_dependencies,
        "license": r_license.render_license,
        "readme": r_readme.render_readme,
        "tests": r_tests.render_tests,
        "source_layout": r_source.render_source_layout,
    }

    registry = infra_registry()
    for kind, detector in registry.items():
        renderer = pairings.get(kind)
        if renderer is None:
            continue  # no renderer module yet — skip silently
        register(InfraSectionRenderer(
            name=kind, detector=detector, surface_renderer=renderer,
        ))
