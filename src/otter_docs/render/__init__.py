"""Doc rendering: five renderers + marker-based injection.

`Repo.render(name)` returns one renderer's markdown fragment.
`Repo.render_document(path)` writes/updates a full doc with all
sections, preserving human prose outside the markers.
"""

from __future__ import annotations

from otter_docs.render.base import Renderer, register, registry, render_section
from otter_docs.render.markers import (
    bootstrap_document,
    inject,
    sections_in,
)

__all__ = [
    "Renderer",
    "bootstrap_document",
    "inject",
    "register",
    "registry",
    "render_section",
    "sections_in",
]


def _bootstrap() -> None:
    from otter_docs.render import renderers as _r  # noqa: F401


_bootstrap()
