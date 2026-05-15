"""Renderer Protocol + registry.

A Renderer turns a scanned/enriched Repo into a markdown fragment.
Renderers are pure read — they query the graph and findings, never
mutate. Each has a stable `name` used both as the registry key and
as the marker section id in generated documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from otter_docs.repo import Repo


class Renderer(Protocol):
    name: str

    def render(self, repo: Repo) -> str: ...


_registry: dict[str, Renderer] = {}


def register(renderer: Renderer) -> None:
    _registry[renderer.name] = renderer


def registry() -> dict[str, Renderer]:
    return dict(_registry)


def render_section(name: str, repo: Repo) -> str:
    """Render one section by name. Raises KeyError on unknown name."""
    return _registry[name].render(repo)
