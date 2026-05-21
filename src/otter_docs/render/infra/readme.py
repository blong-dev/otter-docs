"""Render a Readme surface to a markdown section."""

from __future__ import annotations

from otter_docs.infra.models import Readme


def render_readme(surface: Readme) -> str:
    lines: list[str] = [
        f"*From [`{surface.path}`]({surface.path}):*",
        "",
    ]
    if surface.summary:
        lines.append(f"> {surface.summary}")
        lines.append("")
    if surface.h2_outline:
        lines.append("**Sections:** " + ", ".join(surface.h2_outline))
    return "\n".join(lines).rstrip() + "\n"
