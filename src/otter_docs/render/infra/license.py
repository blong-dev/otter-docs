"""Render a License surface to a markdown section."""

from __future__ import annotations

from otter_docs.infra.models import License


def render_license(surface: License) -> str:
    if surface.spdx_id:
        return (
            f"**License:** {surface.spdx_id} "
            f"(see [`{surface.path}`]({surface.path}))\n"
        )
    snippet = surface.header_summary.strip()
    if snippet:
        return (
            f"**License file:** [`{surface.path}`]({surface.path}) — "
            f"{snippet}\n"
        )
    return f"**License file:** [`{surface.path}`]({surface.path})\n"
