"""Render a TestsLayout surface to a markdown section."""

from __future__ import annotations

from otter_docs.infra.models import TestsLayout


def render_tests(surface: TestsLayout) -> str:
    runner_str = ", ".join(f"`{r}`" for r in surface.runners) if surface.runners else "—"
    dir_str = ", ".join(f"`{d}`" for d in surface.dirs) if surface.dirs else "—"
    return (
        f"- **Test files:** {surface.file_count}\n"
        f"- **Directories:** {dir_str}\n"
        f"- **Runners (inferred):** {runner_str}\n"
    )
