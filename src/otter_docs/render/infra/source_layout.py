"""Render a SourceLayout surface to a markdown section."""

from __future__ import annotations

from otter_docs.infra.models import SourceLayout


def render_source_layout(surface: SourceLayout) -> str:
    lines: list[str] = [
        "| Entry | Kind | Notes |",
        "|-------|------|-------|",
    ]
    for entry in surface.entries:
        kind = "dir" if entry.is_dir else "file"
        notes = entry.annotation or "—"
        suffix = "/" if entry.is_dir else ""
        lines.append(f"| `{entry.name}{suffix}` | {kind} | {notes} |")
    return "\n".join(lines) + "\n"
