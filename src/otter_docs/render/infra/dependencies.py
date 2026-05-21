"""Render a DependencyManifest surface to a markdown section."""

from __future__ import annotations

from otter_docs.infra.models import Dependency, DependencyManifest

_FORMAT_LABELS = {
    "pyproject": "Python (pyproject.toml — PEP 621)",
    "poetry": "Python (pyproject.toml — Poetry)",
    "package_json": "Node (package.json)",
    "go_mod": "Go (go.mod)",
    "cargo": "Rust (Cargo.toml)",
    "gemfile": "Ruby (Gemfile)",
    "pom": "Java (pom.xml)",
    "gradle": "JVM (build.gradle)",
    "podfile": "iOS (Podfile)",
    "composer": "PHP (composer.json)",
    "mix": "Elixir (mix.exs)",
    "pubspec": "Dart (pubspec.yaml)",
    "requirements": "Python (requirements.txt)",
}


def render_dependencies(surface: DependencyManifest) -> str:
    label = _FORMAT_LABELS.get(surface.format, surface.format)
    lines: list[str] = [
        f"*{len(surface.direct)} direct"
        + (f" + {len(surface.dev)} dev" if surface.dev else "")
        + f" — auto-generated from `{surface.path}` ({label})*",
        "",
    ]
    if surface.direct:
        lines.append("**Direct dependencies**")
        lines.append("")
        lines.append("| Package | Version |")
        lines.append("|---------|---------|")
        for dep in surface.direct:
            lines.append(_dep_row(dep))
    if surface.dev:
        lines.append("")
        lines.append("**Development dependencies**")
        lines.append("")
        lines.append("| Package | Version |")
        lines.append("|---------|---------|")
        for dep in surface.dev:
            lines.append(_dep_row(dep))
    if not surface.direct and not surface.dev:
        lines.append("*(manifest found but contains no dependencies)*")
    return "\n".join(lines) + "\n"


def _dep_row(dep: Dependency) -> str:
    ver = dep.version_spec or "—"
    return f"| `{dep.name}` | `{ver}` |"
