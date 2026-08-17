"""The five v0.1 renderers.

Each produces a self-contained markdown fragment. They share helpers
but stay independent so a consumer can render just one. Output is
deterministic given a fixed graph (sorted everywhere) so re-renders
produce minimal diffs.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from otter_docs.discovery import is_test_path
from otter_docs.render.base import register

if TYPE_CHECKING:
    from otter_docs.repo import Repo


def _include_tests() -> bool:
    """Whether production-focused renders should include test code. Off by
    default so test fixtures don't dominate the signal; opt in with
    ``OTTER_INCLUDE_TESTS=1`` (the ``render --include-tests`` flag sets it)."""
    return os.environ.get("OTTER_INCLUDE_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _counts(repo: Repo) -> tuple[int, int, int]:
    mods = list(repo.graph.list_modules(repo.name))
    fns = list(repo.graph.list_functions(repo.name))
    cls = list(repo.graph.list_classes(repo.name))
    return len(mods), len(fns), len(cls)


class SystemOverviewRenderer:
    """High-level shape of the codebase: languages, sizes, biggest modules."""

    name = "system_overview"

    def render(self, repo: Repo) -> str:
        modules = list(repo.graph.list_modules(repo.name))
        fns = list(repo.graph.list_functions(repo.name))
        classes = list(repo.graph.list_classes(repo.name))

        by_lang: Counter[str] = Counter()
        for m in modules:
            lang = m.language.value if hasattr(m.language, "value") else str(m.language)
            by_lang[lang] += 1

        include_tests = _include_tests()
        n_test_modules = sum(1 for m in modules if is_test_path(m.path))

        fns_per_module: Counter[str] = Counter()
        for f in fns:
            if not include_tests and is_test_path(f.module_path):
                continue
            fns_per_module[f.module_path] += 1
        biggest = fns_per_module.most_common(10)

        lines: list[str] = []
        lines.append(f"**{repo.name}** — {len(modules)} modules, "
                     f"{len(fns)} functions, {len(classes)} classes.")
        lines.append("")
        lines.append("Languages:")
        for lang, n in sorted(by_lang.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {lang}: {n} modules")
        lines.append("")
        if biggest:
            label = (
                "Largest modules by function count:"
                if include_tests
                else "Largest production modules by function count:"
            )
            lines.append(label)
            for path, n in biggest:
                lines.append(f"- `{path}` — {n} functions")
            if n_test_modules and not include_tests:
                lines.append("")
                lines.append(
                    f"_{n_test_modules} test module(s) excluded; "
                    "set OTTER_INCLUDE_TESTS=1 to include._"
                )
        return "\n".join(lines)


class FindingsSummaryRenderer:
    """All findings grouped by kind, with counts and mean confidence."""

    name = "findings_summary"

    def render(self, repo: Repo) -> str:
        findings = repo.findings()
        if not findings:
            return "No findings. (Did you run scan/resolve/enrich first?)"
        by_kind: dict[str, list] = defaultdict(list)
        for f in findings:
            by_kind[f.kind].append(f)
        lines = [f"{len(findings)} findings across {len(by_kind)} kinds.", ""]
        lines.append("| kind | count | mean confidence |")
        lines.append("|---|---:|---:|")
        for kind in sorted(by_kind, key=lambda k: -len(by_kind[k])):
            fs = by_kind[kind]
            mean_conf = sum(f.confidence for f in fs) / len(fs)
            lines.append(f"| {kind} | {len(fs)} | {mean_conf:.2f} |")
        return "\n".join(lines)


class RedundancyReportRenderer:
    """Confirmed duplicate pairs only.

    The semantic_equivalence detector is high-recall — many candidate pairs
    turn out to be siblings, shared patterns, or same-name idioms (an `envOr`
    helper in two Go modules, a local `decorator` inside two decorator
    factories). This renderer consults the cached `confirm_redundancy`
    verdicts and lists only pairs confirmed as real duplicates, with a
    concrete consolidation target. Reclassified and not-yet-confirmed
    candidates are summarised, not listed as redundant."""

    name = "redundancy_report"

    def render(self, repo: Repo) -> str:
        findings = list(repo.findings(kinds={"redundancy.semantic_equivalence"}))
        if not findings:
            return (
                "No redundancy findings. Note: this needs enrich() — "
                "without embeddings the semantic_equivalence detector "
                "produces nothing."
            )

        confirmed: list[tuple[Any, Any]] = []
        reclassified = 0
        unconfirmed = 0
        for f in findings:
            verdict = repo.cached_redundancy_verdict(f)
            if verdict is None:
                unconfirmed += 1
            elif verdict.is_duplicate:
                confirmed.append((f, verdict))
            else:
                reclassified += 1
        confirmed.sort(key=lambda fv: fv[1].confidence, reverse=True)

        lines: list[str] = []
        if confirmed:
            lines.append(f"{len(confirmed)} confirmed duplicate pair(s).")
            lines.append("")
            for f, v in confirmed[:50]:
                names = f.evidence.get("function_names", ["?", "?"])
                locs = f.locations[:2]
                a = f"`{locs[0].path}`:{locs[0].line}" if locs else "`?`"
                b = (
                    f"`{locs[1].path}`:{locs[1].line}"
                    if len(locs) > 1
                    else "`?`"
                )
                lines.append(
                    f"- **{names[0]}** ↔ **{names[1]}** "
                    f"(confidence {v.confidence:.2f}) — {a} ↔ {b}"
                )
                lines.append(f"  - Consolidate the two copies ({a} / {b}).")
        else:
            lines.append("No confirmed duplicates.")

        tail: list[str] = []
        if reclassified:
            tail.append(
                f"{reclassified} candidate(s) reclassified as "
                "sibling / shared-pattern / coincidental"
            )
        if unconfirmed:
            tail.append(
                f"{unconfirmed} candidate(s) pending confirmation "
                "(run the confirm pass)"
            )
        if tail:
            lines.append("")
            lines.append("_" + "; ".join(tail) + "._")
        return "\n".join(lines)


class DependencyGraphRenderer:
    """Module-to-module IMPORTS as a mermaid graph (capped for readability)."""

    name = "dependency_graph"

    MAX_EDGES = 60

    def render(self, repo: Repo) -> str:
        modules = list(repo.graph.list_modules(repo.name))
        edges: list[tuple[str, str]] = []
        for m in modules:
            for e in repo.graph.edges_from(repo.name, m.path, kind="IMPORTS"):
                # dst_id is an imported module name (string), not a guid.
                edges.append((m.path, e.dst_id))
        edges.sort()
        if not edges:
            return "No import edges recorded."
        capped = edges[: self.MAX_EDGES]
        lines = ["```mermaid", "graph LR"]
        for src, dst in capped:
            s = _mermaid_id(src)
            d = _mermaid_id(dst)
            lines.append(f'  {s}["{src}"] --> {d}["{dst}"]')
        lines.append("```")
        if len(edges) > self.MAX_EDGES:
            lines.append("")
            lines.append(
                f"_Showing {self.MAX_EDGES} of {len(edges)} import edges._"
            )
        return "\n".join(lines)


class ArchitectureSmellsRenderer:
    """Large functions + fan-in/out outliers from the call graph."""

    name = "architecture_smells"

    def render(self, repo: Repo) -> str:
        include_tests = _include_tests()
        large = sorted(
            repo.findings(kinds={"large_function"}),
            key=lambda f: f.evidence.get("lines", 0),
            reverse=True,
        )
        if not include_tests:
            large = [
                f
                for f in large
                if not (f.locations and is_test_path(f.locations[0].path))
            ]
        fns = list(repo.graph.list_functions(repo.name))
        fan_in: list[tuple[int, str, str]] = []
        for fn in fns:
            if not include_tests and is_test_path(fn.module_path):
                continue
            n = len(repo.graph.callers_of(repo.name, fn.guid))
            if n >= 8:  # arbitrary "hub" threshold for v0.1
                fan_in.append((n, fn.name, fn.module_path))
        fan_in.sort(reverse=True)

        lines: list[str] = []
        lines.append("### Largest functions")
        if large:
            for f in large[:15]:
                loc = f.locations[0]
                lines.append(
                    f"- `{loc.path}`:{loc.line} **{f.evidence.get('function_name')}** "
                    f"— {f.evidence.get('lines')} lines"
                )
        else:
            lines.append("- none over threshold")
        lines.append("")
        lines.append("### Call-graph hubs (fan-in ≥ 8)")
        if fan_in:
            for n, name, path in fan_in[:15]:
                lines.append(f"- **{name}** (`{path}`) — {n} callers")
        else:
            lines.append("- none (or resolve() not run yet)")
        return "\n".join(lines)


def _mermaid_id(s: str) -> str:
    """Mermaid node ids can't contain slashes/dots — slugify."""
    return "n_" + "".join(c if c.isalnum() else "_" for c in s)


register(SystemOverviewRenderer())
register(FindingsSummaryRenderer())
register(RedundancyReportRenderer())
register(DependencyGraphRenderer())
register(ArchitectureSmellsRenderer())
