"""Lightweight benchmark for the scan → resolve → findings pipeline.

Not a microbenchmark framework — just enough to produce the numbers
the README promises (cold scan time, resolve time, findings time)
over a real repo, repeatably. Run locally against v3/gnosis or any
checkout; the numbers go in the README honestly.

CI does NOT run this — timing on shared GitHub runners is noise. It's
a local/manual tool.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Phase:
    name: str
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResult:
    repo: str
    phases: list[Phase] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(p.seconds for p in self.phases)

    def report(self) -> str:
        lines = [f"otter-docs bench — {self.repo}"]
        for p in self.phases:
            extra = ""
            if p.detail:
                extra = "  " + " ".join(f"{k}={v}" for k, v in p.detail.items())
            lines.append(f"  {p.name:10s} {p.seconds:7.2f}s{extra}")
        lines.append(f"  {'TOTAL':10s} {self.total_seconds:7.2f}s")
        return "\n".join(lines)


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def benchmark(repo_root: str | Path, *, name: str | None = None) -> BenchResult:
    """Run scan → resolve → findings against `repo_root`, timing each.

    Uses an in-memory SqliteBackend so disk I/O on the graph doesn't
    skew the numbers — we're measuring otter-docs's work, not sqlite
    fsync. Enrichment is excluded (it's model-bound, benchmarked
    separately when a real embedder is available).
    """
    from otter_docs import Repo
    from otter_docs.backends import SqliteBackend

    root = Path(repo_root)
    repo_name = name or root.name
    result = BenchResult(repo=repo_name)

    with Repo(root, name=repo_name, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        dt, scan_report = _timed(lambda: repo.scan(reset=True))
        result.phases.append(Phase("scan", dt, {
            "files": scan_report.files_parsed,
            "functions": scan_report.functions,
            "edges": scan_report.edges,
        }))

        dt, resolve_reports = _timed(repo.resolve)
        edges = sum(r.edges_emitted for r in resolve_reports.values())
        result.phases.append(Phase("resolve", dt, {"edges": edges}))

        dt, findings = _timed(repo.findings)
        result.phases.append(Phase("findings", dt, {"count": len(findings)}))

    return result
