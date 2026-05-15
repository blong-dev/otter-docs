"""Harness — drives the one-shot agent loop and emits a GradeReport.

    Harness(repo, llm=..., embedder=...).run()

does: (optionally) scan → resolve → enrich → findings → rank →
propose consolidations for the top redundancies → grade. The result
is a GradeReport: an overall letter + per-dimension breakdown + the
ranked shortlist + any LLM-proposed diffs.

The Harness never applies changes. It produces the report; the
caller (a Bamboo job, a scheduled task, a human) decides what to do.

Grading is deliberately simple and explainable — the value is the
findings + diffs, not a precise number. Each dimension starts at 100
and loses points proportional to its confidence-weighted finding
density (findings per indexed symbol). The rationale string spells
out the math so a human can sanity-check the grade.
"""

from __future__ import annotations

from dataclasses import dataclass

from otter_docs.agent.schemas import Grade, GradeReport, score_to_letter
from otter_docs.clients.base import EmbeddingClient, LLMClient
from otter_docs.findings import Finding, Recommendation
from otter_docs.repo import Repo


# Which finding kinds roll up into which health dimension. A kind not
# listed here still counts toward the overall via the catch-all
# "other" dimension so nothing silently disappears.
_DIMENSION_MAP: dict[str, str] = {
    "redundancy.semantic_equivalence": "redundancy",
    "dead_code": "dead_code",
    "large_function": "complexity",
    "empty_module": "structure",
    "description.divergence": "documentation",
}

# How many points one unit of confidence-weighted finding density
# costs. Tuned so a repo with ~1 high-confidence finding per 5 symbols
# in a dimension lands around a C. Intentionally lenient — otter-docs
# surfaces a lot of low-confidence signal and we don't want to nuke
# the grade for it.
_PENALTY_SCALE = 250.0


@dataclass
class Harness:
    repo: Repo
    llm: LLMClient | None = None
    embedder: EmbeddingClient | None = None
    # Cap on how many redundancy findings we spend LLM calls on per run.
    max_consolidations: int = 5

    def run(
        self,
        *,
        do_scan: bool = True,
        do_resolve: bool = True,
        do_enrich: bool = False,
        propose: bool = True,
    ) -> GradeReport:
        """Execute the loop and return a GradeReport.

        Parameters
        ----------
        do_scan / do_resolve / do_enrich :
            Skip stages already done in a previous run. `do_enrich`
            defaults False because it's the expensive one and many
            callers enrich on a schedule, not every harness run.
        propose :
            When True and an LLM is configured, generate consolidation
            diffs for the top redundancy findings.
        """
        if do_scan:
            self.repo.scan()
        if do_resolve:
            self.repo.resolve()
        if do_enrich:
            if self.llm is None or self.embedder is None:
                raise RuntimeError(
                    "do_enrich=True needs both an LLM and an embedder on the Harness"
                )
            self.repo.enrich(self.llm, self.embedder)

        findings = self.repo.findings()
        ranked = _rank(findings)

        # Symbol count is the denominator for finding density. Use
        # functions + classes; modules aren't where defects concentrate.
        symbol_count = (
            len(list(self.repo.graph.list_functions(self.repo.name)))
            + len(list(self.repo.graph.list_classes(self.repo.name)))
        )

        enriched = _is_enriched(self.repo)
        grades = _grade_dimensions(findings, symbol_count, enriched=enriched)
        overall = _overall_score(grades)

        proposed: list[Recommendation] = []
        if propose and self.llm is not None:
            for f in ranked:
                if len(proposed) >= self.max_consolidations:
                    break
                if f.kind.startswith("redundancy."):
                    proposed.append(
                        self.repo.propose_consolidation(f, self.llm)
                    )

        return GradeReport(
            repo=self.repo.name,
            overall_score=overall,
            overall_letter=score_to_letter(overall),
            summary=_summary(self.repo.name, overall, grades, len(findings)),
            grades=grades,
            top_findings=ranked[:20],
            proposed_changes=proposed,
            stats={
                "total_findings": len(findings),
                "symbols": symbol_count,
                "consolidations_proposed": len(proposed),
                "enriched": enriched,
            },
        )


def _is_enriched(repo: Repo) -> bool:
    """True if any function carries a description_vec.

    Sampling the first few functions is enough — enrich() is
    all-or-nothing per run, so if the front of the list has vectors
    the graph was enriched. Cheap signal that lets the grader mark
    embedding-tier dimensions assessed vs unknown.
    """
    for i, fn in enumerate(repo.graph.list_functions(repo.name)):
        if fn.description_vec is not None:
            return True
        if i >= 20:
            break
    return False


def _rank(findings: list[Finding]) -> list[Finding]:
    """Sort by confidence × edge_confidence (treating null edge as 1.0).

    A finding the call graph couldn't trust (low edge_confidence)
    sinks below an equally-confident purely-local one, which is the
    behavior the spec's `edge_confidence` design calls for.
    """
    def key(f: Finding) -> float:
        ec = f.edge_confidence if f.edge_confidence is not None else 1.0
        return f.confidence * ec

    return sorted(findings, key=key, reverse=True)


def _grade_dimensions(
    findings: list[Finding], symbol_count: int, *, enriched: bool,
) -> list[Grade]:
    # Bucket confidence-weighted counts by dimension.
    weighted: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f in findings:
        dim = _DIMENSION_MAP.get(f.kind, "other")
        ec = f.edge_confidence if f.edge_confidence is not None else 1.0
        weighted[dim] = weighted.get(dim, 0.0) + f.confidence * ec
        counts[dim] = counts.get(dim, 0) + 1

    # These dimensions are only meaningful once enrich() has produced
    # vectors — their detectors are embedding-tier. Without enrichment
    # they're *unknown*, not perfect; we mark them unassessed so a
    # skipped stage can't inflate the overall grade.
    embedding_dims = {"redundancy", "documentation"}

    denom = max(1, symbol_count)
    grades: list[Grade] = []
    dimensions = ["redundancy", "dead_code", "complexity", "structure", "documentation"]
    if "other" in weighted:
        dimensions.append("other")
    for dim in dimensions:
        assessed = enriched or dim not in embedding_dims
        w = weighted.get(dim, 0.0)
        density = w / denom
        score = max(0.0, 100.0 - density * _PENALTY_SCALE)
        if not assessed:
            grades.append(Grade(
                dimension=dim,
                score=0.0,
                letter="F",  # placeholder; excluded from overall
                finding_count=0,
                assessed=False,
                rationale=(
                    f"not assessed — {dim} needs enrich() (embedding-tier "
                    f"detectors). Run with do_enrich=True for a real grade."
                ),
            ))
            continue
        grades.append(Grade(
            dimension=dim,
            score=score,
            letter=score_to_letter(score),
            finding_count=counts.get(dim, 0),
            assessed=True,
            rationale=(
                f"{counts.get(dim, 0)} findings, confidence-weighted "
                f"{w:.1f} over {symbol_count} symbols "
                f"(density {density:.3f} × {_PENALTY_SCALE:.0f} = "
                f"{density * _PENALTY_SCALE:.1f} pt penalty)"
            ),
        ))
    return grades


def _overall_score(grades: list[Grade]) -> float:
    assessed = [g for g in grades if g.assessed]
    if not assessed:
        return 100.0
    # Equal-weight mean across *assessed* dimensions only.
    return sum(g.score for g in assessed) / len(assessed)


def _summary(
    repo: str, overall: float, grades: list[Grade], total: int,
) -> str:
    assessed = [g for g in grades if g.assessed]
    unassessed = [g.dimension for g in grades if not g.assessed]
    worst = min(assessed, key=lambda g: g.score) if assessed else None
    worst_txt = (
        f" Weakest assessed dimension: {worst.dimension} ({worst.letter})."
        if worst is not None else ""
    )
    unassessed_txt = (
        f" Not assessed (needs enrich): {', '.join(unassessed)}."
        if unassessed else ""
    )
    return (
        f"{repo}: overall {score_to_letter(overall)} "
        f"({overall:.0f}/100) across {len(assessed)} assessed "
        f"dimensions, {total} findings.{worst_txt}{unassessed_txt}"
    )
