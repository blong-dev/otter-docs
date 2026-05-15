"""Structured-output schemas for agent integration.

Re-exports the typed product (Finding, Recommendation, Review) and
adds the grading shapes (Grade, GradeReport) the Harness emits. Every
model here has a JSON Schema available via `json_schema(Model)` so a
non-Python agent can validate the library's structured output.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from otter_docs.findings import Finding, Recommendation
from otter_docs.llm_direct import Review

__all__ = [
    "Finding",
    "Grade",
    "GradeReport",
    "Recommendation",
    "Review",
    "json_schema",
]


# Letter grades map onto a 0-100 score. Kept coarse on purpose — the
# value is the breakdown + findings, not false precision in the number.
LetterGrade = Literal["A", "B", "C", "D", "F"]


class Grade(BaseModel):
    """A single graded dimension of codebase health.

    `dimension` is one of the analysis axes (redundancy, dead_code,
    complexity, ...). `score` is 0-100, `letter` is the coarse bucket,
    `rationale` is one sentence an agent can surface to a human.
    """

    model_config = ConfigDict(frozen=True)

    dimension: str
    score: float = Field(ge=0.0, le=100.0)
    letter: LetterGrade
    rationale: str
    finding_count: int = 0
    # False when the detectors backing this dimension never ran (e.g.
    # redundancy needs enrich(); without it the dimension is *unknown*,
    # not perfect). Unassessed dimensions are excluded from the overall
    # score so a skipped stage can't inflate the grade.
    assessed: bool = True


class GradeReport(BaseModel):
    """The Harness's top-level output.

    overall_score is the weighted mean of the per-dimension grades;
    `grades` is the breakdown; `top_findings` is the ranked shortlist
    the agent should act on first; `proposed_changes` carries any
    LLM-generated consolidation diffs produced during the run.
    """

    model_config = ConfigDict(frozen=True)

    repo: str
    overall_score: float = Field(ge=0.0, le=100.0)
    overall_letter: LetterGrade
    summary: str
    grades: list[Grade] = Field(default_factory=list)
    top_findings: list[Finding] = Field(default_factory=list)
    proposed_changes: list[Recommendation] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


def score_to_letter(score: float) -> LetterGrade:
    """Map a 0-100 score onto a letter. Standard US-ish cutoffs."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema for a pydantic model.

    Thin wrapper over pydantic's `model_json_schema()` so callers
    don't need to know which pydantic version is installed or the
    exact method name.
    """
    return model.model_json_schema()
