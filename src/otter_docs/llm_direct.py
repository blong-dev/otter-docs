"""LLM-direct tier — agent-callable methods that ask the LLM for specific work.

Unlike the embedding-tier detectors which run at scan-time over vectors,
the methods here are on-demand: the agent has already triaged Findings
and wants the model to do something concrete — propose a consolidation
diff, review a proposed change, describe a single symbol.

Library still never applies changes. Each method returns a typed
artifact (Recommendation with proposed_diff, Review, Description) and
the harness decides what to do with it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from otter_docs.backends.base import GraphBackend
from otter_docs.clients.base import LLMClient
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import FunctionRecord


# ── public schemas ──────────────────────────────────────────────────────


ReviewVerdict = Literal["approve", "request_changes", "comment"]


class Review(BaseModel):
    """Structured assessment of a proposed change.

    Returned by `review_change`. Designed for an agent to consume:
    `overall` is the headline (approve / request_changes / comment),
    the lists drill down into specifics. Empty lists are valid —
    `approve` with no blockers and no new_risks is the green-light case.
    """

    model_config = ConfigDict(frozen=True)

    summary: str
    overall: ReviewVerdict
    addresses_findings: list[str] = Field(default_factory=list)
    new_risks: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


# ── prompts (kept in code for now, not externalized) ────────────────────


_CONSOLIDATION_PROMPT = """\
Two functions in the same codebase appear semantically equivalent.

Canonical (keep this one):
  path: {canonical_path}
  name: {canonical_name}
  lines {canonical_start}-{canonical_end}

```{language}
{canonical_source}
```

Redundant (consolidate into the canonical):
  path: {redundant_path}
  name: {redundant_name}
  lines {redundant_start}-{redundant_end}

```{language}
{redundant_source}
```

Produce a unified diff that:
  1. Deletes the redundant function from its current location.
  2. If the canonical name differs, also updates the redundant function's
     name at any call sites you can see in the snippets above.

Rules:
  - Output ONLY the unified diff, no commentary, no fenced code block.
  - Use standard diff format with `---`, `+++`, `@@` headers.
  - If you can't safely produce a consolidation (different signatures,
    different side effects, or you're not confident), output an empty
    string — nothing else.
"""


_REVIEW_PROMPT = """\
You are reviewing a proposed change to a codebase.

Diff:
```
{diff}
```

{findings_context}

Assess the change. Return a JSON object with these exact fields:

  summary           one short sentence on what the diff does
  overall           one of "approve", "request_changes", "comment"
  addresses_findings  list of finding-kind strings the diff appears to fix
  new_risks         list of short strings describing potential new issues
  blockers          list of short strings describing problems that MUST be
                    fixed before merging

If the diff is straightforward and safe, "approve" with empty lists is the
right answer. If you'd want to ask questions before approving, that's
"comment". Only "request_changes" when there are concrete blockers.

Return JSON only, no commentary, no fenced block.
"""


# ── public entry points ─────────────────────────────────────────────────


def propose_consolidation(
    *,
    finding: Finding,
    repo: str,
    repo_root: Path,
    graph: GraphBackend,
    llm: LLMClient,
    llm_options: dict[str, Any] | None = None,
) -> Recommendation:
    """Generate a consolidation diff for a redundancy finding.

    Returns a Recommendation with `proposed_diff` populated (or empty
    string if the LLM declined). Always returns a Recommendation;
    callers attach it to the Finding if they want to persist it.

    Raises ValueError if `finding.kind` isn't a redundancy/equivalence
    shape — the function expects exactly two functions in evidence.
    """
    if not finding.kind.startswith("redundancy."):
        raise ValueError(
            f"propose_consolidation expects a redundancy.* finding, got {finding.kind!r}"
        )
    if len(finding.locations) < 2:
        raise ValueError("redundancy finding must have at least two locations")
    canonical_guid = finding.evidence.get("canonical_guid")
    if not canonical_guid:
        # No explicit canonical → use the longer of the two.
        a, b = finding.locations[0], finding.locations[1]
        canonical_loc, redundant_loc = (a, b) if _lines(a) >= _lines(b) else (b, a)
    else:
        canonical_loc = next(
            (loc for loc in finding.locations if loc.guid == canonical_guid),
            finding.locations[0],
        )
        redundant_loc = next(
            (loc for loc in finding.locations if loc.guid != canonical_guid),
            finding.locations[1],
        )

    canonical_fn = _fetch_function(graph, repo, canonical_loc.guid)
    redundant_fn = _fetch_function(graph, repo, redundant_loc.guid)
    if canonical_fn is None or redundant_fn is None:
        return Recommendation(
            summary="Cannot propose consolidation",
            rationale="One or both functions in the finding could not be found in the graph.",
        )

    canonical_src = _read_slice(repo_root, canonical_fn)
    redundant_src = _read_slice(repo_root, redundant_fn)
    if canonical_src is None or redundant_src is None:
        return Recommendation(
            summary="Cannot propose consolidation",
            rationale="Could not read source for one or both functions on disk.",
        )

    language = _infer_language(graph, repo, canonical_fn)
    prompt = _CONSOLIDATION_PROMPT.format(
        canonical_path=canonical_fn.module_path,
        canonical_name=canonical_fn.name,
        canonical_start=canonical_fn.line,
        canonical_end=canonical_fn.end_line,
        canonical_source=canonical_src,
        redundant_path=redundant_fn.module_path,
        redundant_name=redundant_fn.name,
        redundant_start=redundant_fn.line,
        redundant_end=redundant_fn.end_line,
        redundant_source=redundant_src,
        language=language,
    )
    raw = llm.complete(prompt, **(llm_options or {"temperature": 0.0, "num_predict": 800}))
    diff = _extract_diff(raw)

    summary = (
        f"Consolidate `{redundant_fn.name}` into `{canonical_fn.name}`"
        if diff else
        f"LLM declined to produce a consolidation for `{canonical_fn.name}`"
        f" / `{redundant_fn.name}`"
    )
    rationale = (
        f"LLM generated a unified diff that removes `{redundant_fn.name}` and "
        f"keeps `{canonical_fn.name}` as the canonical. Apply only after "
        f"reviewing — the diff was produced from snippet context, not the "
        f"whole repo."
        if diff else
        "The LLM was unable to safely produce a consolidation diff for this pair. "
        "Possible causes: signatures differ, side effects differ, or the model "
        "wasn't confident. Review the pair manually."
    )
    return Recommendation(
        summary=summary,
        rationale=rationale,
        proposed_diff=diff or None,
        blast_radius=[canonical_fn.guid, redundant_fn.guid],
        suggested_tests=[
            f"grep -rn '\\b{redundant_fn.name}\\b' .",
            "run the test suite after applying the diff",
        ],
    )


def review_change(
    *,
    diff: str,
    related_findings: list[Finding] | None = None,
    llm: LLMClient,
    llm_options: dict[str, Any] | None = None,
) -> Review:
    """Ask the LLM to review a unified diff.

    Returns a structured Review. If the LLM's response doesn't parse
    as JSON, we fall back to a "comment" verdict with the raw text
    as `summary` — never raises just because the model went off-script.
    """
    findings_context = ""
    if related_findings:
        rendered = "\n".join(
            f"  - {f.kind}: {f.evidence.get('function_name') or f.locations[0].path}"
            for f in related_findings
        )
        findings_context = (
            "Related findings to consider:\n" + rendered + "\n\n"
            "Use these to populate `addresses_findings` if the diff resolves any.\n"
        )
    prompt = _REVIEW_PROMPT.format(diff=diff, findings_context=findings_context)
    raw = llm.complete(prompt, **(llm_options or {"temperature": 0.0, "num_predict": 600}))
    return _parse_review(raw)


# ── helpers ─────────────────────────────────────────────────────────────


def _lines(loc) -> int:
    if loc.line is None or loc.end_line is None:
        return 0
    return loc.end_line - loc.line


def _fetch_function(
    graph: GraphBackend, repo: str, guid: str | None,
) -> FunctionRecord | None:
    if not guid:
        return None
    return graph.get_function(repo, guid)


def _read_slice(repo_root: Path, fn: FunctionRecord) -> str | None:
    try:
        source = (repo_root / fn.module_path).read_bytes()
    except OSError:
        return None
    lines = source.splitlines(keepends=True)
    start = max(0, fn.line - 1)
    end = min(len(lines), fn.end_line)
    return b"".join(lines[start:end]).decode("utf-8", errors="replace")


def _infer_language(graph: GraphBackend, repo: str, fn: FunctionRecord) -> str:
    module = graph.get_module(repo, fn.module_path)
    if module is None:
        return "text"
    return module.language.value if hasattr(module.language, "value") else str(module.language)


_FENCE_RE = re.compile(r"```(?:diff|patch)?\n([\s\S]*?)\n```")


def _extract_diff(raw: str) -> str:
    """Pull the unified diff out of a model response.

    Models sometimes wrap diffs in fences despite being told not to.
    We strip a fenced block if present; otherwise we return the raw
    text trimmed. An empty response means "no consolidation possible"
    per the prompt contract.
    """
    if not raw or not raw.strip():
        return ""
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _parse_review(raw: str) -> Review:
    """Parse the LLM's JSON review; fall back to a comment-shape on failure."""
    text = raw.strip()
    # Strip fenced JSON if present.
    m = re.match(r"```(?:json)?\n([\s\S]*?)\n```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Review(
            summary=text[:200] if text else "(empty LLM response)",
            overall="comment",
        )
    if not isinstance(data, dict):
        return Review(summary=str(data)[:200], overall="comment")
    overall = data.get("overall")
    if overall not in ("approve", "request_changes", "comment"):
        overall = "comment"
    return Review(
        summary=str(data.get("summary", "")),
        overall=overall,
        addresses_findings=_str_list(data.get("addresses_findings")),
        new_risks=_str_list(data.get("new_risks")),
        blockers=_str_list(data.get("blockers")),
    )


def _str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x) for x in v if x is not None]
