"""Phase 7 — LLM-direct tier tests (propose_consolidation, review_change, describe).

Uses an LLM stub that returns canned responses keyed by what the
prompt contains. Real-LLM smoke tests against Qwen would be opt-in
behind --run-integration (not added here yet; the FakeLLM path
exercises every code branch).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.findings import Finding
from otter_docs.llm_direct import Review, _extract_diff, _parse_review
from otter_docs.models import Location


class _ScriptedLLM:
    """LLM stub that returns responses based on substring matches in prompts."""

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        self.rules = rules
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        for needle, response in self.rules:
            if needle in prompt:
                return response
        return ""


def _setup_redundancy_repo(tmp_path: Path):
    """Scan a 2-function repo and craft a redundancy finding by hand.

    Uses real scan() so guids match what the backend would produce.
    Saves us having to hand-craft a Finding that matches a real graph.
    """
    (tmp_path / "a.py").write_text(
        "def add_one(x):\n    return x + 1\n\n"
        "def increment(x):\n    return x + 1\n"
    )
    backend = SqliteBackend(":memory:", vector_dim=8)
    repo = Repo(tmp_path, backend=backend)
    repo.scan()
    fns = list(repo.graph.list_functions(repo.name))
    add_one = next(f for f in fns if f.name == "add_one")
    increment = next(f for f in fns if f.name == "increment")
    return repo, add_one, increment


# ── propose_consolidation ─────────────────────────────────────────────


def test_propose_consolidation_attaches_diff(tmp_path: Path):
    repo, add_one, increment = _setup_redundancy_repo(tmp_path)
    canned_diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -4,2 +4,0 @@\n"
        "-def increment(x):\n-    return x + 1\n"
    )
    llm = _ScriptedLLM([("Two functions in the same codebase", canned_diff)])
    finding = Finding(
        kind="redundancy.semantic_equivalence",
        confidence=0.95,
        locations=[
            Location(repo=repo.name, path="a.py", line=1, end_line=2, guid=add_one.guid),
            Location(repo=repo.name, path="a.py", line=4, end_line=5, guid=increment.guid),
        ],
        evidence={"canonical_guid": add_one.guid, "function_names": ["add_one", "increment"]},
    )
    rec = repo.propose_consolidation(finding, llm)
    assert rec.proposed_diff == canned_diff.strip()
    assert "add_one" in rec.summary
    assert "increment" in rec.summary
    assert set(rec.blast_radius) == {add_one.guid, increment.guid}
    repo.close()


def test_propose_consolidation_handles_empty_llm_response(tmp_path: Path):
    """LLM declines (empty response) → Recommendation with proposed_diff=None."""
    repo, add_one, increment = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("Two functions in the same codebase", "")])
    finding = Finding(
        kind="redundancy.semantic_equivalence",
        confidence=0.95,
        locations=[
            Location(repo=repo.name, path="a.py", line=1, end_line=2, guid=add_one.guid),
            Location(repo=repo.name, path="a.py", line=4, end_line=5, guid=increment.guid),
        ],
        evidence={"canonical_guid": add_one.guid},
    )
    rec = repo.propose_consolidation(finding, llm)
    assert rec.proposed_diff is None
    assert "declined" in rec.summary.lower() or "cannot" in rec.summary.lower()
    repo.close()


def test_propose_consolidation_strips_fenced_diff(tmp_path: Path):
    """Some models wrap diffs in ```diff blocks; we should unwrap them."""
    repo, add_one, increment = _setup_redundancy_repo(tmp_path)
    fenced = "```diff\n--- a/a.py\n+++ b/a.py\n@@ -4,2 +4,0 @@\n-def increment(x):\n-    return x + 1\n```"
    llm = _ScriptedLLM([("Two functions", fenced)])
    finding = Finding(
        kind="redundancy.semantic_equivalence",
        confidence=0.95,
        locations=[
            Location(repo=repo.name, path="a.py", line=1, end_line=2, guid=add_one.guid),
            Location(repo=repo.name, path="a.py", line=4, end_line=5, guid=increment.guid),
        ],
        evidence={"canonical_guid": add_one.guid},
    )
    rec = repo.propose_consolidation(finding, llm)
    assert rec.proposed_diff is not None
    assert "```" not in rec.proposed_diff
    assert "--- a/a.py" in rec.proposed_diff
    repo.close()


def test_propose_consolidation_rejects_non_redundancy_finding(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    finding = Finding(
        kind="dead_code", confidence=0.5,
        locations=[Location(repo=repo.name, path="a.py", line=1, end_line=2)],
    )
    with pytest.raises(ValueError, match="redundancy"):
        repo.propose_consolidation(finding, _ScriptedLLM([]))
    repo.close()


# ── review_change ─────────────────────────────────────────────────────


def test_review_change_parses_approved_json(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("reviewing a proposed change", """\
{"summary": "Removes redundant helper", "overall": "approve",
 "addresses_findings": ["redundancy.semantic_equivalence"],
 "new_risks": [], "blockers": []}""")])
    review = repo.review_change("--- a/a.py\n+++ b/a.py\n", llm)
    assert isinstance(review, Review)
    assert review.overall == "approve"
    assert "redundancy.semantic_equivalence" in review.addresses_findings
    repo.close()


def test_review_change_falls_back_on_unparseable_response(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("reviewing a proposed change", "looks fine to me, no concerns")])
    review = repo.review_change("--- a/a.py\n", llm)
    assert review.overall == "comment"
    assert review.summary  # non-empty
    repo.close()


def test_review_change_normalizes_bad_overall_to_comment(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("reviewing a proposed change",
                         '{"summary": "?", "overall": "lgtm-with-changes"}')])
    review = repo.review_change("--- a/a.py\n", llm)
    assert review.overall == "comment"
    repo.close()


def test_review_change_strips_fenced_json(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("reviewing a proposed change",
                         '```json\n{"summary": "ok", "overall": "approve"}\n```')])
    review = repo.review_change("--- a/a.py\n", llm)
    assert review.overall == "approve"
    repo.close()


def test_review_change_includes_findings_context(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("reviewing a proposed change", '{"summary":"ok","overall":"approve"}')])
    findings = [Finding(
        kind="dead_code", confidence=0.5,
        locations=[Location(repo=repo.name, path="a.py", line=1, end_line=2)],
        evidence={"function_name": "ghost"},
    )]
    repo.review_change("--- a/a.py\n", llm, related_findings=findings)
    assert "dead_code" in llm.calls[0]
    assert "ghost" in llm.calls[0]
    repo.close()


# ── describe ──────────────────────────────────────────────────────────


def test_repo_describe_by_guid(tmp_path: Path):
    repo, add_one, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("function", "FAKE_FUNCTION_DESC")])
    desc = repo.describe(llm, guid=add_one.guid)
    assert desc is not None
    assert desc.text == "FAKE_FUNCTION_DESC"
    repo.close()


def test_repo_describe_by_path(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    llm = _ScriptedLLM([("module", "FAKE_MODULE_DESC")])
    desc = repo.describe(llm, path="a.py")
    assert desc is not None
    assert desc.text == "FAKE_MODULE_DESC"
    repo.close()


def test_repo_describe_returns_none_for_unknown_guid(tmp_path: Path):
    repo, _, _ = _setup_redundancy_repo(tmp_path)
    assert repo.describe(_ScriptedLLM([]), guid="nope") is None
    repo.close()


def test_repo_describe_requires_exactly_one_selector(tmp_path: Path):
    repo, fn, _ = _setup_redundancy_repo(tmp_path)
    with pytest.raises(ValueError):
        repo.describe(_ScriptedLLM([]))
    with pytest.raises(ValueError):
        repo.describe(_ScriptedLLM([]), guid=fn.guid, path="a.py")
    repo.close()


# ── unit helpers ──────────────────────────────────────────────────────


def test_extract_diff_handles_fenced_diff():
    assert _extract_diff("```diff\n---\n+++\n```") == "---\n+++"
    assert _extract_diff("```patch\nhello\n```") == "hello"


def test_extract_diff_returns_empty_for_empty_input():
    assert _extract_diff("") == ""
    assert _extract_diff("   \n") == ""


def test_parse_review_handles_partial_json():
    review = _parse_review('{"summary":"x","overall":"approve","new_risks":["a","b"]}')
    assert review.overall == "approve"
    assert review.new_risks == ["a", "b"]
    assert review.blockers == []  # missing field defaults to empty


def test_parse_review_coerces_non_string_list_items():
    review = _parse_review('{"summary":"x","overall":"comment","blockers":[1,null,"x"]}')
    # `None` is dropped; ints coerced to strings.
    assert "1" in review.blockers
    assert "x" in review.blockers
    assert None not in review.blockers
