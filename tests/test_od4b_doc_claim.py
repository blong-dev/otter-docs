"""OD-4b — confirm_doc_description: hand-written docs (README/CLAUDE/specs) vs
the code they name.

The prose sibling of OD-4. `Repo.doc_findings()` reads `.md` off disk and emits
one `doc.divergence` candidate per (doc section, referenced symbol) via
deterministic mention matching (backtick spans / paths → graph symbols); the
LLM judge `confirm_doc_description` rules accurate/partial/stale/wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.docscan import (
    MAX_NAME_FANOUT,
    _extract_tokens,
    _MentionIndex,
    split_sections,
)
from otter_docs.findings import Finding


class _ScriptedLLM:
    def __init__(self, rules: list[tuple[str, str]]) -> None:
        self.rules = rules
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        for needle, response in self.rules:
            if needle in prompt:
                return response
        return ""


_NEEDLE = "A passage of hand-written documentation references"


def _verdict_json(kind: str, is_stale: bool, conf: float, reason: str) -> str:
    return json.dumps(
        {"is_stale": is_stale, "confidence": conf, "kind": kind, "reason": reason}
    )


def _repo_with(tmp_path: Path, code: str, docs: dict[str, str]) -> Repo:
    (tmp_path / "parser.py").write_text(code)
    for rel, text in docs.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    repo = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    repo.scan()
    return repo


_CODE = (
    "def calculate_total(items):\n"
    '    """Sum the prices of items."""\n'
    "    return sum(i.price for i in items)\n"
)


# ── section splitter ─────────────────────────────────────────────────────────


def test_split_sections_is_fence_aware():
    doc = (
        "# T\nintro\n\n## Body\nprose `sym`.\n\n"
        "```python\n# not a heading\ndef f(): pass\n```\n\nafter.\n\n## End\n"
    )
    secs = split_sections(doc)
    assert [s.heading for s in secs] == ["T", "Body", "End"]
    body = next(s for s in secs if s.heading == "Body")
    assert "after." in body.text  # fence didn't split the section


def test_extract_tokens_skips_fenced_code():
    text = "use `real_sym` and `path/to/x.py`.\n```\n`fenced_sym`\n```\n"
    toks = _extract_tokens(text)
    assert "real_sym" in toks and "path/to/x.py" in toks
    assert "fenced_sym" not in toks


# ── mention resolver ─────────────────────────────────────────────────────────


def test_doc_findings_resolves_backtick_symbol(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# `calculate_total`\n\nThe `calculate_total` helper sums item prices.\n"},
    )
    finds = repo.doc_findings()
    assert len(finds) == 1
    f = finds[0]
    assert f.kind == "doc.divergence"
    assert f.evidence["symbol_name"] == "calculate_total"
    assert f.evidence["symbol_kind"] == "function"
    assert f.evidence["doc_path"] == "README.md"
    assert f.locations[0].path == "README.md"
    repo.close()


def test_doc_findings_resolves_file_path_to_module(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"docs/specs/x.md": "# `parser.py`\n\nSee `parser.py` for the summation logic.\n"},
    )
    finds = repo.doc_findings()
    kinds = {f.evidence["symbol_kind"] for f in finds}
    assert "module" in kinds
    mod = next(f for f in finds if f.evidence["symbol_kind"] == "module")
    assert mod.evidence["symbol_path"] == "parser.py"
    repo.close()


def test_doc_findings_ignores_unresolvable_and_short_names(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# G\n\nThe `run` loop and `nonexistent_symbol` here.\n"},
    )
    # `run` is too short / not a symbol; `nonexistent_symbol` isn't in the graph
    assert repo.doc_findings() == []
    repo.close()


def test_in_passing_single_mention_is_not_a_candidate(tmp_path: Path):
    # `calculate_total` named once, not in the heading → reference in passing,
    # not a claim about it. Prominence gate (heading OR ≥2 mentions) drops it.
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# Overview\n\nWe call `calculate_total` somewhere.\n"},
    )
    assert repo.doc_findings() == []
    repo.close()


def test_prominence_via_repeated_mention(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# Money\n\n`calculate_total` sums prices; call "
                      "`calculate_total` per order.\n"},
    )
    finds = repo.doc_findings()
    assert len(finds) == 1 and finds[0].evidence["symbol_name"] == "calculate_total"
    repo.close()


def test_generated_doc_is_skipped(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# `calculate_total`\n\n<!-- generated -->\nAuto-generated. "
                      "`calculate_total` sums prices.\n"},
    )
    assert repo.doc_findings() == []  # generated-marker in head → skipped
    repo.close()


def test_dated_history_doc_is_skipped(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"docs/specs/audit-2026-05-18.md":
            "# `calculate_total`\n\n`calculate_total` summed prices back then.\n"},
    )
    assert repo.doc_findings() == []  # dated filename → append-only record
    repo.close()


def test_test_symbol_mention_is_excluded(tmp_path: Path):
    # a spec section that prominently names a TEST symbol is not the doc-audit
    # signal (OD-2 test/prod separation) — resolved test symbols are dropped.
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_thing.py").write_text(
        "def test_summation():\n    assert True\n"
    )
    (tmp_path / "parser.py").write_text(_CODE)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "specs").mkdir()
    (tmp_path / "docs" / "specs" / "s.md").write_text(
        "# `test_summation`\n\n`test_summation` checks the sum in "
        "`tests/test_thing.py`.\n"
    )
    repo = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    repo.scan()
    assert repo.doc_findings() == []
    repo.close()


def test_mention_index_drops_ambiguous_names(tmp_path: Path):
    # more than MAX_NAME_FANOUT functions named the same → unresolvable
    lines = "".join(
        f"class C{i}:\n    def handler(self):\n        return {i}\n"
        for i in range(MAX_NAME_FANOUT + 1)
    )
    repo = _repo_with(tmp_path, lines, {})
    idx = _MentionIndex(repo.name, repo.graph)
    assert idx.resolve("handler") is None  # ambiguous across many classes
    repo.close()


# ── the judge ────────────────────────────────────────────────────────────────


def test_confirm_doc_description_stale(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# `calculate_total`\n\nThe `calculate_total` helper returns the *average* price.\n"},
    )
    f = repo.doc_findings()[0]
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("wrong", True, 0.9, "sums, not averages"))])
    v = repo.confirm_doc_description(f, llm)
    assert v.kind == "wrong" and v.is_stale is True and v.confidence == 0.9
    # the prompt actually carried the section prose + the symbol's code
    assert "average" in llm.calls[0] and "sum(i.price" in llm.calls[0]
    repo.close()


def test_confirm_doc_description_accurate(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# `calculate_total`\n\nThe `calculate_total` helper sums item prices.\n"},
    )
    f = repo.doc_findings()[0]
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("accurate", False, 0.95, "matches"))])
    v = repo.confirm_doc_description(f, llm)
    assert v.kind == "accurate" and v.is_stale is False
    repo.close()


def test_confirm_doc_description_caches_and_read_only_accessor(tmp_path: Path):
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# `calculate_total`\n\n`calculate_total` returns a count.\n"},
    )
    f = repo.doc_findings()[0]
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("stale", True, 0.8, "drifted"))])
    repo.confirm_doc_description(f, llm)
    repo.confirm_doc_description(f, llm)  # cache hit
    assert len(llm.calls) == 1
    cached = repo.cached_doc_description_verdict(f)
    assert cached is not None and cached.kind == "stale" and cached.is_stale is True
    repo.close()


def test_confirm_doc_description_rejects_non_doc_finding(tmp_path: Path):
    repo = _repo_with(tmp_path, _CODE, {})
    bad = Finding(
        kind="description.divergence", confidence=0.5, locations=[],
        source_detector="x",
    )
    with pytest.raises(ValueError):
        repo.confirm_doc_description(bad, _ScriptedLLM([]))
    repo.close()


def test_doc_findings_excludes_missing_symbol_body_gracefully(tmp_path: Path):
    # a doc.divergence pointing at a guid not in the graph → accurate/no-op
    repo = _repo_with(
        tmp_path, _CODE,
        {"README.md": "# `calculate_total`\n\n`calculate_total` sums prices.\n"},
    )
    f = repo.doc_findings()[0].model_copy(
        update={"evidence": {**repo.doc_findings()[0].evidence,
                             "symbol_guid": "nonsense", "symbol_kind": "function"}}
    )
    v = repo.confirm_doc_description(f, _ScriptedLLM([]))
    assert v.kind == "accurate" and v.confidence == 0.0
    repo.close()
