"""OD-4 — confirm_description: LLM-judge doc-vs-code staleness.

The high-precision second pass over the description.divergence cosine signal.
Reads the docstring + body and rules accurate / partial / stale / wrong;
consumers gate on `is_stale`. Verdicts cache in graph.db (mirrors
confirm_redundancy).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.findings import Finding
from otter_docs.llm_direct import DescriptionVerdict, _parse_description_verdict
from otter_docs.models import Location
from otter_docs.verdictcache import InMemoryDescriptionVerdictCache


class _ScriptedLLM:
    """Returns a canned response when the prompt contains a needle."""

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        self.rules = rules
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        for needle, response in self.rules:
            if needle in prompt:
                return response
        return ""


_NEEDLE = "A function's docstring and its implementation"


def _setup(tmp_path: Path, body: str = None):
    src = body or (
        "def parse(text):\n"
        '    """Return the number of words in text."""\n'
        "    return len(text.split())\n"
    )
    (tmp_path / "m.py").write_text(src)
    repo = Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8))
    repo.scan()
    fn = next(iter(repo.graph.list_functions(repo.name)))
    return repo, fn


def _finding(repo, fn) -> Finding:
    return Finding(
        kind="description.divergence",
        confidence=0.7,
        locations=[
            Location(
                repo=repo.name,
                path=fn.module_path,
                line=fn.line,
                end_line=fn.end_line,
                guid=fn.guid,
            )
        ],
        evidence={"function_name": fn.name},
        source_detector="description.divergence",
    )


def _verdict_json(kind: str, is_stale: bool, conf: float, reason: str) -> str:
    import json

    return json.dumps(
        {"is_stale": is_stale, "confidence": conf, "kind": kind, "reason": reason}
    )


# ── the judge ───────────────────────────────────────────────────────────────


def test_confirm_description_stale(tmp_path: Path):
    repo, fn = _setup(tmp_path)
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("stale", True, 0.9, "Doc predates rewrite."))])
    v = repo.confirm_description(_finding(repo, fn), llm)
    assert v.kind == "stale" and v.is_stale is True and v.confidence == 0.9
    assert "rewrite" in v.reason
    repo.close()


def test_confirm_description_accurate(tmp_path: Path):
    repo, fn = _setup(tmp_path)
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("accurate", False, 0.95, "Matches."))])
    v = repo.confirm_description(_finding(repo, fn), llm)
    assert v.kind == "accurate" and v.is_stale is False
    repo.close()


def test_confirm_description_partial_is_not_stale(tmp_path: Path):
    # partial → is_stale False even if the model sloppily set the boolean true.
    repo, fn = _setup(tmp_path)
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("partial", True, 0.6, "omits error path"))])
    v = repo.confirm_description(_finding(repo, fn), llm)
    assert v.kind == "partial" and v.is_stale is False
    repo.close()


def test_confirm_description_caches_and_read_only_accessor(tmp_path: Path):
    repo, fn = _setup(tmp_path)
    llm = _ScriptedLLM([(_NEEDLE, _verdict_json("wrong", True, 0.8, "contradicts"))])
    f = _finding(repo, fn)
    repo.confirm_description(f, llm)
    repo.confirm_description(f, llm)  # second call hits the cache
    assert len(llm.calls) == 1
    cached = repo.cached_description_verdict(f)  # no LLM, model-agnostic
    assert cached is not None and cached.kind == "wrong" and cached.is_stale is True
    repo.close()


def test_confirm_description_no_docstring_is_accurate(tmp_path: Path):
    repo, fn = _setup(tmp_path, "def bare(x):\n    return x\n")
    v = repo.confirm_description(_finding(repo, fn), _ScriptedLLM([]))
    assert v.kind == "accurate" and v.is_stale is False and v.confidence == 0.0
    repo.close()


def test_confirm_description_rejects_non_description_finding(tmp_path: Path):
    repo, fn = _setup(tmp_path)
    bad = _finding(repo, fn).model_copy(update={"kind": "large_function"})
    with pytest.raises(ValueError):
        repo.confirm_description(bad, _ScriptedLLM([]))
    repo.close()


# ── parser + cache primitives ────────────────────────────────────────────────


def test_parse_re_derives_is_stale_from_kind():
    v = _parse_description_verdict(_verdict_json("accurate", True, 0.5, "x"))
    assert v.kind == "accurate" and v.is_stale is False  # forced false


def test_parse_unparseable_falls_back_to_accurate():
    v = _parse_description_verdict("this is not json")
    assert v.kind == "accurate" and v.is_stale is False and v.confidence == 0.0


def test_description_cache_get_any_model_agnostic():
    c = InMemoryDescriptionVerdictCache()
    c.put("h", llm_model="m1", verdict=DescriptionVerdict(
        is_stale=True, confidence=0.9, kind="stale", reason="r"))
    assert c.get_any("h").is_stale is True
    assert c.get_any("nope") is None
