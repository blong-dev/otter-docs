"""OD-1 — redundancy_report renders confirmed duplicates only.

Covers the model-agnostic verdict read (`get_any`) and the renderer's
partitioning of high-recall candidates into confirmed / reclassified /
unconfirmed, so the report stops listing raw pairs like
"Consolidate `envOr` into `envOr`".
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from otter_docs.llm_direct import RedundancyVerdict
from otter_docs.render.renderers import RedundancyReportRenderer
from otter_docs.verdictcache import (
    InMemoryRedundancyCache,
    SqliteRedundancyCache,
)


def _dup(conf: float = 0.9) -> RedundancyVerdict:
    return RedundancyVerdict(
        is_duplicate=True, confidence=conf, kind="duplicate", reason="same body"
    )


def _sibling() -> RedundancyVerdict:
    return RedundancyVerdict(
        is_duplicate=False, confidence=0.4, kind="sibling", reason="sibling methods"
    )


# ── get_any: model-agnostic read ───────────────────────────────────────────


def test_get_any_returns_verdict_regardless_of_model_inmemory():
    c = InMemoryRedundancyCache()
    c.put("h1", llm_model="qwen-9b", verdict=_dup())
    # A reader with no LLM handle (no model) still sees the verdict.
    v = c.get_any("h1")
    assert v is not None and v.is_duplicate
    assert c.get_any("missing") is None


def test_get_any_sqlite_returns_most_recent():
    conn = sqlite3.connect(":memory:")
    c = SqliteRedundancyCache(conn)
    c.put("h1", llm_model="model-a", verdict=_sibling())
    c.put("h1", llm_model="model-b", verdict=_dup(0.95))
    v = c.get_any("h1")
    assert v is not None and v.is_duplicate and v.confidence == 0.95
    assert c.get_any("nope") is None


# ── renderer: confirmed only, no "X into X" noise ───────────────────────────


class _FakeFinding:
    def __init__(self, names, paths):
        self.evidence = {"function_names": names}
        self.locations = [
            SimpleNamespace(path=p, line=i + 1, guid=f"g{i}")
            for i, p in enumerate(paths)
        ]
        self.confidence = 0.7


class _FakeRepo:
    """Minimal repo surface the renderer touches."""

    def __init__(self, findings, verdicts):
        self._findings = findings
        self._verdicts = verdicts  # id(finding) -> verdict | None

    def findings(self, kinds=None):
        return list(self._findings)

    def cached_redundancy_verdict(self, finding):
        return self._verdicts.get(id(finding))


def test_renderer_lists_confirmed_and_summarises_the_rest():
    dup = _FakeFinding(["load_cfg", "read_cfg"], ["a.py", "b.py"])
    sib = _FakeFinding(["save", "save"], ["c.py", "d.py"])
    pending = _FakeFinding(["envOr", "envOr"], ["bus.go", "gw.go"])
    repo = _FakeRepo(
        [dup, sib, pending],
        {id(dup): _dup(0.88), id(sib): _sibling(), id(pending): None},
    )
    out = RedundancyReportRenderer().render(repo)

    # The confirmed pair is listed with a real consolidation target.
    assert "1 confirmed duplicate pair(s)." in out
    assert "load_cfg" in out and "`a.py`:1" in out and "`b.py`:2" in out
    assert "Consolidate the two copies" in out
    # Reclassified + unconfirmed are summarised, not listed as redundant.
    assert "1 candidate(s) reclassified" in out
    assert "1 candidate(s) pending confirmation" in out
    # The old high-recall noise is gone.
    assert "likely-redundant pairs" not in out
    assert "into **" not in out  # no "Consolidate X into X"


def test_renderer_no_confirmed_says_so():
    pending = _FakeFinding(["envOr", "envOr"], ["bus.go", "gw.go"])
    repo = _FakeRepo([pending], {id(pending): None})
    out = RedundancyReportRenderer().render(repo)
    assert "No confirmed duplicates." in out
    assert "1 candidate(s) pending confirmation" in out
    assert "likely-redundant pairs" not in out
