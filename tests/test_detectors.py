"""Detector-level tests."""

from __future__ import annotations

from pathlib import Path

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.detectors import registry
from otter_docs.detectors.large_function import LargeFunctionDetector


def test_builtin_detectors_registered():
    """Importing detectors triggers their register() calls."""
    kinds = set(registry().keys())
    assert {"dead_code", "large_function", "empty_module"}.issubset(kinds)


def test_dead_code_flags_uncalled_function(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "def used():\n    return 1\n\n"
        "def caller():\n    return used()\n\n"
        "def orphan():\n    return 42\n"
    )
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"dead_code"})
        orphan_findings = [f for f in findings if f.evidence["function_name"] == "orphan"]
        assert len(orphan_findings) == 1
        # `used` is called by `caller`, so it shouldn't be flagged.
        assert not any(f.evidence["function_name"] == "used" for f in findings)


def test_dead_code_skips_entry_points(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "def main():\n    return 0\n\n"
        "def test_thing():\n    assert True\n\n"
        "def __init__():\n    pass\n"
    )
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"dead_code"})
        names = {f.evidence["function_name"] for f in findings}
        assert "main" not in names
        assert "test_thing" not in names
        assert "__init__" not in names


def test_dead_code_visibility_ranking(tmp_path: Path):
    """Three orphan functions at three visibility levels — confidence
    should rank private > public > public_export so downstream can
    sort instead of treating dead_code as a flat 0.5/0.75."""
    (tmp_path / "a.py").write_text(
        "def _hidden():\n    return 1\n\n"
        "def visible():\n    return 1\n\n"
        "def PublicExport():\n    return 1\n"
    )
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"dead_code"})
        by_name = {f.evidence["function_name"]: f for f in findings}
        assert by_name["_hidden"].evidence["visibility"] == "private"
        assert by_name["visible"].evidence["visibility"] == "public"
        assert by_name["PublicExport"].evidence["visibility"] == "public_export"
        # The ranking is the whole point: each tier must beat the next.
        assert (
            by_name["_hidden"].confidence
            > by_name["visible"].confidence
            > by_name["PublicExport"].confidence
        )


def test_large_function_threshold(tmp_path: Path):
    body = "    pass\n" * 90  # function ~91 lines
    (tmp_path / "long.py").write_text(f"def big():\n{body}\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"large_function"})
        assert len(findings) == 1
        assert findings[0].evidence["function_name"] == "big"
        assert findings[0].evidence["lines"] >= 80


def test_large_function_threshold_can_be_overridden():
    """Detector instance with a custom threshold flags shorter functions."""
    det = LargeFunctionDetector(threshold=5)
    assert det.threshold == 5


def test_large_function_cyclomatic_threshold(tmp_path: Path):
    """A short but branchy function trips the cyclomatic gate even when
    it falls under the line threshold."""
    branchy = (
        "def branchy(x, y, z):\n"
        "    if x > 0:\n"
        "        if y > 0:\n"
        "            return 1\n"
        "        elif y < 0:\n"
        "            return 2\n"
        "    elif x < 0:\n"
        "        if z > 0:\n"
        "            return 3\n"
        "        elif z < 0:\n"
        "            return 4\n"
        "    if x and y:\n"
        "        return 5\n"
        "    if x or z:\n"
        "        return 6\n"
        "    return 0\n"
    )
    (tmp_path / "a.py").write_text(branchy)
    # Use a low cyclomatic threshold to confirm the gate works; the
    # function above has CC well above 5 but only ~16 lines.
    det = LargeFunctionDetector(threshold=200, cyclomatic_threshold=5)
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = [f for f in det.run(repo.name, repo._backend)
                    if f.evidence["function_name"] == "branchy"]
        assert len(findings) == 1
        cc = findings[0].evidence["cyclomatic_complexity"]
        assert cc is not None and cc >= 5
        # Only cyclomatic tripped → moderate confidence
        assert findings[0].confidence == 0.6


def test_large_function_both_gates_confidence_bump(tmp_path: Path):
    """Long AND branchy → 0.85 confidence."""
    # Build a function that is both long and branchy.
    head = "def big(x):\n"
    branch_block = (
        "    if x > 0:\n"
        "        pass\n"
        "    elif x < 0:\n"
        "        pass\n"
    ) * 6  # ≥6 branches → CC≥7
    pad = "    pass\n" * 80
    (tmp_path / "long.py").write_text(head + branch_block + pad)
    det = LargeFunctionDetector(threshold=80, cyclomatic_threshold=5)
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = [f for f in det.run(repo.name, repo._backend)
                    if f.evidence["function_name"] == "big"]
        assert len(findings) == 1
        assert findings[0].confidence == 0.85
        assert findings[0].evidence["lines"] >= 80
        assert findings[0].evidence["cyclomatic_complexity"] >= 5


def test_empty_module_detection(tmp_path: Path):
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "real.py").write_text("def f(): pass\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"empty_module"})
        paths = {loc.path for f in findings for loc in f.locations}
        assert "__init__.py" in paths
        assert "real.py" not in paths


def test_empty_module_package_marker_is_low_confidence(tmp_path: Path):
    """An empty `__init__.py` (no imports, no docstring) is a structural
    package marker — emit at LIKELY_MARKER_CONFIDENCE so it's observable
    on the bus but never cards under a default subscriber threshold."""
    (tmp_path / "__init__.py").write_text("")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"empty_module"})
        assert len(findings) == 1
        f = findings[0]
        assert f.evidence["is_marker_likely"] is True
        assert f.confidence == 0.05


def test_empty_module_with_imports_is_default_confidence(tmp_path: Path):
    """An empty module that DOES import something is more interesting —
    it might be a re-export shim worth a glance. Stays at the default
    0.3 confidence so it surfaces above marker-only noise."""
    (tmp_path / "shim.py").write_text("import os  # noqa\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"empty_module"})
        assert len(findings) == 1
        f = findings[0]
        assert f.evidence["is_marker_likely"] is False
        assert f.confidence == 0.3


def test_findings_filter_by_cost_tier(tmp_path: Path):
    """cost_tiers={'static'} runs static detectors only.

    All built-ins are static today, so it's mostly a smoke test until
    Phase 6 lands an embedding-tier detector.
    """
    (tmp_path / "a.py").write_text("def x(): pass\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        static = repo.findings(cost_tiers={"static"})
        embed_only = repo.findings(cost_tiers={"embedding"})
        # No embedding-tier detectors registered yet → empty.
        assert embed_only == []
        # And some static findings exist (orphan function `x`).
        assert any(f.kind == "dead_code" for f in static)


def test_finding_carries_source_detector(tmp_path: Path):
    (tmp_path / "a.py").write_text("def x(): pass\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        for f in repo.findings():
            assert f.source_detector  # populated by every built-in detector


# ── findings_stream() — iterator API for bus-publishing ─────────────


def test_findings_stream_matches_findings_list(tmp_path: Path):
    """Whatever findings() returns, findings_stream() yields the same set."""
    (tmp_path / "a.py").write_text(
        "def x(): pass\n"
        "def y(): pass\n"
        "class C:\n    def m(self): pass\n"
    )
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        materialized = repo.findings()
        streamed = list(repo.findings_stream())
        assert {f.model_dump_json() for f in materialized} \
            == {f.model_dump_json() for f in streamed}


def test_findings_stream_is_lazy(tmp_path: Path):
    """A registered streaming detector must reach the consumer mid-flight —
    `next()` must yield before *every* detector has finished. Asserts the
    API is genuinely lazy, not a list comprehension in disguise."""
    from otter_docs.detectors import register as _register
    from otter_docs.detectors.base import _registry
    from otter_docs.findings import Finding
    from otter_docs.models import Location

    sentinel = "test.streamed_marker"

    class StreamFirstDetector:
        kind = sentinel
        cost_tier = "static"

        def run(self, repo: str, graph):
            return list(self.run_stream(repo, graph))

        def run_stream(self, repo: str, graph):
            yield Finding(
                kind=sentinel,
                confidence=1.0,
                locations=[Location(repo=repo, path="x", line=1)],
                source_detector="StreamFirstDetector",
            )
            # If the consumer takes only the first finding, the second
            # is never produced — that's the laziness contract.
            raise AssertionError("second yield should not be reached")

    saved = dict(_registry)
    try:
        _registry.clear()
        _register(StreamFirstDetector())
        with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
            repo.scan()
            stream = repo.findings_stream()
            first = next(iter(stream))
            assert first.kind == sentinel
            del stream
    finally:
        _registry.clear()
        _registry.update(saved)


def test_findings_stream_honors_filters(tmp_path: Path):
    """kinds= and cost_tiers= apply to the streaming form too."""
    (tmp_path / "a.py").write_text("def lonely(): pass\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        only_dead = list(repo.findings_stream(kinds={"dead_code"}))
        assert only_dead
        assert all(f.kind == "dead_code" for f in only_dead)
        none = list(repo.findings_stream(kinds={"nonexistent"}))
        assert none == []
