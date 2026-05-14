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


def test_empty_module_detection(tmp_path: Path):
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "real.py").write_text("def f(): pass\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        findings = repo.findings(kinds={"empty_module"})
        paths = {loc.path for f in findings for loc in f.locations}
        assert "__init__.py" in paths
        assert "real.py" not in paths


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
