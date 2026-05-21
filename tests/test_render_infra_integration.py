"""End-to-end: full fixture repo → OTTER.md via Repo.render_document.

This is the architectural smoke test: drop a realistic mini-repo on
disk, run scan + render_document, confirm all five infra sections
appear with sensible content, code-graph sections still work, and
markers are idempotent across reruns.
"""

from __future__ import annotations

from pathlib import Path

from otter_docs import Repo
from otter_docs.backends import SqliteBackend


def _build_fixture(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "samplelib"\nversion = "0.1.0"\n'
        'dependencies = ["pydantic>=2.6", "httpx ~=0.27"]\n'
        '[project.optional-dependencies]\ndev = ["pytest>=8"]\n'
        '[tool.pytest.ini_options]\nasyncio_mode = "auto"\n'
    )
    (root / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2026\n\n"
        "Permission is hereby granted, free of charge…\n"
    )
    (root / "README.md").write_text(
        "# samplelib\n\nA tiny library used as a fixture.\n\n"
        "## Install\n\n`pip install samplelib`\n\n"
        "## Usage\n\nSee the docs.\n"
    )
    (root / "src").mkdir()
    (root / "src" / "core.py").write_text("def add(x, y):\n    return x + y\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_core.py").write_text("def test_add(): assert True\n")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# guide\n")


def test_full_render_includes_all_infra_sections(tmp_path: Path):
    _build_fixture(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="sample", backend=backend) as r:
        r.scan()
        out = tmp_path / "OTTER.md"
        text = r.render_document(out)

    # Headings (with title-cased section names from the marker ids).
    assert "## Readme" in text
    assert "## Dependencies" in text
    assert "## License" in text
    assert "## Source Layout" in text
    assert "## Tests" in text

    # Markers populated (NOT empty / NOT the placeholder).
    for section in ("readme", "dependencies", "license", "source_layout", "tests"):
        begin = f"<!-- BEGIN GENERATED:{section} -->"
        end = f"<!-- END GENERATED:{section} -->"
        assert begin in text and end in text
        body = text.split(begin, 1)[1].split(end, 1)[0].strip()
        assert body and "*(not detected" not in body, (
            f"section {section!r} rendered empty or as placeholder"
        )

    # Sanity-check content (not exhaustive; per-detector tests cover specifics).
    assert "pydantic" in text
    assert "pytest" in text
    assert "MIT" in text  # SPDX
    assert "A tiny library used as a fixture" in text  # README summary
    assert "source root" in text  # src/ annotation

    # Section order: readme first, then deps/license/source_layout/tests,
    # then existing code-graph sections.
    indexes = {
        s: text.index(f"<!-- BEGIN GENERATED:{s} -->")
        for s in [
            "readme", "dependencies", "license", "source_layout", "tests",
            "system_overview", "findings_summary",
        ]
    }
    for a, b in zip(
        ["readme", "dependencies", "license", "source_layout", "tests",
         "system_overview"],
        ["dependencies", "license", "source_layout", "tests", "system_overview",
         "findings_summary"],
        strict=False,
    ):
        assert indexes[a] < indexes[b], f"{a} must come before {b}"


def test_render_omits_section_when_surface_missing(tmp_path: Path):
    """A repo without a license file still renders, license section
    body is the 'not detected' placeholder."""
    # Build a fixture missing the LICENSE file.
    (tmp_path / "README.md").write_text("# x\n\nsummary.\n")
    (tmp_path / "src.py").write_text("def f(): pass\n")
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="sample", backend=backend) as r:
        r.scan()
        out = tmp_path / "OTTER.md"
        text = r.render_document(out)
    # license section still rendered with placeholder
    body = text.split("<!-- BEGIN GENERATED:license -->", 1)[1].split(
        "<!-- END GENERATED:license -->", 1
    )[0]
    assert "*(not detected in this repo)*" in body


def test_rerender_is_idempotent(tmp_path: Path):
    _build_fixture(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="sample", backend=backend) as r:
        r.scan()
        out = tmp_path / "OTTER.md"
        text1 = r.render_document(out)
        text2 = r.render_document(out)
    assert text1 == text2


def test_rerender_preserves_user_prose(tmp_path: Path):
    """Content between markers is replaced; content outside is preserved."""
    _build_fixture(tmp_path)
    backend = SqliteBackend(":memory:", vector_dim=8)
    with Repo(tmp_path, name="sample", backend=backend) as r:
        r.scan()
        out = tmp_path / "OTTER.md"
        r.render_document(out)
        text = out.read_text()
        # Insert hand-written prose between two sections.
        marker = "<!-- END GENERATED:license -->"
        injected = text.replace(
            marker,
            marker + "\n\nHand-written notes — must survive re-render.\n",
        )
        out.write_text(injected)
        # Re-render.
        r.render_document(out)
        final = out.read_text()
    assert "Hand-written notes — must survive re-render." in final
