"""Tests for TestsDetector."""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.tests import TestsDetector


def _detect(tmp_path: Path):
    return TestsDetector().detect(repo="t", repo_root=tmp_path)


def test_empty_repo_returns_none(tmp_path: Path):
    assert _detect(tmp_path) is None


def test_pytest_layout(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_alpha.py").write_text("def test_x(): pass\n")
    (tmp_path / "tests" / "test_beta.py").write_text("def test_y(): pass\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.dirs == ["tests"]
    assert surface.file_count == 2
    assert "pytest" in surface.runners


def test_pyproject_pytest_ini_options_alone(tmp_path: Path):
    """`[tool.pytest.ini_options]` in pyproject.toml is enough to flag pytest."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n[tool.pytest.ini_options]\nasyncio_mode = "auto"\n'
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert "pytest" in surface.runners


def test_jest_layout(tmp_path: Path):
    (tmp_path / "__tests__").mkdir()
    (tmp_path / "__tests__" / "a.test.ts").write_text("test('x', () => {});\n")
    (tmp_path / "jest.config.js").write_text("module.exports = {};\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert "jest" in surface.runners
    assert "jest-or-vitest" not in surface.runners
    assert surface.file_count == 1


def test_vitest_layout(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "core.test.ts").write_text("test('x', () => {});\n")
    (tmp_path / "vitest.config.ts").write_text("export default {};\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert "vitest" in surface.runners


def test_go_test_files(tmp_path: Path):
    (tmp_path / "core_test.go").write_text("package x\n")
    (tmp_path / "util_test.go").write_text("package x\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.file_count == 2
    assert "go test" in surface.runners


def test_specs_dir_without_test_files_excluded(tmp_path: Path):
    """A `specs/` directory containing only markdown is NOT a test dir.
    Bug found on telekora: docs/specs/ was incorrectly flagged because
    `specs` was in the dir-name allowlist. Fix requires the dir to
    contain at least one matched test file."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "specs").mkdir()
    (tmp_path / "docs" / "specs" / "feature.md").write_text("# spec\n")
    (tmp_path / "docs" / "specs" / "another.md").write_text("# spec\n")
    surface = _detect(tmp_path)
    # No test files anywhere — detector returns None.
    assert surface is None


def test_specs_dir_with_real_test_files_still_works(tmp_path: Path):
    """Conversely, a `specs/` dir that does contain test files should
    still be reported."""
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "foo.spec.ts").write_text("test('x', () => {});\n")
    (tmp_path / "vitest.config.ts").write_text("export default {};\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert "specs" in surface.dirs


def test_mixed_python_and_js(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x(): pass\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "a.spec.ts").write_text("test('x', () => {});\n")
    (tmp_path / "vitest.config.ts").write_text("export default {};\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert "pytest" in surface.runners
    assert "vitest" in surface.runners
    assert surface.file_count == 2
