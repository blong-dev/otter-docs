"""Tests for LicenseDetector."""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.license import LicenseDetector


def _detect(tmp_path: Path):
    return LicenseDetector().detect(repo="t", repo_root=tmp_path)


def test_no_license_file_returns_none(tmp_path: Path):
    (tmp_path / "README.md").write_text("hi")
    assert _detect(tmp_path) is None


def test_mit_license(tmp_path: Path):
    (tmp_path / "LICENSE").write_text(
        "MIT License\n\nCopyright (c) 2026\n\n"
        "Permission is hereby granted, free of charge, to any person\n"
        "obtaining a copy of this software…\n"
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.path == "LICENSE"
    assert surface.spdx_id == "MIT"
    assert "MIT" in surface.header_summary


def test_apache_license(tmp_path: Path):
    (tmp_path / "LICENSE.txt").write_text(
        "                                 Apache License\n"
        "                           Version 2.0, January 2004\n"
        "                        http://www.apache.org/licenses/\n"
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.spdx_id == "Apache-2.0"


def test_unknown_license_keeps_summary(tmp_path: Path):
    (tmp_path / "LICENSE").write_text("Custom internal license. All rights reserved.\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.spdx_id is None
    assert "Custom internal license" in surface.header_summary


def test_copying_filename_works(tmp_path: Path):
    (tmp_path / "COPYING").write_text(
        "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.path == "COPYING"
    assert surface.spdx_id == "GPL-3.0"


def test_case_insensitive_filename(tmp_path: Path):
    (tmp_path / "license").write_text("MIT License\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.path == "license"
    assert surface.spdx_id == "MIT"
