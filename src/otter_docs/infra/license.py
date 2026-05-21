"""LicenseDetector — find a license file in the repo root.

Best-effort SPDX identification: matches against a small list of
known headers (MIT, Apache-2.0, GPL family, BSD family, MPL, ISC,
AGPL, Unlicense). When no header matches, `spdx_id` is None and the
renderer falls back to displaying `header_summary` only.

We don't ship a license classifier model — the universal layer
stays pure-filesystem + cheap.
"""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.base import register_infra_detector
from otter_docs.infra.models import License

# Case-insensitive lookup. First match wins; the order is convention
# (a repo with both LICENSE and LICENSE.md is unusual, but if seen
# we'd prefer the plain filename).
_CANDIDATE_NAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.md",
    "COPYING.txt",
)

# Substring patterns that indicate a known SPDX license. Lowercased
# header text → SPDX id. Order matters when patterns overlap (Apache
# header mentions GPL incompatibility, GPL header mentions Apache, …);
# more-specific patterns come first.
_SPDX_PATTERNS: list[tuple[str, str]] = [
    ("apache license", "Apache-2.0"),
    ("mozilla public license", "MPL-2.0"),
    ("gnu affero general public license", "AGPL-3.0"),
    ("gnu lesser general public license", "LGPL-3.0"),
    ("gnu general public license", "GPL-3.0"),
    ("eclipse public license", "EPL-2.0"),
    ("the unlicense", "Unlicense"),
    ("isc license", "ISC"),
    ('"the mit license"', "MIT"),
    ("mit license", "MIT"),
    ("permission is hereby granted, free of charge", "MIT"),
    ("redistribution and use in source and binary", "BSD"),
    ("creative commons", "CC"),
]


class LicenseDetector:
    kind = "license"

    def detect(self, *, repo: str, repo_root: Path) -> License | None:
        path = _find_license_file(repo_root)
        if path is None:
            return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        rel = path.relative_to(repo_root).as_posix()
        header_summary = _first_nonblank_line(text)
        spdx = _match_spdx(text)
        return License(
            repo=repo,
            path=rel,
            spdx_id=spdx,
            header_summary=header_summary[:140],
        )


def _find_license_file(repo_root: Path) -> Path | None:
    # Build a case-insensitive lookup of root-level files so the user
    # doesn't have to use exact case (some repos ship `license`).
    lower_map = {
        entry.name.lower(): entry
        for entry in repo_root.iterdir()
        if entry.is_file()
    } if repo_root.is_dir() else {}
    for name in _CANDIDATE_NAMES:
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _first_nonblank_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _match_spdx(text: str) -> str | None:
    lower = text.lower()
    for needle, spdx in _SPDX_PATTERNS:
        if needle in lower:
            return spdx
    return None


register_infra_detector(LicenseDetector())
