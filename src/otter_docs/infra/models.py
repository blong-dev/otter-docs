"""InfraSurface record types.

Pydantic models per spec §3.1. Each detector emits exactly one of
these (or None when its surface isn't present in the repo). The
`InfraSurface` union exists so the registry can type its return
shape; renderers narrow to the specific record they handle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ManifestFormat = Literal[
    "pyproject", "poetry", "package_json", "go_mod",
    "cargo", "gemfile", "pom", "gradle", "podfile",
    "composer", "mix", "pubspec", "requirements",
]


class Dependency(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version_spec: str | None = None
    source: str | None = None  # "registry" (default), "git+...", "path:..."


class DependencyManifest(BaseModel):
    """A repo's primary dependency manifest.

    `direct` is the user-facing dep list; `dev` covers test / build
    deps when the format makes the distinction. Formats that don't
    distinguish (e.g. plain `requirements.txt`) put everything in
    `direct` and leave `dev` empty.
    """

    model_config = ConfigDict(frozen=True)

    repo: str
    path: str  # repo-relative
    format: ManifestFormat
    direct: list[Dependency] = Field(default_factory=list)
    dev: list[Dependency] = Field(default_factory=list)


class License(BaseModel):
    """One license file in the repo root.

    `spdx_id` is best-effort: matched against a small list of known
    headers (MIT, Apache-2.0, GPL variants, BSD, MPL, ISC, AGPL). When
    no header matches, `spdx_id` is None and the renderer falls back
    to displaying `header_summary` only.
    """

    model_config = ConfigDict(frozen=True)

    repo: str
    path: str
    spdx_id: str | None = None
    header_summary: str = ""  # first non-blank line, truncated ~100 chars


class Readme(BaseModel):
    """The repo's README, if present.

    `summary` is the first non-heading paragraph — what a human reader
    would expect to see right under the title. `h2_outline` lists the
    `##` headings in document order so the renderer can build a quick
    table of contents.
    """

    model_config = ConfigDict(frozen=True)

    repo: str
    path: str
    summary: str = ""
    h2_outline: list[str] = Field(default_factory=list)


class TestsLayout(BaseModel):
    """Where tests live and how they're run.

    `dirs` is the test directories discovered (repo-relative);
    `file_count` is the count of test files across them;
    `runners` is the inferred test runners (e.g. {"pytest", "jest"}).
    A repo with both Python and JS tests gets one record with both
    runners.
    """

    model_config = ConfigDict(frozen=True)

    repo: str
    dirs: list[str] = Field(default_factory=list)
    file_count: int = 0
    runners: list[str] = Field(default_factory=list)  # sorted for stable rendering


class SourceLayoutEntry(BaseModel):
    """One top-level entry in the repo's source layout."""

    model_config = ConfigDict(frozen=True)

    name: str
    is_dir: bool
    annotation: str | None = None  # "source root", "tests", "docs", ...


class SourceLayout(BaseModel):
    """Top-level directory map with annotations on known patterns."""

    model_config = ConfigDict(frozen=True)

    repo: str
    entries: list[SourceLayoutEntry] = Field(default_factory=list)


InfraSurface = (
    DependencyManifest
    | License
    | Readme
    | TestsLayout
    | SourceLayout
)
