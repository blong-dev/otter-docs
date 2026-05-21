"""Tests for DependencyManifestDetector across formats."""

from __future__ import annotations

from pathlib import Path

from otter_docs.infra.dependencies import DependencyManifestDetector


def _detect(tmp_path: Path):
    return DependencyManifestDetector().detect(repo="t", repo_root=tmp_path)


def test_no_manifest_returns_none(tmp_path: Path):
    assert _detect(tmp_path) is None


def test_pyproject_pep621(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n'
        'dependencies = ["pydantic>=2.6", "httpx[http2] ~= 0.27 ; python_version >= \'3.10\'"]\n'
        '[project.optional-dependencies]\n'
        'dev = ["pytest>=8"]\n'
        'docs = ["sphinx"]\n'
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "pyproject"
    direct = {d.name: d.version_spec for d in surface.direct}
    assert direct["pydantic"] == ">=2.6"
    # extras + marker stripped, version preserved.
    assert direct["httpx"] is not None and "0.27" in direct["httpx"]
    dev = {d.name: d.version_spec for d in surface.dev}
    # `dev` is a known dev group; `docs` is also (per _looks_dev_group).
    assert "pytest" in dev
    assert "sphinx" in dev


def test_pyproject_poetry(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.poetry]\nname = "x"\nversion = "0.1.0"\n'
        '[tool.poetry.dependencies]\n'
        'python = "^3.11"\n'
        'pydantic = "^2.6"\n'
        'httpx = {version = "^0.27", extras = ["http2"]}\n'
        '[tool.poetry.group.dev.dependencies]\n'
        'pytest = "^8"\n'
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "poetry"
    direct = {d.name: d.version_spec for d in surface.direct}
    # python interpreter constraint should be excluded
    assert "python" not in direct
    assert direct["pydantic"] == "^2.6"
    assert direct["httpx"] == "^0.27"
    dev = {d.name: d.version_spec for d in surface.dev}
    assert dev["pytest"] == "^8"


def test_package_json(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"name": "x", "version": "1.0.0",'
        ' "dependencies": {"react": "^18.2.0", "next": "14.x"},'
        ' "devDependencies": {"vitest": "^1.0.0"}}'
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "package_json"
    direct = {d.name: d.version_spec for d in surface.direct}
    assert direct == {"react": "^18.2.0", "next": "14.x"}
    dev = {d.name: d.version_spec for d in surface.dev}
    assert dev == {"vitest": "^1.0.0"}


def test_go_mod(tmp_path: Path):
    (tmp_path / "go.mod").write_text(
        "module example.com/x\n\n"
        "go 1.22\n\n"
        "require github.com/spf13/cobra v1.8.0\n\n"
        "require (\n"
        "\tgithub.com/stretchr/testify v1.9.0\n"
        "\tgolang.org/x/sys v0.20.0 // indirect\n"
        ")\n"
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "go_mod"
    names = {d.name for d in surface.direct}
    assert "github.com/spf13/cobra" in names
    assert "github.com/stretchr/testify" in names
    assert "golang.org/x/sys" in names


def test_cargo(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "x"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[dependencies]\nserde = "1"\ntokio = {version = "1.36", features = ["full"]}\n\n'
        '[dev-dependencies]\nproptest = "1"\n'
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "cargo"
    direct = {d.name: d.version_spec for d in surface.direct}
    assert direct["serde"] == "1"
    assert direct["tokio"] == "1.36"
    assert {d.name for d in surface.dev} == {"proptest"}


def test_gemfile(tmp_path: Path):
    (tmp_path / "Gemfile").write_text(
        'source "https://rubygems.org"\n'
        'gem "rails", "~> 7.1.0"\n'
        'gem "puma"\n'
        'group :test do\n  gem "rspec", "~> 3"\nend\n'
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "gemfile"
    names = {d.name for d in surface.direct}
    assert {"rails", "puma", "rspec"}.issubset(names)


def test_requirements_txt_only(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text(
        "requests>=2.31\n"
        "click==8.1.7\n"
        "# a comment\n"
        "\n"
        "-e git+https://example.com/x.git#egg=x  # editable\n"
    )
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "requirements"
    direct = {d.name: d.version_spec for d in surface.direct}
    assert direct["requests"] == ">=2.31"
    assert direct["click"] == "==8.1.7"


def test_pyproject_wins_over_requirements(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["pydantic>=2"]\n'
    )
    (tmp_path / "requirements.txt").write_text("ignored>=1\n")
    surface = _detect(tmp_path)
    assert surface is not None
    assert surface.format == "pyproject"
    assert {d.name for d in surface.direct} == {"pydantic"}


def test_malformed_manifest_returns_none(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("not valid toml [[[")
    assert _detect(tmp_path) is None


def test_monorepo_subtree_manifests(tmp_path: Path):
    """No root manifest, multiple subtree manifests → list of records."""
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text(
        '{"dependencies": {"react": "^18"}}'
    )
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "package.json").write_text(
        '{"dependencies": {"express": "^4"}}'
    )
    (tmp_path / "service").mkdir()
    (tmp_path / "service" / "go.mod").write_text(
        "module example.com/service\n\ngo 1.22\n\nrequire github.com/spf13/cobra v1.8.0\n"
    )
    result = _detect(tmp_path)
    assert isinstance(result, list)
    paths = {m.path for m in result}
    assert paths == {"web/package.json", "api/package.json", "service/go.mod"}


def test_root_manifest_takes_precedence_over_subtrees(tmp_path: Path):
    """If a root manifest exists, subtree discovery is skipped — keeps
    the common single-package case simple."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "root"\ndependencies = ["root-dep"]\n'
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "package.json").write_text(
        '{"dependencies": {"sub-dep": "1"}}'
    )
    result = _detect(tmp_path)
    assert not isinstance(result, list)
    assert result.path == "pyproject.toml"
    assert {d.name for d in result.direct} == {"root-dep"}


def test_skips_node_modules_etc(tmp_path: Path):
    """Vendored/build manifests inside skip dirs must not contaminate
    monorepo discovery."""
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text(
        '{"dependencies": {"real": "1"}}'
    )
    # Vendored sub-package — must be ignored.
    (tmp_path / "web" / "node_modules").mkdir()
    (tmp_path / "web" / "node_modules" / "vendored").mkdir()
    (tmp_path / "web" / "node_modules" / "vendored" / "package.json").write_text(
        '{"dependencies": {"transitive": "1"}}'
    )
    result = _detect(tmp_path)
    assert isinstance(result, list)
    paths = {m.path for m in result}
    assert paths == {"web/package.json"}
