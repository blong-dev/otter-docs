"""Procedural onboarding tests — manifest, idempotent onboard, status,
degradation, systemd emit. All run offline (no LLM/embedder): enrich
degrades cleanly when the endpoint is unreachable, which is exactly
the path we assert.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from otter_docs.onboarding import (
    Manifest,
    ModelConfig,
    RepoEntry,
    collect_status,
    load_manifest,
    onboard_all,
    onboard_repo,
    status_is_healthy,
    systemd_units,
)


def _write_manifest(tmp_path: Path, repos: list[tuple[str, Path, bool]]) -> Path:
    lines = [
        "[defaults]",
        'llm_url = "http://127.0.0.1:9"',     # unreachable on purpose
        'embed_url = "http://127.0.0.1:9"',
        'embed_model = "x"',
        'llm_model = "x"',
    ]
    for name, path, enrich in repos:
        lines += [
            "[[repo]]",
            f'name = "{name}"',
            f'path = "{path}"',
            f"enrich = {str(enrich).lower()}",
            'install_hooks = false',
        ]
    p = tmp_path / "repos.toml"
    p.write_text("\n".join(lines))
    return p


def _py_repo(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "a.py").write_text(
        "def used():\n    return 1\n\n"
        "def caller():\n    return used()\n\n"
        "def orphan():\n    return 9\n"
    )
    return d


# ── manifest ────────────────────────────────────────────────────────────


def test_load_manifest_roundtrip(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    mpath = _write_manifest(tmp_path, [("r1", r, True)])
    m = load_manifest(mpath)
    assert isinstance(m, Manifest)
    assert len(m.repos) == 1
    assert m.repos[0].name == "r1"
    assert m.repos[0].enrich is True
    assert m.defaults.llm_url == "http://127.0.0.1:9"


def test_env_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("OTTER_LLM_URL", "http://env-host:1234")
    monkeypatch.setenv("OTTER_EMBED_DIM", "1024")
    cfg = ModelConfig(llm_url="http://manifest:1", embed_dim=768)
    eff = cfg.with_env_overrides()
    assert eff.llm_url == "http://env-host:1234"
    assert eff.embed_dim == 1024


def test_per_repo_model_override():
    m = Manifest(
        defaults=ModelConfig(llm_model="default-m"),
        repos=[RepoEntry(name="x", path="/x",
                          models=ModelConfig(llm_model="special-m"))],
    )
    assert m.model_for(m.repos[0]).llm_model == "special-m"


# ── onboard (idempotent + graceful degradation) ─────────────────────────


def test_onboard_structural_only_succeeds_offline(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    entry = RepoEntry(name="r1", path=str(r), enrich=False,
                      install_hooks=False)
    res = onboard_repo(entry, ModelConfig())
    assert res.ok is True
    assert res.scanned == 1
    assert res.findings > 0          # orphan() → dead_code
    assert res.enriched is False
    # Heartbeat written.
    hb = r / ".otter-docs" / "status.json"
    assert hb.exists()
    assert json.loads(hb.read_text())["ok"] is True


def test_onboard_enrich_degrades_when_endpoint_unreachable(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    entry = RepoEntry(name="r1", path=str(r), enrich=True,
                      install_hooks=False)
    # ModelConfig points at a closed port → enrich must degrade, NOT fail.
    res = onboard_repo(entry, ModelConfig(llm_url="http://127.0.0.1:9",
                                          embed_url="http://127.0.0.1:9"))
    assert res.ok is True            # structural tier still succeeded
    assert res.enriched is False
    assert any("enrich skipped" in d for d in res.degradations)


def test_onboard_missing_path_is_error_not_crash(tmp_path: Path):
    entry = RepoEntry(name="ghost", path=str(tmp_path / "nope"),
                       enrich=False)
    res = onboard_repo(entry, ModelConfig())
    assert res.ok is False
    assert any("does not exist" in e for e in res.errors)


def test_onboard_is_idempotent(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    entry = RepoEntry(name="r1", path=str(r), enrich=False,
                      install_hooks=False)
    a = onboard_repo(entry, ModelConfig())
    b = onboard_repo(entry, ModelConfig())
    # Same structural counts on a re-run (upsert + marker injection).
    assert (a.scanned, a.findings) == (b.scanned, b.findings)
    doc = (r / "SYSTEM.md").read_text()
    # Inject human prose, re-onboard, prose survives (marker contract).
    (r / "SYSTEM.md").write_text(doc + "\n\nHUMAN NOTE keep me\n")
    onboard_repo(entry, ModelConfig())
    assert "HUMAN NOTE keep me" in (r / "SYSTEM.md").read_text()


def test_onboard_all_filters_and_forces_no_enrich(tmp_path: Path):
    r1 = _py_repo(tmp_path, "r1")
    r2 = _py_repo(tmp_path, "r2")
    m = Manifest(repos=[
        RepoEntry(name="r1", path=str(r1), enrich=True, install_hooks=False),
        RepoEntry(name="r2", path=str(r2), enrich=True, install_hooks=False),
    ])
    only = onboard_all(m, only="r1", enrich=False)
    assert len(only) == 1 and only[0].name == "r1"
    assert only[0].enriched is False   # forced structural-only


# ── status / heartbeat ──────────────────────────────────────────────────


def test_status_flags_never_onboarded(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    m = Manifest(repos=[RepoEntry(name="r1", path=str(r), enrich=False)])
    st = collect_status(m)
    assert st[0].present is False
    assert st[0].stale is True
    assert not status_is_healthy(st)


def test_status_healthy_after_onboard(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    entry = RepoEntry(name="r1", path=str(r), enrich=False,
                      install_hooks=False)
    onboard_repo(entry, ModelConfig())
    m = Manifest(repos=[entry])
    st = collect_status(m)
    assert st[0].present is True
    assert st[0].ok is True
    assert st[0].stale is False
    assert status_is_healthy(st) is True


def test_status_detects_staleness(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    entry = RepoEntry(name="r1", path=str(r), enrich=False,
                      install_hooks=False)
    onboard_repo(entry, ModelConfig())
    m = Manifest(repos=[entry])
    # Zero-second staleness window → the just-written heartbeat is stale.
    st = collect_status(m, stale_after_seconds=0)
    assert st[0].stale is True
    assert not status_is_healthy(st)


def test_status_handles_corrupt_heartbeat(tmp_path: Path):
    r = _py_repo(tmp_path, "r1")
    (r / ".otter-docs").mkdir(parents=True)
    (r / ".otter-docs" / "status.json").write_text("{not json")
    m = Manifest(repos=[RepoEntry(name="r1", path=str(r), enrich=False)])
    st = collect_status(m)
    assert st[0].stale is True
    assert any("unreadable" in e for e in st[0].errors)


# ── systemd ─────────────────────────────────────────────────────────────


def test_systemd_units_shape():
    units = systemd_units(manifest_path="/etc/otter/repos.toml",
                          user="b", on_calendar="*-*-* 03:30:00")
    assert set(units) == {"otter-docs-onboard.service",
                          "otter-docs-onboard.timer"}
    svc = units["otter-docs-onboard.service"]
    assert "User=b" in svc
    assert "onboard --manifest /etc/otter/repos.toml" in svc
    assert "Type=oneshot" in svc
    timer = units["otter-docs-onboard.timer"]
    assert "OnCalendar=*-*-* 03:30:00" in timer
    assert "Persistent=true" in timer


# ── CLI surface ─────────────────────────────────────────────────────────


def test_cli_onboard_and_status(tmp_path: Path, capsys):
    from otter_docs.cli import main

    r = _py_repo(tmp_path, "r1")
    mpath = _write_manifest(tmp_path, [("r1", r, False)])
    rc = main(["onboard", "--manifest", str(mpath)])
    assert rc == 0
    assert "[ok] r1" in capsys.readouterr().out
    rc = main(["status", "--manifest", str(mpath)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cli_status_nonzero_when_unhealthy(tmp_path: Path, capsys):
    from otter_docs.cli import main

    r = _py_repo(tmp_path, "r1")
    mpath = _write_manifest(tmp_path, [("r1", r, False)])
    # No onboard run → no heartbeat → status must exit non-zero.
    rc = main(["status", "--manifest", str(mpath)])
    assert rc == 1


def test_cli_systemd_emits_files(tmp_path: Path):
    from otter_docs.cli import main

    mpath = _write_manifest(tmp_path, [("r1", tmp_path / "r1", False)])
    rc = main(["systemd", "--manifest", str(mpath),
               "--out-dir", str(tmp_path / "units"), "--user", "b"])
    assert rc == 0
    assert (tmp_path / "units" / "otter-docs-onboard.timer").exists()


@pytest.mark.parametrize("missing", ["repo", "path", "name"])
def test_manifest_requires_core_fields(tmp_path: Path, missing):
    base = {
        "name": 'name = "r"',
        "path": f'path = "{tmp_path}"',
    }
    body = "[[repo]]\n"
    if missing != "repo":
        body += "\n".join(v for k, v in base.items() if k != missing) + "\n"
    else:
        body = ""  # no [[repo]] at all → empty manifest, not an error
    p = tmp_path / "m.toml"
    p.write_text(body)
    if missing == "repo":
        assert load_manifest(p).repos == []
    else:
        with pytest.raises(KeyError):
            load_manifest(p)
