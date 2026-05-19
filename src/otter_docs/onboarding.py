"""Procedural, automatic multi-repo onboarding.

The point: wiring N repos must be a *procedure*, not N remembered
invocations. A declarative manifest is the source of truth; one
idempotent command brings every listed repo to a known state; a
heartbeat makes silent breakage loud. This exists specifically so we
don't recreate the failure that started this project — automation
that rotted invisibly for 28 days because it had no declarative
config, no self-healing, and no observability.

Five pieces, all here:
  - Manifest         declarative `repos.toml` (stdlib tomllib).
  - ModelConfig      LLM/embed endpoints; env-overridable.
  - onboard_repo     idempotent: verify tooling → scan → resolve →
                     optional enrich (graceful skip if a model
                     endpoint is down) → render → install hooks.
                     Writes .otter-docs/status.json.
  - collect_status   reads every repo's heartbeat; flags staleness
                     and errors loudly.
  - systemd_units    timer/service text for the slow semantic pass
                     (user installs with sudo; we never auto-sudo).
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# ── manifest schema ─────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """LLM + embedder endpoints for the semantic (enrich) tier.

    OpenAI-compatible URLs (llama.cpp / vLLM / OpenAI). Env vars
    override the manifest so the same manifest works across machines:
    OTTER_LLM_URL / OTTER_LLM_MODEL / OTTER_EMBED_URL /
    OTTER_EMBED_MODEL / OTTER_EMBED_DIM.
    """

    model_config = ConfigDict(frozen=True)

    llm_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embed_url: str = "http://localhost:11435"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    def with_env_overrides(self) -> ModelConfig:
        return ModelConfig(
            llm_url=os.environ.get("OTTER_LLM_URL", self.llm_url),
            llm_model=os.environ.get("OTTER_LLM_MODEL", self.llm_model),
            embed_url=os.environ.get("OTTER_EMBED_URL", self.embed_url),
            embed_model=os.environ.get("OTTER_EMBED_MODEL", self.embed_model),
            embed_dim=int(os.environ.get("OTTER_EMBED_DIM", self.embed_dim)),
        )


class RepoEntry(BaseModel):
    """One repo's onboarding policy."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    # Run the semantic (enrich) tier? False = structural-only, which
    # is free and fast. Slow/expensive repos can opt out.
    enrich: bool = True
    # Generated document filename, relative to the repo root.
    doc: str = "SYSTEM.md"
    # Install git hooks during onboarding?
    install_hooks: bool = True
    # Optional per-repo model override (else manifest defaults).
    models: ModelConfig | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    defaults: ModelConfig = Field(default_factory=ModelConfig)
    repos: list[RepoEntry] = Field(default_factory=list)

    def model_for(self, entry: RepoEntry) -> ModelConfig:
        base = entry.models or self.defaults
        return base.with_env_overrides()


def load_manifest(path: str | Path) -> Manifest:
    """Parse a `repos.toml`.

    Shape:

        [defaults]
        llm_url = "http://localhost:11434"
        llm_model = "qwen3.5:9b"
        embed_url = "http://localhost:11435"
        embed_model = "nomic-embed-text"
        embed_dim = 768

        [[repo]]
        name = "v3"
        path = "/abs/path/to/v3"
        enrich = true
        doc = "docs/SYSTEM.md"

        [[repo]]
        name = "telekora"
        path = "/abs/path/to/telekora"
        enrich = false        # structural-only
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    defaults = ModelConfig(**raw.get("defaults", {}))
    repos: list[RepoEntry] = []
    for r in raw.get("repo", []):
        models = ModelConfig(**r["models"]) if "models" in r else None
        repos.append(RepoEntry(
            name=r["name"],
            path=r["path"],
            enrich=r.get("enrich", True),
            doc=r.get("doc", "SYSTEM.md"),
            install_hooks=r.get("install_hooks", True),
            models=models,
        ))
    return Manifest(defaults=defaults, repos=repos)


# ── onboarding ──────────────────────────────────────────────────────────


@dataclass
class OnboardResult:
    name: str
    path: str
    ok: bool
    scanned: int = 0
    resolved_edges: int = 0
    enriched: bool = False
    findings: int = 0
    degradations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolver_tooling() -> dict[str, bool]:
    """Which language servers are reachable on PATH right now.

    Used only to *report* degraded coverage — onboarding never fails
    because a language server is missing; the resolver layer already
    degrades to AST-only.
    """
    import importlib.util

    return {
        "python(jedi)": importlib.util.find_spec("jedi") is not None,
        "typescript(tsserver)": shutil.which("typescript-language-server") is not None,
        "go(gopls)": shutil.which("gopls") is not None,
    }


def onboard_repo(entry: RepoEntry, models: ModelConfig) -> OnboardResult:
    """Bring one repo to a known state. Idempotent.

    scan/resolve/enrich/render are all idempotent already (upsert +
    description cache + marker injection); install_hooks overwrites by
    convention. Re-running is a safe no-op-or-update.

    Graceful degradation: a missing language server → AST-only edges
    (logged as a degradation, not an error). An unreachable model
    endpoint during enrich → structural-only for that run (logged,
    not fatal). One flaky dependency must never wedge the fleet.
    """
    from otter_docs import Repo
    from otter_docs.hooks import install_hooks

    result = OnboardResult(name=entry.name, path=entry.path, ok=False)
    t0 = time.time()
    root = Path(entry.path)
    if not root.exists():
        result.errors.append(f"path does not exist: {entry.path}")
        result.seconds = round(time.time() - t0, 2)
        _write_heartbeat(root, entry, result, models)
        return result

    tooling = _resolver_tooling()
    for lang, present in tooling.items():
        if not present:
            result.degradations.append(f"{lang} unavailable → AST-only edges")

    try:
        with Repo(root, name=entry.name) as repo:
            sc = repo.scan(reset=False)
            result.scanned = sc.files_parsed
            if sc.errors:
                result.degradations.append(
                    f"{len(sc.errors)} files failed to parse"
                )

            rep = repo.resolve()
            result.resolved_edges = sum(r.edges_emitted for r in rep.values())

            if entry.enrich:
                ok, note = _try_enrich(repo, models)
                result.enriched = ok
                if not ok:
                    result.degradations.append(note)

            repo.render_document(root / entry.doc)
            result.findings = len(repo.findings())

        if entry.install_hooks:
            written = install_hooks(root, out=entry.doc)
            if not written:
                result.degradations.append(
                    "no .git dir → hooks not installed"
                )

        result.ok = not result.errors
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            # A concurrent writer slipped past the onboard lock (e.g. a
            # manual `otter-docs scan` during the timer window). This
            # is transient and self-correcting — the next scheduled run
            # reprocesses. Degrade, never FAIL: a momentary lock must
            # not flip the systemd oneshot to failed (the exact prod
            # bug this guard exists to kill).
            result.degradations.append(
                "skipped this run: graph.db locked by a concurrent "
                "process (next scheduled run reprocesses)"
            )
            result.ok = not result.errors
        else:
            result.errors.append(f"{type(e).__name__}: {e}")
    except Exception as e:
        result.errors.append(f"{type(e).__name__}: {e}")

    result.seconds = round(time.time() - t0, 2)
    _write_heartbeat(root, entry, result, models)
    return result


def _try_enrich(repo, models: ModelConfig) -> tuple[bool, str]:
    """Run the semantic tier; degrade cleanly if the endpoint is down.

    Returns (enriched_ok, note). A model endpoint being unreachable is
    a degradation, not a failure — the repo still gets structural docs
    + findings + hooks.
    """
    try:
        from otter_docs.clients import (
            OpenAICompatEmbeddingClient,
            OpenAICompatLLMClient,
        )

        llm = OpenAICompatLLMClient(
            model=models.llm_model, base_url=models.llm_url,
            default_max_tokens=200, timeout=30.0,
        )
        emb = OpenAICompatEmbeddingClient(
            model=models.embed_model, base_url=models.embed_url,
            dim=models.embed_dim, timeout=30.0,
        )
        report = repo.enrich(llm, emb)
        produced = (
            report.modules_enriched
            + report.functions_enriched
            + report.classes_enriched
        )
        # enrich() swallows per-symbol failures into report.errors and
        # returns instead of raising. If NOTHING got vectors, the
        # endpoint is effectively down — that's a degradation, not a
        # success with "some errors".
        if produced == 0:
            return False, (
                "enrich skipped (model endpoint produced no vectors — "
                f"{len(report.errors)} symbol errors)"
            )
        if report.errors:
            return True, (
                f"enrich partial: {produced} symbols, "
                f"{len(report.errors)} errors"
            )
        return True, ""
    except Exception as e:
        return False, f"enrich skipped (model endpoint unreachable: {type(e).__name__})"


def _write_heartbeat(
    root: Path, entry: RepoEntry, result: OnboardResult, models: ModelConfig,
) -> None:
    """Persist .otter-docs/status.json — the anti-silent-breakage record.

    Best-effort: if we can't even write the heartbeat (path missing),
    swallow it; the in-memory OnboardResult is still returned to the
    caller / CLI.
    """
    try:
        data_dir = root / ".otter-docs"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "status.json").write_text(json.dumps({
            "name": entry.name,
            "last_onboard": _utc_now(),
            "ok": result.ok,
            "scanned": result.scanned,
            "resolved_edges": result.resolved_edges,
            "enriched": result.enriched,
            "findings": result.findings,
            "degradations": result.degradations,
            "errors": result.errors,
            "seconds": result.seconds,
            "models": {"llm": models.llm_model, "embed": models.embed_model},
        }, indent=2), encoding="utf-8")
    except OSError:
        pass


# ── concurrency guard ───────────────────────────────────────────────────
#
# Two `onboard` runs over the same manifest hit the same SQLite
# graph.db files and produce `database is locked` churn. Observed in
# prod 2026-05-19: the nightly timer overlapped a leftover manual
# catch-up run, v3's leg hard-failed (0 files), and the systemd
# oneshot service exited 1/FAILURE. The fix is twofold: an exclusive
# inter-process lock so runs never overlap (here), and a soft-degrade
# on any residual lock so a transient never FAILs the service
# (onboard_repo's except handler).


def default_onboard_lock_path(manifest_path: str | Path) -> Path:
    """Stable lock path for a given manifest.

    Keyed on the resolved manifest path so distinct manifests don't
    block each other but the same manifest always maps to one lock.
    Lives in the system temp dir — robust to a read-only manifest dir
    and conventional for a runtime lock.
    """
    h = hashlib.sha1(
        str(Path(manifest_path).resolve()).encode()
    ).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"otter-docs-onboard-{h}.lock"


@contextlib.contextmanager
def onboard_lock(lock_path: str | Path) -> Iterator[bool]:
    """Exclusive, non-blocking inter-process lock for a full onboard run.

    Yields True if this process holds the lock; False if another
    onboard already holds it (caller should skip this cycle — a
    concurrent run just produces lock churn, and a daily Persistent
    timer reprocesses next cycle anyway).

    flock is advisory and bound to the open fd, so the OS releases it
    automatically if the holder is SIGKILLed/OOM-ed — no stale lock to
    clean up, unlike a bare lockfile-exists check. No-ops to True where
    `fcntl` is unavailable (non-Linux); the systemd timer this guards
    is Linux-only.
    """
    try:
        import fcntl
    except ImportError:
        yield True
        return
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "w")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                yield False
                return
            raise
        try:
            yield True
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def onboard_all(
    manifest: Manifest, *, only: str | None = None, enrich: bool | None = None,
) -> list[OnboardResult]:
    """Onboard every repo in the manifest (or just `only`).

    `enrich=False` forces structural-only for this run regardless of
    per-repo manifest setting (used by the fast-tier hook/CI path);
    `enrich=None` honors each entry's setting.
    """
    out: list[OnboardResult] = []
    for entry in manifest.repos:
        if only is not None and entry.name != only:
            continue
        eff = entry
        if enrich is False and entry.enrich:
            eff = entry.model_copy(update={"enrich": False})
        out.append(onboard_repo(eff, manifest.model_for(entry)))
    return out


# ── status / heartbeat ──────────────────────────────────────────────────


@dataclass
class RepoStatus:
    name: str
    present: bool
    last_onboard: str | None
    age_seconds: float | None
    ok: bool
    scanned: int
    findings: int
    enriched: bool
    degradations: list[str]
    errors: list[str]
    stale: bool


def collect_status(
    manifest: Manifest, *, stale_after_seconds: float = 60 * 60 * 36,
) -> list[RepoStatus]:
    """Read every repo's heartbeat. A repo with no heartbeat, an old
    one, or recorded errors is flagged — that's the loud signal that
    replaces silent rot. Default staleness window: 36h (a daily timer
    has missed at least one full cycle)."""
    now = time.time()
    out: list[RepoStatus] = []
    for entry in manifest.repos:
        hb = Path(entry.path) / ".otter-docs" / "status.json"
        if not hb.exists():
            out.append(RepoStatus(
                name=entry.name, present=False, last_onboard=None,
                age_seconds=None, ok=False, scanned=0, findings=0,
                enriched=False, degradations=[],
                errors=["no heartbeat — never onboarded?"], stale=True,
            ))
            continue
        try:
            d = json.loads(hb.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            out.append(RepoStatus(
                name=entry.name, present=True, last_onboard=None,
                age_seconds=None, ok=False, scanned=0, findings=0,
                enriched=False, degradations=[],
                errors=[f"unreadable heartbeat: {e}"], stale=True,
            ))
            continue
        age: float | None = None
        ts = d.get("last_onboard")
        if ts:
            try:
                age = now - datetime.fromisoformat(ts).timestamp()
            except ValueError:
                age = None
        stale = age is None or age > stale_after_seconds
        out.append(RepoStatus(
            name=entry.name, present=True, last_onboard=ts,
            age_seconds=round(age, 0) if age is not None else None,
            ok=bool(d.get("ok")), scanned=int(d.get("scanned", 0)),
            findings=int(d.get("findings", 0)),
            enriched=bool(d.get("enriched")),
            degradations=list(d.get("degradations", [])),
            errors=list(d.get("errors", [])),
            stale=stale,
        ))
    return out


def status_is_healthy(statuses: list[RepoStatus]) -> bool:
    """True iff no repo is stale or errored. The exit-code contract for
    `otter-docs status` so it works as a monitoring health check."""
    return all(s.ok and not s.stale and not s.errors for s in statuses)


# ── systemd units (slow semantic pass on a schedule) ────────────────────


def systemd_units(
    *,
    manifest_path: str,
    user: str,
    otter_docs_bin: str = "otter-docs",
    on_calendar: str = "*-*-* 03:30:00",
    path_env: str | None = None,
) -> dict[str, str]:
    """Return {'otter-docs-onboard.service': ..., '.timer': ...} text.

    Modeled on the v3 gnosis-code-graph.timer pattern. The timer runs
    the FULL onboard (semantic tier included) on a schedule so the
    slow pass is self-executing — no human running a script. The fast
    structural tier still rides git hooks per-commit. Install with:

        sudo tee /etc/systemd/system/otter-docs-onboard.service < ...
        sudo tee /etc/systemd/system/otter-docs-onboard.timer   < ...
        sudo systemctl daemon-reload
        sudo systemctl enable --now otter-docs-onboard.timer

    We never auto-sudo; this only emits the text.

    CRITICAL: `otter_docs_bin` must be an ABSOLUTE path and `path_env`
    must include the resolver tooling (gopls/tsserver) dirs. systemd
    runs with a minimal environment — a bare `otter-docs` or a PATH
    without the language servers produces a unit that *silently
    fails*, which is precisely the rot this feature exists to
    prevent. The CLI resolves both from the live environment.
    """
    env_line = f"Environment=PATH={path_env}\n" if path_env else ""
    service = f"""\
[Unit]
Description=otter-docs scheduled onboard (semantic tier, all repos)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User={user}
{env_line}ExecStart={otter_docs_bin} onboard --manifest {manifest_path}
TimeoutStartSec=7200
"""
    timer = f"""\
[Unit]
Description=Run otter-docs onboard on a schedule

[Timer]
OnCalendar={on_calendar}
Persistent=true

[Install]
WantedBy=timers.target
"""
    return {
        "otter-docs-onboard.service": service,
        "otter-docs-onboard.timer": timer,
    }
