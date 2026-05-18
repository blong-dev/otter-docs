"""otter-docs command-line interface.

Stdlib argparse only — no click/typer dep. Subcommands:

  otter-docs init [path]            bootstrap SYSTEM.md with markers
  otter-docs scan [path]            scan + cross-file resolve
  otter-docs find [path] --kind K   run detectors, print findings
  otter-docs render [path]          (re)write the generated document
  otter-docs install-hooks [path]   git pre-commit / pre-push hooks
  otter-docs serve [path]           run the MCP server (needs [mcp] extra)

Every command takes an optional positional repo path (default ".").
The CLI never calls an LLM/embedder — enrichment and the LLM-direct
tier are library-only for v0.1 so the CLI stays fast and offline. A
later version can add `--llm-url` once we've nailed the config story.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _open_repo(path: str):
    from otter_docs import Repo
    return Repo(Path(path).resolve())


def cmd_init(args: argparse.Namespace) -> int:
    repo = _open_repo(args.path)
    try:
        repo.scan()
        repo.resolve()
        doc_path = Path(args.path) / args.out
        repo.render_document(doc_path)
        print(f"Wrote {doc_path}")
        return 0
    finally:
        repo.close()


def cmd_scan(args: argparse.Namespace) -> int:
    repo = _open_repo(args.path)
    try:
        report = repo.scan(reset=args.reset)
        print(
            f"scan: {report.files_parsed} files, {report.modules} modules, "
            f"{report.functions} functions, {report.classes} classes, "
            f"{report.edges} edges, {len(report.errors)} errors"
        )
        if not args.no_resolve:
            reports = repo.resolve()
            for lang, rep in reports.items():
                print(f"resolve[{lang.value}]: {rep.edges_emitted} edges")
        return 0
    finally:
        repo.close()


def cmd_find(args: argparse.Namespace) -> int:
    repo = _open_repo(args.path)
    try:
        repo.scan()
        if not args.no_resolve:
            repo.resolve()
        kinds = set(args.kind) if args.kind else None
        findings = repo.findings(kinds=kinds)
        if args.json:
            print(json.dumps([f.model_dump() for f in findings], indent=2, default=str))
            return 0
        if not findings:
            print("No findings.")
            return 0
        # Rank by confidence × edge_confidence for a useful default order.
        def key(f):
            ec = f.edge_confidence if f.edge_confidence is not None else 1.0
            return f.confidence * ec
        for f in sorted(findings, key=key, reverse=True)[: args.limit]:
            loc = f.locations[0] if f.locations else None
            where = f"{loc.path}:{loc.line}" if loc and loc.line else (loc.path if loc else "?")
            name = f.evidence.get("function_name", "")
            print(f"[{key(f):.2f}] {f.kind:35s} {where} {name}")
        print(f"\n{len(findings)} findings (showing up to {args.limit}).")
        return 0
    finally:
        repo.close()


def cmd_render(args: argparse.Namespace) -> int:
    repo = _open_repo(args.path)
    try:
        repo.scan()
        if not args.no_resolve:
            repo.resolve()
        if args.section:
            print(repo.render(args.section))
            return 0
        doc_path = Path(args.path) / args.out
        repo.render_document(doc_path)
        print(f"Wrote {doc_path}")
        return 0
    finally:
        repo.close()


def cmd_install_hooks(args: argparse.Namespace) -> int:
    from otter_docs.hooks import install_hooks
    written = install_hooks(Path(args.path).resolve(), out=args.out)
    if not written:
        print("No .git directory found — is this a git repo?", file=sys.stderr)
        return 1
    for p in written:
        print(f"Installed {p}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from otter_docs.mcp import serve
    except ImportError as e:
        print(
            f"MCP server needs the [mcp] extra: pip install otter-docs[mcp]\n({e})",
            file=sys.stderr,
        )
        return 1
    serve(Path(args.path).resolve())
    return 0


def cmd_onboard(args: argparse.Namespace) -> int:
    from otter_docs.onboarding import load_manifest, onboard_all

    manifest = load_manifest(args.manifest)
    enrich = False if args.no_enrich else None
    results = onboard_all(manifest, only=args.repo, enrich=enrich)
    if not results:
        print("nothing to onboard (no matching repos in manifest)",
              file=sys.stderr)
        return 1
    failed = 0
    for r in results:
        tag = "ok" if r.ok else "FAIL"
        print(f"[{tag}] {r.name}: {r.scanned} files, "
              f"{r.resolved_edges} edges, {r.findings} findings, "
              f"enriched={r.enriched}, {r.seconds}s")
        for d in r.degradations:
            print(f"    degraded: {d}")
        for e in r.errors:
            print(f"    error: {e}", file=sys.stderr)
        if not r.ok:
            failed += 1
    return 1 if failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    from otter_docs.onboarding import (
        collect_status,
        load_manifest,
        status_is_healthy,
    )

    manifest = load_manifest(args.manifest)
    statuses = collect_status(manifest)
    for s in statuses:
        flag = "OK  " if (s.ok and not s.stale and not s.errors) else "BAD "
        age = f"{s.age_seconds/3600:.1f}h" if s.age_seconds is not None else "—"
        stale = " STALE" if s.stale else ""
        print(f"[{flag}] {s.name}: last={s.last_onboard or 'never'} "
              f"age={age}{stale} scanned={s.scanned} "
              f"findings={s.findings} enriched={s.enriched}")
        for d in s.degradations:
            print(f"    degraded: {d}")
        for e in s.errors:
            print(f"    error: {e}")
    # Exit non-zero if anything is unhealthy → usable as a cron/CI
    # health check.
    return 0 if status_is_healthy(statuses) else 1


def cmd_systemd(args: argparse.Namespace) -> int:
    import getpass
    import os
    import shutil

    from otter_docs.onboarding import systemd_units

    # Resolve an ABSOLUTE otter-docs path. systemd has no PATH; a bare
    # name silently fails. Prefer the running interpreter's bin dir
    # (the venv this CLI is installed in), fall back to which().
    bin_dir = Path(sys.executable).parent
    cand = bin_dir / "otter-docs"
    otter_bin = str(cand) if cand.exists() else (
        shutil.which("otter-docs") or "otter-docs"
    )
    # Build a PATH that includes the venv bin + wherever the resolver
    # language servers (gopls / typescript-language-server) currently
    # live, so the scheduled run has the same coverage as this shell.
    path_parts: list[str] = [str(bin_dir)]
    for tool in ("gopls", "typescript-language-server", "node", "go"):
        p = shutil.which(tool)
        if p:
            d = str(Path(p).parent)
            if d not in path_parts:
                path_parts.append(d)
    for sysdir in ("/usr/local/bin", "/usr/bin", "/bin"):
        if sysdir not in path_parts:
            path_parts.append(sysdir)
    units = systemd_units(
        manifest_path=str(Path(args.manifest).resolve()),
        user=args.user or getpass.getuser(),
        otter_docs_bin=otter_bin,
        on_calendar=args.on_calendar,
        path_env=os.pathsep.join(path_parts),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in units.items():
        (out_dir / fname).write_text(text, encoding="utf-8")
        print(f"wrote {out_dir / fname}")
    print("\nInstall (we never auto-sudo):")
    print(f"  sudo cp {out_dir}/otter-docs-onboard.* /etc/systemd/system/")
    print("  sudo systemctl daemon-reload")
    print("  sudo systemctl enable --now otter-docs-onboard.timer")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="otter-docs", description="Polyglot codebase inspection.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_path(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("path", nargs="?", default=".", help="repo root (default: .)")

    sp = sub.add_parser("init", help="bootstrap a generated SYSTEM.md")
    add_path(sp)
    sp.add_argument("--out", default="SYSTEM.md", help="document filename")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("scan", help="scan + cross-file resolve")
    add_path(sp)
    sp.add_argument("--reset", action="store_true", help="wipe this repo's rows first")
    sp.add_argument("--no-resolve", action="store_true", help="skip cross-file resolution")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("find", help="run detectors, print findings")
    add_path(sp)
    sp.add_argument("--kind", action="append", help="filter to a finding kind (repeatable)")
    sp.add_argument("--limit", type=int, default=40, help="max findings to print")
    sp.add_argument("--json", action="store_true", help="emit JSON")
    sp.add_argument("--no-resolve", action="store_true")
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("render", help="render a section or the full document")
    add_path(sp)
    sp.add_argument("--section", help="render just this section to stdout")
    sp.add_argument("--out", default="SYSTEM.md", help="document filename")
    sp.add_argument("--no-resolve", action="store_true")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("install-hooks", help="install git pre-commit/pre-push hooks")
    add_path(sp)
    sp.add_argument("--out", default="SYSTEM.md", help="document the hook regenerates")
    sp.set_defaults(func=cmd_install_hooks)

    sp = sub.add_parser("serve", help="run the MCP server (needs [mcp] extra)")
    add_path(sp)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser(
        "onboard",
        help="idempotently onboard every repo in a manifest "
             "(scan+resolve+enrich+render+hooks)",
    )
    sp.add_argument("--manifest", required=True, help="path to repos.toml")
    sp.add_argument("--repo", help="onboard only this repo by name")
    sp.add_argument("--no-enrich", action="store_true",
                    help="force structural-only (skip the semantic tier)")
    sp.set_defaults(func=cmd_onboard)

    sp = sub.add_parser(
        "status",
        help="per-repo heartbeat from a manifest; exits non-zero if "
             "any repo is stale or errored",
    )
    sp.add_argument("--manifest", required=True, help="path to repos.toml")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser(
        "systemd",
        help="emit the scheduled-onboard systemd unit files",
    )
    sp.add_argument("--manifest", required=True, help="path to repos.toml")
    sp.add_argument("--out-dir", default=".", help="where to write the unit files")
    sp.add_argument("--user", help="systemd User= (default: current user)")
    sp.add_argument("--on-calendar", default="*-*-* 03:30:00",
                    help="systemd OnCalendar= (default: daily 03:30)")
    sp.set_defaults(func=cmd_systemd)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
