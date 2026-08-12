# otter-docs

[![PyPI version](https://img.shields.io/pypi/v/otter-docs.svg)](https://pypi.org/project/otter-docs/)
[![Python versions](https://img.shields.io/pypi/pyversions/otter-docs.svg)](https://pypi.org/project/otter-docs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A polyglot codebase inspection library for agent-driven development.

> **Status (2026-05-20):** `0.1.0rc2`. Library is functional end-to-end
> across Python / Go / TypeScript / TSX / JS / Rust / Java. 274 tests
> pass on the default install. PyPI release imminent.

## Quickstart

```bash
pip install otter-docs
```

```bash
otter-docs scan .            # tree-sitter AST → graph
otter-docs find .            # static findings (dead_code, large_function, …)
otter-docs render .          # write SYSTEM.md with marker-based injection
otter-docs install-hooks .   # pre-commit + pre-push
```

Or from Python:

```python
from otter_docs import Repo

with Repo(".") as r:
    r.scan()
    r.resolve()
    for f in r.findings():
        print(f.kind, f.locations[0].path)
```

## What it is

otter-docs builds a queryable model of a codebase — modules, functions,
classes, calls, imports — augmented with LLM-generated description
embeddings, and emits structured **findings** (redundancy, drift, dead
code, architectural smells) that an agent can act on. Each finding can
carry a recommendation with rationale, and the LLM-direct tier can
produce an apply-ready unified diff.

The library is designed for agents to consume, not humans to read. The
human operates the agent. otter-docs never applies changes itself — it
emits typed findings and proposed diffs; the harness owns
implementation.

## Install matrix

The base wheel is fully usable on its own (scan + static findings +
render + hooks + GUID assignment + SQLite backend). The extras unlock
specific layers:

| install | unlocks | external tooling |
|---|---|---|
| `pip install otter-docs` | scan, static findings, render, install-hooks, assign-guids | — |
| `pip install otter-docs[python-resolver]` | cross-file resolve for Python | — (pulls `jedi`) |
| `pip install otter-docs[neo4j]` | Neo4j backend | a running Neo4j |
| `pip install otter-docs[mcp]` | `otter-docs serve` (MCP server) | — |
| `pip install otter-docs[all-resolvers]` | every available resolver extra | — |
| `pip install otter-docs[dev]` | tests + ruff + every optional dep | — |

Go and TypeScript resolvers don't have pip extras — they require their
language servers on PATH:

```bash
go install golang.org/x/tools/gopls@latest                    # Go
npm install -g typescript typescript-language-server          # TS / TSX
```

If a language's resolver isn't registered (extra not installed, or LSP
not on PATH) but otter-docs scans source files in that language, you
get a loud warning naming the install command. Silence per-language
with `OTTER_RESOLVER_QUIET=go` (etc).

For the **enrich** tier (LLM descriptions + three-vector embeddings),
bring any OpenAI-compatible LLM endpoint and embedder endpoint
(llama.cpp / vLLM / Ollama / OpenAI). See the [pipeline](#pipeline)
section below.

## Pipeline

```python
from otter_docs import Repo
from otter_docs.clients import OpenAICompatLLMClient, OpenAICompatEmbeddingClient

with Repo("/path/to/repo") as repo:
    repo.scan()      # tree-sitter AST → modules/functions/classes + edges
    repo.resolve()   # cross-file call resolution (jedi / tsserver / gopls)
    repo.enrich(llm, embedder)   # three vectors per symbol (optional)
    findings = repo.findings()   # typed Finding list
    rec = repo.propose_consolidation(findings[0], llm)  # LLM-direct
```

Or drive it from an agent:

```python
from otter_docs.agent import Harness
report = Harness(repo, llm=llm, embedder=embedder).run()
# report.overall_letter, report.grades, report.top_findings, ...
```

Or from the CLI:

```
otter-docs scan .            # scan + cross-file resolve
otter-docs find . --kind dead_code
otter-docs render .          # write/update SYSTEM.md
otter-docs init .            # bootstrap SYSTEM.md with markers
otter-docs install-hooks .   # git pre-commit/pre-push
otter-docs serve .           # MCP server (needs the [mcp] extra)
otter-docs assign-guids .    # mint `# guid:` / `// guid:` markers
otter-docs onboard --manifest repos.toml   # multi-repo fleet onboarding
otter-docs status  --manifest repos.toml   # per-repo heartbeat (non-zero exit if stale)
otter-docs systemd --manifest repos.toml   # emit scheduled-onboard systemd unit files
```

## What's implemented

- **Polyglot AST** via tree-sitter — Python, Go, TypeScript/TSX, JS,
  Rust, Java.
- **Cross-file resolution** via mature per-language solvers: jedi
  (Python, validated), `typescript-language-server` (TS, validated),
  `gopls` (Go, validated against gopls v0.21.1 — resolves cross-file
  calls and receiver methods). Each registers only when its tooling is
  present; a polyglot repo with partial tooling still gets partial
  coverage **and a loud warning naming the missing piece**.
- **Three-vector indexing** per symbol: an LLM-generated description,
  the code slice, and the docstring — each embedded separately.
- **Content-addressed caches** (`describe` and `embed`) — re-running on
  unchanged code is free.
- **Detectors**:
  - static tier — `dead_code` (visibility-ranked: `private` /
    `public` / `public_export` multiplicatively scale the confidence
    so downstream can rank within a single detector run),
    `large_function` (line-count *or* McCabe cyclomatic complexity
    — Python parser populates the latter), `empty_module`
    (`__init__.py` package-marker case auto-downgraded to
    informational)
  - embedding tier — `redundancy.semantic_equivalence` (emits an
    `evidence.shape` of `likely_duplicate` / `sibling_methods` /
    `lifecycle_hook` so downstream can treat dunder-method pairs and
    sibling methods on different classes differently from real
    duplicates; test-file pairs filtered automatically)
  - `llm_direct` tier (opt-in; excluded from a default `findings()`
    call) — `description.divergence`. See Known Limitations.
- **LLM-direct tier** — `propose_consolidation` (generates a unified
  diff), `review_change` (structured review of a diff), `describe`,
  `confirm_redundancy` (reads both function bodies and classifies a
  redundancy finding as `duplicate` / `sibling` / `shared_pattern` /
  `coincidental` — high-precision second pass over the embedding
  detector's high-recall output; content-addressed verdict cache in
  `graph.db` makes steady-state re-runs ~free).
- **Auto-docs infrastructure layer** — pure-filesystem detectors that
  pick up the surfaces every codebase carries: dependency manifests
  (pyproject / package.json / go.mod / Cargo.toml / Gemfile /
  requirements.txt — single-package and monorepo with subtree
  discovery), license (best-effort SPDX), README (summary +
  H2 outline; subtree READMEs in monorepos when the directory has a
  manifest), test layout (dir + file count + inferred runner:
  pytest / jest / vitest / go test / rspec), top-level source map
  with annotations. Each renders into its own marker section of
  OTTER.md; sections without a surface in this repo emit a quiet
  "not detected" placeholder rather than disappearing.
- **Agent harness** — `schemas`, `prompts`, `tools` (MCP-spec
  emittable), and a `Harness` that grades a codebase.
- **Renderers** — code-graph sections (`system_overview`,
  `findings_summary`, `redundancy_report`, `dependency_graph`,
  `architecture_smells`) plus the infrastructure-layer sections
  above (`readme`, `dependencies`, `license`, `source_layout`,
  `tests`). All use the same marker-based injection that preserves
  human prose across reruns.
- **Backends** — SQLite + sqlite-vec (default, zero-config); Neo4j
  adapter (opt-in, validated against a live instance).
- **Clients** — Ollama-native and OpenAI-compatible (llama.cpp / vLLM /
  OpenAI) LLM + embedding adapters, plus deterministic fakes.
- **GUID assignment** — `assign-guids` mints `# guid:<uuid>` /
  `// guid:<uuid>` inline markers as a cross-tool primary key across
  every supported language. Idempotent, diff-only when wired into git
  hooks.
- **Streaming findings** — `repo.findings_stream()` yields Findings as
  detectors produce them, so consumers can publish each Finding to a
  message bus the moment it's available rather than waiting for the
  full list. Filters (`kinds`, `cost_tiers`) apply just like
  `findings()`.
- **Multi-repo onboarding** — declarative `repos.toml` manifest,
  idempotent `otter-docs onboard`, flock-guarded against concurrent
  writers, `.otter-docs/status.json` heartbeat per repo.

## Evaluation

`redundancy.semantic_equivalence` ranks on the **description vector**
(an LLM-generated prose summary) so it catches semantic clones that
source-trained models miss. Headline number on CodeNet-Python800 (the
permissively-licensed Type-4 benchmark we vetted): **F1 0.854** on the
type-4-enforced set, with a +0.030 contamination delta vs. the
unfiltered baseline — i.e. the method captures semantic equivalence,
not surface similarity.

Full methodology, sampler design, and the reproducibility recipe are in
[`docs/evaluation.md`](docs/evaluation.md).

## Known limitations

- `dead_code` is heuristic. With cross-file resolution it's a strong
  signal (gnosis: 28% fewer findings after `resolve()`), but methods
  reached via dynamic dispatch (`self.x.method()`) still escape it.
  Findings carry `confidence` and `edge_confidence` for exactly this
  reason — weight by them.
- All three resolvers are validated against their live language servers
  (jedi, typescript-language-server, gopls v0.21.1).
- `description.divergence` is **disabled by default** (cost_tier
  `llm_direct`, excluded from an unfiltered `findings()` call). Cosine
  distance between a function's description vector and code vector
  conflates real docstring staleness with "terse code + verbose
  description" — a one-liner with a perfectly accurate docstring
  scores the same as one with a stale docstring (empirically ~0.53 on
  nomic-embed-text). The distance signal alone isn't trustworthy. The
  planned fix (`confirm_description`, v0.2) is an LLM judge that reads
  both and rules accurate / partial / stale / wrong — same shape as
  `confirm_redundancy`. Request the raw signal explicitly with
  `findings(kinds={"description.divergence"})` if you want it anyway.
- `risk.behavior_propagation` (call-graph-aware risk) is deferred past
  v0.1.
- Embedding quality is the embedder's; we don't fine-tune.

## License

MIT.

## Links

- Repository: <https://github.com/blong-dev/otter-docs>
- Issues: <https://github.com/blong-dev/otter-docs/issues>
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Evaluation: [`docs/evaluation.md`](docs/evaluation.md)
