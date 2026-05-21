# Changelog

All notable changes to **otter-docs** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-21

### Added

- **Auto-docs infrastructure layer** — five new filesystem detectors
  that pick up the surfaces every codebase carries:
  - `DependencyManifestDetector` — pyproject (PEP 621 + Poetry),
    package.json, go.mod, Cargo.toml, Gemfile, requirements.txt.
    Monorepo support: when there is no root manifest, walks
    subdirectories (depth-bounded, skip-list for node_modules /
    vendor / target / etc.) and returns one record per discovered
    subtree.
  - `LicenseDetector` — `LICENSE` / `LICENSE.md` / `LICENSE.txt` /
    `COPYING` (case-insensitive). Best-effort SPDX identification
    against a small list of canonical headers (MIT, Apache-2.0, GPL
    family, BSD family, MPL, ISC, AGPL, Unlicense).
  - `ReadmeDetector` — README at root or `docs/`; in a monorepo
    walks subtrees but only emits READMEs from directories that
    also contain a dependency manifest (the package-root signal).
    Extracts the prose between H1 and the first H2 plus the H2
    outline.
  - `TestsDetector` — discovers `tests/` / `__tests__/` / `test/` /
    `spec/` directories plus root-level test files
    (`test_*.py`, `*.test.ts`, `*.spec.ts`, `*_test.go`, …).
    Infers runners (pytest / jest / vitest / go test / rspec /
    minitest) from filename pattern + nearby config files.
  - `SourceLayoutDetector` — top-level directory map with curated
    annotations (`src/` → source root, `tests/` → tests, `docs/` →
    documentation, …). Filters build artifacts and hidden caches.
- **`Repo.confirm_redundancy(finding, llm) → RedundancyVerdict`** —
  LLM-direct tier method that classifies a `redundancy.*` Finding
  as `duplicate` / `sibling` / `shared_pattern` / `coincidental`.
  Designed for the publisher path: the embedding-tier detector is
  recall-oriented; this is the precision-oriented second pass that
  reads both function bodies. Content-addressed verdict cache
  (`SqliteRedundancyCache`, sibling of `code_descriptions` and
  `code_embeddings`) makes steady-state re-runs ~free.
- **Cyclomatic complexity** on `FunctionRecord` — McCabe metric
  computed at parse time by the Python parser (1 + decision-point
  count; comprehensive grammar coverage: if/elif/for/while/except/
  bool-op/ternary/comprehension clauses). Other-language parsers
  leave it `None`. `large_function` detector now trips on either
  line count OR cyclomatic complexity, with a confidence bump when
  both gates trip.

### Changed

- `redundancy.semantic_equivalence` — emits `evidence.shape ∈
  {likely_duplicate, sibling_methods, lifecycle_hook}` with confidence
  calibrated per shape. Same-name Python lifecycle hooks (`__init__`,
  `setUp`, `__repr__`, …) on different classes → `lifecycle_hook`
  with confidence 0.05; same-name methods on different classes (via
  `CONTAINS`-edge evidence) → `sibling_methods` with code-similarity
  gate raised to 0.93 and confidence × 0.7; everything else →
  `likely_duplicate` (the previous behavior). Default
  `description_threshold` raised 0.92 → 0.95 — the 0.92–0.95 band
  was where bottom-of-list garbage lived. Test-file pairs (either
  side under `tests/` or matching `*_test.py` / `*.test.ts` / etc.)
  are skipped entirely.
- `dead_code` — adds `evidence.visibility ∈ {private, public,
  public_export}`. Confidence is the base score (which still
  reflects whether `resolve()` ran) multiplied by ×1.0 / ×0.7 /
  ×0.4 respectively. Lets downstream rank instead of treating
  every orphan as one number.
- `empty_module` — `is_marker_likely=True` (`__init__.py` with no
  imports + no docstring) drops to 0.05 confidence so it's
  observable on the bus but never cards under a default subscriber
  threshold.
- `scan()` per-file orphan purge — re-scanning a file now deletes
  records (functions + classes) whose hash-anchored guid is no
  longer emitted by the parser. Closes a silent recall gap where a
  function moving between scans left a ghost row behind; v3's
  telekora repo had 3–4 records per actively-edited TSX component
  before this fix. Backends without the new method skip cleanup
  with a documented limitation.
- `render_document()` default section order now places infrastructure
  sections (readme → dependencies → license → source_layout → tests)
  before the code-graph sections (system_overview onward) — matches
  the reading order a human expects.
- `description.divergence` moved to `cost_tier="llm_direct"` and is
  **excluded from a default `findings()` call**. A default
  (unfiltered) scan now runs the static + embedding tiers only;
  `llm_direct` detectors are opt-in via `cost_tiers={"llm_direct"}`
  or by naming the kind explicitly. Reason: cosine distance between
  description and code vectors conflates real docstring staleness with
  terse-code/verbose-description — too noisy to surface by default.
  An LLM-judge replacement (`confirm_description`) is planned for v0.2.

## [0.1.0rc2] — 2026-05-20

### Added

- **Rust parser** — `.rs` files; functions, methods (via `impl`),
  structs, enums, traits (including trait method signatures), unions.
  Impl methods qualified as `Type.method`. Tree-sitter-rust 0.23+.
- **Java parser** — `.java` files; classes, interfaces, enums, records,
  methods, constructors. Methods qualified as `Class.method`. Nested
  types walked. Tree-sitter-java 0.23+.
- **`Repo.findings_stream()`** — iterator-form of `findings()` so
  consumers can publish each Finding to a message bus the moment it's
  emitted instead of waiting for every detector to finish. Same
  filters; honors a per-detector `run_stream` hook for true laziness.

## [0.1.0rc1] — 2026-05-20

Initial release candidate. Library is functional end-to-end across
Python / Go / TypeScript / TSX / JS. 257 tests pass on the default
install.

### Added

- Polyglot AST via tree-sitter (Python, Go, TypeScript, TSX, JS).
- Cross-file resolvers: jedi (Python), `typescript-language-server`
  (TS), `gopls` (Go). Each registers only if its tooling is reachable.
- **Loud-skip warnings**: when scan sees source for a language whose
  resolver isn't registered, `resolve()` emits an actionable warning
  with the install command. Silence per-language via
  `OTTER_RESOLVER_QUIET=<lang>[,<lang>...]`.
- Three-vector indexing per symbol (description / code / docstring),
  each embedded separately. Content-addressed caches for both
  description generation and embedding so re-runs on unchanged code
  are free.
- Detectors: `dead_code`, `large_function`, `empty_module` (static
  tier); `redundancy.semantic_equivalence`, `description.divergence`
  (embedding tier).
- LLM-direct tier: `propose_consolidation`, `review_change`,
  `describe`.
- Agent harness with `schemas`, `prompts`, MCP-emittable tools, and a
  graded `Harness.run()`.
- Renderers (`system_overview`, `findings_summary`,
  `redundancy_report`, `dependency_graph`, `architecture_smells`) with
  marker-based injection that preserves human prose across reruns.
- Backends: SQLite + sqlite-vec (default); Neo4j (opt-in via `[neo4j]`
  extra).
- LLM/embedding clients: Ollama-native, OpenAI-compatible (works with
  llama.cpp, vLLM, OpenAI), plus deterministic fakes.
- GUID assignment (`otter-docs assign-guids`) for cross-tool symbol
  identity. Idempotent. `# guid:` / `// guid:` markers.
- Multi-repo onboarding from a declarative `repos.toml` manifest with
  flock-guarded concurrency, soft-degrade on transient SQLite locks,
  and `.otter-docs/status.json` heartbeats.
- MCP server (`otter-docs serve`) behind the `[mcp]` extra.

### Evaluation

- Bundled smoke set (12 hand-labeled pairs, `nomic-embed-text`):
  F1 = 1.00 at thresholds 0.725-0.95.
- CodeNet-Python800 scale benchmark: F1 = 0.854 on the type-4-enforced
  set (200 positives + 200 negatives, 72 problems), with a +0.030
  contamination delta vs. the unfiltered same-problem baseline.
  Methodology in `docs/evaluation.md`.

### Not yet shipped

- PyPI release — building from this tag.
- `risk.behavior_propagation` (call-graph-aware risk).
- CLI flags for `enrich` and Neo4j backend selection (library-only for
  v0.1).

[0.1.0rc2]: https://github.com/blong-dev/otter-docs/releases/tag/v0.1.0rc2
[0.1.0rc1]: https://github.com/blong-dev/otter-docs/releases/tag/v0.1.0rc1
