# otter-docs

A polyglot codebase inspection library for agent-driven development.

> **Status (2026-05-15): v0.1 in active development on `main`. Library
> is functional end-to-end; not yet released to PyPI.**

## What it is

otter-docs builds a queryable model of a codebase — modules,
functions, classes, calls, imports — augmented with LLM-generated
description embeddings, and emits structured **findings** (redundancy,
drift, dead code, architectural smells) that an agent can act on. Each
finding can carry a recommendation with rationale, and the LLM-direct
tier can produce an apply-ready unified diff.

The library is designed for agents to consume, not humans to read. The
human operates the agent. otter-docs never applies changes itself —
it emits typed findings and proposed diffs; the harness owns
implementation.

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
```

## What's implemented

- **Polyglot AST** via tree-sitter — Python, Go, TypeScript/TSX, JS.
- **Cross-file resolution** via mature per-language solvers: jedi
  (Python, validated), `typescript-language-server` (TS, validated),
  `gopls` (Go, validated against gopls v0.21.1 — resolves
  cross-file calls and receiver methods).
  Each registers only when its tooling is present; a polyglot repo
  with partial tooling still gets partial coverage.
- **Three-vector indexing** per symbol: an LLM-generated description,
  the code slice, and the docstring — each embedded separately.
- **Detectors**:
  - static tier — `dead_code`, `large_function`, `empty_module`
  - embedding tier — `redundancy.semantic_equivalence`,
    `description.divergence`
- **LLM-direct tier** — `propose_consolidation` (generates a unified
  diff), `review_change` (structured review of a diff), `describe`.
- **Agent harness** — `schemas`, `prompts`, `tools` (MCP-spec
  emittable), and a `Harness` that grades a codebase.
- **Renderers** — `system_overview`, `findings_summary`,
  `redundancy_report`, `dependency_graph`, `architecture_smells`,
  with marker-based injection that preserves human prose across
  reruns.
- **Backends** — SQLite + sqlite-vec (default, zero-config); Neo4j
  adapter (opt-in, validated against a live instance).
- **Clients** — Ollama-native and OpenAI-compatible (llama.cpp /
  vLLM / OpenAI) LLM + embedding adapters, plus deterministic fakes.

## Evaluation — honest numbers

The `redundancy.semantic_equivalence` detector is the wedge: it
should catch "100 ways to skin a cat" duplication that source-trained
clone models miss, because it ranks on the *description* vector (the
LLM's prose abstracts away surface differences).

**Bundled smoke set (12 hand-labeled pairs, real embedder
`nomic-embed-text`):** F1 = 1.00 at thresholds 0.725–0.95, including
the Type-4 cases (iterative vs recursive factorial; two structurally
different palindrome checks; two linked-list reversals). This
validates the *mechanism* — description-vector cosine cleanly
separates semantic clones from look-alikes.

**What this is NOT:** 12 hand-picked pairs with idealized
(hand-written, identical-for-clones) descriptions does not establish
production-scale precision. The bundled number proves the mechanism;
it is not a benchmark figure.

**Scale benchmark — IBM Project CodeNet (Python800).** We chose
CodeNet over GPTCloneBench deliberately: GPTCloneBench's data is
CC BY-NC-ND (a NonCommercial + NoDerivatives gray area for a
commercial product), and it's mostly Java/C/C# — only its Python
slice overlaps our parsers. CodeNet is **CDLA-Permissive-2.0**
(commercial use + derivatives explicitly OK), Python-native, and
type-4 by construction: every accepted submission to a problem is a
semantically-equivalent solution; different problems are non-clones.

The sampler is the part that has to be honest, not hand-wavy. A
naive same-problem→clone sampler is meaningless because same-problem
submissions are full of copy-paste. `otter_docs.eval_codenet`
enforces:
- **type-4 positives only** — a same-problem pair is kept only if
  token-set Jaccard is below a threshold AND its AST node-type
  histogram is structurally divergent. Copy-paste / renamed-var
  pairs are excluded. An unfiltered same-problem set is scored in
  parallel so the report shows the *contamination delta* (easy vs
  hard number) explicitly.
- **two negative strata** — `hard` (different-problem pairs that are
  surface-similar, the case surface-trained models fail) and
  `random`, reported separately.
- **no description leakage** — each snippet is described by the
  shipping describer from its *code only*, never the problem id.
- frozen seed + config, printed with the number → reproducible by
  construction.

Reproduce it yourself: `examples/codenet_eval.py` (download
instructions in the file header). Not a CI step — CI has no dataset,
LLM, or embedder; CI runs the harness on the bundled set with a
fake embedder to guard the precision/recall/threshold math.

**Result (seed 1729, 200 type-4 positives + 100 hard + 100 random
negatives, 72 distinct problems, real `nomic-embed-text` over
LLM-generated descriptions):**

| set | threshold | precision | recall | F1 |
|---|---|---|---|---|
| **type-4 enforced (headline)** | 0.775 | 0.82 | 0.89 | **0.854** |
| unfiltered same-problem (baseline) | 0.80 | 0.91 | 0.86 | 0.884 |

The number that matters is not 0.854 in isolation — it's the
**+0.030 contamination delta**: the structurally-hard type-4 set
scores almost as high as the copy-paste-contaminated baseline. The
method is *not* riding surface similarity; it's capturing semantic
equivalence on genuinely-different-structure code. That small gap is
the evidence the description-vector thesis holds.

Calibration: 0.85 sits above the C4 ≈ 0.70 cross-language SOTA — but
C4's figure is on *GPTCloneBench* and this is on *CodeNet-Python800*
with a 400-pair sample, so this is **directional, not a strict
"beats C4" claim**. Different dataset, different scale.
Reproduce with `examples/codenet_eval.py` (config above is the
frozen `SamplerConfig`; same seed + models → same number).

CodeNet: <https://github.com/IBM/Project_CodeNet> ·
CDLA-Permissive-2.0: <https://cdla.dev/permissive-2-0/> ·
Why BigCloneBench is corrupted: <https://arxiv.org/html/2505.04311v1>.

## Known limitations

- `dead_code` is heuristic. With cross-file resolution it's a strong
  signal (gnosis: 28% fewer findings after `resolve()`), but methods
  reached via dynamic dispatch (`self.x.method()`) still escape it.
  Findings carry `confidence` and `edge_confidence` for exactly this
  reason — weight by them.
- All three resolvers are validated against their live language
  servers (jedi, typescript-language-server, gopls v0.21.1).
- `risk.behavior_propagation` (call-graph-aware risk) is deferred
  past v0.1.
- Embedding quality is the embedder's; we don't fine-tune.

## License

MIT.

## Links

- Repository: <https://github.com/blong-dev/otter-docs>
- Issues: <https://github.com/blong-dev/otter-docs/issues>
