# Evaluation

This document is the long-form companion to the brief mention in
[README](../README.md). It covers what the `redundancy.semantic_equivalence`
detector is measured against, why we chose CodeNet, and the headline
numbers.

## What we're measuring

`redundancy.semantic_equivalence` is otter-docs's wedge: it should
catch "100 ways to skin a cat" duplication that source-trained clone
models miss, because it ranks on the **description vector** (the LLM's
prose abstracts away surface differences).

## Bundled smoke set

12 hand-labeled pairs, real embedder `nomic-embed-text`: **F1 = 1.00 at
thresholds 0.725–0.95**, including the Type-4 cases (iterative vs
recursive factorial; two structurally different palindrome checks; two
linked-list reversals). This validates the *mechanism* —
description-vector cosine cleanly separates semantic clones from
look-alikes.

**What this is NOT:** 12 hand-picked pairs with idealized (hand-written,
identical-for-clones) descriptions does not establish production-scale
precision. The bundled number proves the mechanism; it is not a
benchmark figure.

## Scale benchmark — IBM Project CodeNet (Python800)

We chose CodeNet over GPTCloneBench deliberately:

- **License**: GPTCloneBench is CC BY-NC-ND (a NonCommercial +
  NoDerivatives gray area for a commercial product); CodeNet is
  **CDLA-Permissive-2.0** (commercial use + derivatives explicitly OK).
- **Language coverage**: GPTCloneBench is mostly Java/C/C# — only its
  Python slice overlaps our parsers. CodeNet is Python-native.
- **Type-4 by construction**: every accepted submission to a problem is
  a semantically-equivalent solution; different problems are non-clones.

### Honest sampler

The sampler is the part that has to be honest, not hand-wavy. A naive
same-problem→clone sampler is meaningless because same-problem
submissions are full of copy-paste. `otter_docs.eval_codenet` enforces:

- **type-4 positives only** — a same-problem pair is kept only if
  token-set Jaccard is below a threshold AND its AST node-type histogram
  is structurally divergent. Copy-paste / renamed-var pairs are
  excluded. An unfiltered same-problem set is scored in parallel so the
  report shows the *contamination delta* (easy vs hard number)
  explicitly.
- **two negative strata** — `hard` (different-problem pairs that are
  surface-similar, the case surface-trained models fail) and `random`,
  reported separately.
- **no description leakage** — each snippet is described by the shipping
  describer from its *code only*, never the problem id.
- **frozen seed + config**, printed with the number — reproducible by
  construction.

Reproduce it yourself: `examples/codenet_eval.py` (download instructions
in the file header). Not a CI step — CI has no dataset, LLM, or
embedder; CI runs the harness on the bundled set with a fake embedder
to guard the precision/recall/threshold math.

### Result

**Seed 1729, 200 type-4 positives + 100 hard + 100 random negatives, 72
distinct problems, real `nomic-embed-text` over LLM-generated
descriptions:**

| set | threshold | precision | recall | F1 |
|---|---|---|---|---|
| **type-4 enforced (headline)** | 0.775 | 0.82 | 0.89 | **0.854** |
| unfiltered same-problem (baseline) | 0.80 | 0.91 | 0.86 | 0.884 |

The number that matters is not 0.854 in isolation — it's the **+0.030
contamination delta**: the structurally-hard type-4 set scores almost as
high as the copy-paste-contaminated baseline. The method is *not* riding
surface similarity; it's capturing semantic equivalence on
genuinely-different-structure code. That small gap is the evidence the
description-vector thesis holds.

### Calibration

0.85 sits above the C4 ≈ 0.70 cross-language SOTA — but C4's figure is
on *GPTCloneBench* and this is on *CodeNet-Python800* with a 400-pair
sample, so this is **directional, not a strict "beats C4" claim**.
Different dataset, different scale.

Reproduce with `examples/codenet_eval.py` (config above is the frozen
`SamplerConfig`; same seed + models → same number).

## Links

- CodeNet: <https://github.com/IBM/Project_CodeNet>
- CDLA-Permissive-2.0: <https://cdla.dev/permissive-2-0/>
- Why BigCloneBench is corrupted: <https://arxiv.org/html/2505.04311v1>
