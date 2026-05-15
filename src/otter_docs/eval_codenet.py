"""CodeNet (Python800) clone-pair sampler — the commercially-clean
scale benchmark.

Project CodeNet is CDLA-Permissive-2.0 (commercial use + derivatives
OK). The `Python800` subset is 800 problems × 300 accepted,
IBM-near-dup-removed Python submissions. Same-problem submissions are
type-4 clones *by construction* (different code, same function);
different-problem submissions are non-clones.

A naive same-problem→clone sampler produces a meaningless number
because same-problem submissions still contain near-identical
(type-1/2/3) code. This sampler defends against every known failure
mode explicitly:

  POSITIVES — structural-divergence filter. A same-problem pair only
    counts as a type-4 clone if it's structurally *different*:
    token-set Jaccard below `max_token_jaccard` AND AST node-type
    histogram distance at/above `min_ast_distance`. That forces
    positives to be genuine "skin a cat", not copy-paste. We also
    sample an UNFILTERED same-problem set in parallel so the report
    can show the contamination delta (easy number vs hard number).

  NEGATIVES — two strata. `hard`: different-problem pairs with HIGH
    surface similarity (the case where surface-trained models fail
    and a description-vector approach should win). `random`:
    uniform different-problem pairs. Reported separately.

  INDEPENDENCE — cap pairs per problem, require ≥ `min_problems`
    distinct problems, fixed seed. No single easy problem dominates.

  NO DESCRIPTION LEAKAGE — this module never reads the problem id
    into a description. `fill_descriptions` runs the *shipping*
    Describer over each snippet's code independently — the exact
    production path.

The config object is frozen and meant to be printed verbatim next to
any number this produces, so the result is reproducible and the
easy/hard split is visible.
"""

from __future__ import annotations

import random
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language as _TsLang
from tree_sitter import Parser as _TsParser

from otter_docs.eval import ClonePair

_PY_LANG = _TsLang(tree_sitter_python.language())
_PY_PARSER = _TsParser(_PY_LANG)


@dataclass(frozen=True)
class SamplerConfig:
    """Every knob that affects which pairs come out. Print this with
    the result — it's what makes the number reproducible."""

    seed: int = 1729
    target_positives: int = 200
    target_negatives: int = 200
    # Negative split: fraction that are *hard* (surface-similar,
    # different problem). The rest are random different-problem pairs.
    hard_negative_fraction: float = 0.5
    max_pairs_per_problem: int = 3
    min_problems: int = 80
    # POSITIVE structural-divergence gate (type-4 enforcement):
    #   keep a same-problem pair only if token Jaccard is BELOW this
    #   AND AST-histogram cosine-distance is AT/ABOVE min_ast_distance.
    max_token_jaccard: float = 0.55
    min_ast_distance: float = 0.10
    # HARD-NEGATIVE gate: a different-problem pair qualifies as a hard
    # negative only if its token Jaccard is AT/ABOVE this (surface
    # looks similar, function differs).
    hard_neg_min_jaccard: float = 0.40


@dataclass
class SampleReport:
    """What the sampler produced — counts + the parallel unfiltered
    positive set used for the contamination-delta line in the report."""

    config: SamplerConfig
    positives: list[ClonePair] = field(default_factory=list)
    negatives_hard: list[ClonePair] = field(default_factory=list)
    negatives_random: list[ClonePair] = field(default_factory=list)
    # Same-problem pairs WITHOUT the structural-divergence filter, same
    # count as `positives`. Evaluating on these too exposes how much of
    # an unfiltered "all same-problem = clone" score is just type-1/2
    # contamination.
    positives_unfiltered: list[ClonePair] = field(default_factory=list)
    problems_used: int = 0

    def labeled(self) -> list[ClonePair]:
        """The headline set: type-4-enforced positives + both negative
        strata."""
        return self.positives + self.negatives_hard + self.negatives_random

    def unfiltered_labeled(self) -> list[ClonePair]:
        """The contamination-baseline set: unfiltered same-problem
        positives + the same negatives."""
        return (
            self.positives_unfiltered
            + self.negatives_hard
            + self.negatives_random
        )


# ── code-shape helpers (local, fast — no model) ─────────────────────────


def _token_set(src: str) -> set[str]:
    """Multiset-flattened token set (names/ops/keywords/numbers).

    Uses the stdlib tokenizer; on a syntax error we fall back to a
    cheap whitespace split so a weird submission doesn't crash the
    sampler (CodeNet is real-world; a few files don't tokenize).
    """
    out: set[str] = set()
    try:
        for tok in tokenize.tokenize(BytesIO(src.encode("utf-8")).readline):
            if tok.type in (tokenize.ENCODING, tokenize.NEWLINE,
                            tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
                            tokenize.ENDMARKER):
                continue
            s = tok.string.strip()
            if s:
                out.add(s)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        out = {w for w in src.split() if w}
    return out


def _ast_histogram(src: str) -> Counter[str]:
    """Counter of tree-sitter node types over the whole tree.

    Captures structural shape independent of identifiers — two
    functionally-equal solutions with different control flow have
    visibly different histograms; copy-paste does not.
    """
    tree = _PY_PARSER.parse(src.encode("utf-8"))
    hist: Counter[str] = Counter()
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        hist[n.type] += 1
        stack.extend(n.children)
    return hist


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _hist_distance(a: Counter[str], b: Counter[str]) -> float:
    """1 - cosine over node-type histograms → 0 identical shape,
    →1 maximally different shape."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 1.0
    # Clamp: identical histograms give cosine 1.0 ± FP epsilon, which
    # would otherwise yield a microscopically negative distance.
    return max(0.0, min(1.0, 1.0 - dot / (na * nb)))


# ── the sampler ─────────────────────────────────────────────────────────


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def sample(root: str | Path, config: SamplerConfig | None = None) -> SampleReport:
    """Build a labeled clone/non-clone set from a Python800 checkout.

    `root` is the extracted `Project_CodeNet_Python800` directory
    (problem subdirs of `.py` submissions). Deterministic given
    `config.seed`.
    """
    cfg = config or SamplerConfig()
    rng = random.Random(cfg.seed)
    root = Path(root)

    problems = sorted(d for d in root.iterdir() if d.is_dir())
    rng.shuffle(problems)
    if len(problems) < cfg.min_problems:
        raise ValueError(
            f"only {len(problems)} problems under {root}; "
            f"need ≥ {cfg.min_problems}"
        )

    report = SampleReport(config=cfg)
    used_problems: set[str] = set()

    # ---- positives (type-4 enforced) + parallel unfiltered ----
    for prob in problems:
        if len(report.positives) >= cfg.target_positives:
            break
        subs = sorted(prob.glob("*.py"))
        if len(subs) < 2:
            continue
        rng.shuffle(subs)
        kept = 0
        unfiltered_kept = 0
        # Walk candidate pairs from the shuffled list deterministically.
        for i in range(0, len(subs) - 1, 2):
            if (kept >= cfg.max_pairs_per_problem
                    or len(report.positives) >= cfg.target_positives):
                break
            a, b = subs[i], subs[i + 1]
            sa, sb = _read(a), _read(b)
            # Unfiltered baseline: same-problem pair, no shape gate.
            if unfiltered_kept < cfg.max_pairs_per_problem:
                report.positives_unfiltered.append(ClonePair(
                    code_a=sa, code_b=sb, is_clone=True,
                    clone_type="SAME-PROBLEM",
                ))
                unfiltered_kept += 1
            # Structural-divergence gate → genuine type-4.
            jac = _jaccard(_token_set(sa), _token_set(sb))
            dist = _hist_distance(_ast_histogram(sa), _ast_histogram(sb))
            if jac < cfg.max_token_jaccard and dist >= cfg.min_ast_distance:
                report.positives.append(ClonePair(
                    code_a=sa, code_b=sb, is_clone=True, clone_type="T4",
                ))
                kept += 1
                used_problems.add(prob.name)

    # Trim the unfiltered baseline to exactly match positive count so
    # the two evals are comparable size.
    report.positives_unfiltered = report.positives_unfiltered[: len(report.positives)]

    # ---- negatives: hard (surface-similar, diff problem) + random ----
    n_hard = int(cfg.target_negatives * cfg.hard_negative_fraction)
    n_rand = cfg.target_negatives - n_hard

    # Build a flat pool of (problem, path) for cross-problem sampling.
    pool: list[tuple[str, Path]] = []
    for prob in problems[: max(cfg.min_problems, 200)]:
        ps = sorted(prob.glob("*.py"))
        rng.shuffle(ps)
        for p in ps[:8]:  # a few per problem keeps the pool diverse
            pool.append((prob.name, p))
    rng.shuffle(pool)

    # Random negatives: different-problem pairs, no surface constraint.
    while len(report.negatives_random) < n_rand and len(pool) >= 2:
        (pa, fa), (pb, fb) = rng.sample(pool, 2)
        if pa == pb:
            continue
        report.negatives_random.append(ClonePair(
            code_a=_read(fa), code_b=_read(fb), is_clone=False,
            clone_type="NEG-random",
        ))

    # Hard negatives: scan cross-problem pairs, keep only surface-similar
    # ones (high token Jaccard despite different function).
    attempts = 0
    max_attempts = n_hard * 400  # bounded; hard negs are rare
    while len(report.negatives_hard) < n_hard and attempts < max_attempts:
        attempts += 1
        (pa, fa), (pb, fb) = rng.sample(pool, 2)
        if pa == pb:
            continue
        sa, sb = _read(fa), _read(fb)
        if _jaccard(_token_set(sa), _token_set(sb)) >= cfg.hard_neg_min_jaccard:
            report.negatives_hard.append(ClonePair(
                code_a=sa, code_b=sb, is_clone=False,
                clone_type="NEG-hard",
            ))

    report.problems_used = len(used_problems)
    return report


def fill_descriptions(
    pairs: list[ClonePair], llm, cache=None,
) -> list[ClonePair]:
    """Run the SHIPPING describer over each snippet's code (production
    path) and return new pairs with description_a/description_b set.

    This is the only place descriptions enter the CodeNet eval, and it
    never sees a problem id — exactly what the detector does at scan
    time. `cache` lets repeat runs skip re-describing identical code.
    """
    from otter_docs.describe import Describer

    describer = Describer(llm, cache)
    out: list[ClonePair] = []
    for p in pairs:
        da = describer.describe(
            kind="function", guid="codenet-a",
            language="python", source=p.code_a.encode("utf-8"),
        ).text
        db = describer.describe(
            kind="function", guid="codenet-b",
            language="python", source=p.code_b.encode("utf-8"),
        ).text
        out.append(ClonePair(
            code_a=p.code_a, code_b=p.code_b, is_clone=p.is_clone,
            clone_type=p.clone_type, description_a=da, description_b=db,
        ))
    return out


def dump_pairs(pairs: list[ClonePair], path: str | Path, n: int = 15) -> None:
    """Write the first `n` pairs to a human-readable file for eyeball
    review before committing to a long describe+embed run."""
    lines: list[str] = []
    for i, p in enumerate(pairs[:n]):
        jac = _jaccard(_token_set(p.code_a), _token_set(p.code_b))
        dist = _hist_distance(
            _ast_histogram(p.code_a), _ast_histogram(p.code_b)
        )
        lines.append("=" * 72)
        lines.append(
            f"# pair {i}  label={'CLONE' if p.is_clone else 'NOT-CLONE'}  "
            f"type={p.clone_type}  token_jaccard={jac:.2f}  "
            f"ast_dist={dist:.2f}"
        )
        lines.append("-" * 72 + "  A")
        lines.append(p.code_a.rstrip())
        lines.append("-" * 72 + "  B")
        lines.append(p.code_b.rstrip())
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
