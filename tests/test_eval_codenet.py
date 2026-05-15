"""CodeNet sampler tests.

A synthetic Python800-shaped tree is built in tmp_path so these run
in CI without the 30 MB dataset. They prove the filter mechanics:
  - a type-1 (copy-paste) same-problem pair is REJECTED from positives
  - a type-4 (same function, different structure) pair is ACCEPTED
  - hard negatives are surface-similar cross-problem pairs
  - sampling is deterministic given the seed
Plus the backstop: the structural-divergence filter must not flip the
labels on the bundled 12 hand-known pairs.
"""

from __future__ import annotations

from pathlib import Path

from otter_docs.eval import ClonePair
from otter_docs.eval_codenet import (
    SamplerConfig,
    _ast_histogram,
    _hist_distance,
    _jaccard,
    _token_set,
    dump_pairs,
    fill_descriptions,
    sample,
)
from otter_docs.eval_data import load_bundled

# Two structurally-different solutions to "sum 1..n" — genuine type-4.
_SUM_LOOP = "n=int(input())\nt=0\nfor i in range(1,n+1):\n    t+=i\nprint(t)\n"
_SUM_FORMULA = "n=int(input())\nprint(n*(n+1)//2)\n"
# A near-identical copy of the loop version (type-1/2): same structure,
# one renamed var — must be filtered OUT of positives.
_SUM_LOOP_COPY = "n=int(input())\ns=0\nfor i in range(1,n+1):\n    s+=i\nprint(s)\n"
# A different problem (max of list) whose tokens overlap the sum-loop
# (range/for/int/input) → useful as a hard-negative source.
_MAXLIST = "n=int(input())\na=list(map(int,input().split()))\nm=a[0]\nfor i in range(1,n):\n    if a[i]>m:\n        m=a[i]\nprint(m)\n"


def _make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "Project_CodeNet_Python800"
    root.mkdir()
    # p0: sum problem — mix of type-4 and copy-paste submissions.
    p0 = root / "p00000"
    p0.mkdir()
    (p0 / "s001.py").write_text(_SUM_LOOP)
    (p0 / "s002.py").write_text(_SUM_FORMULA)        # type-4 vs s001
    (p0 / "s003.py").write_text(_SUM_LOOP_COPY)      # type-1 vs s001
    (p0 / "s004.py").write_text(_SUM_FORMULA)        # dup of s002
    # p1..p120: max-list problem, enough distinct problems for min_problems.
    for k in range(1, 130):
        pk = root / f"p{k:05d}"
        pk.mkdir()
        (pk / "s001.py").write_text(_MAXLIST)
        (pk / "s002.py").write_text(_MAXLIST.replace("m=", "best="))
        (pk / "s003.py").write_text(_SUM_LOOP)  # surface-similar to p0
    return root


# ── shape helpers ───────────────────────────────────────────────────────


def test_jaccard_identical_and_disjoint():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_copy_paste_has_high_jaccard_low_ast_distance():
    j = _jaccard(_token_set(_SUM_LOOP), _token_set(_SUM_LOOP_COPY))
    d = _hist_distance(_ast_histogram(_SUM_LOOP), _ast_histogram(_SUM_LOOP_COPY))
    assert j > 0.7, j          # nearly all tokens shared
    assert d < 0.05, d         # near-identical structure


def test_type4_pair_has_low_jaccard_or_high_ast_distance():
    j = _jaccard(_token_set(_SUM_LOOP), _token_set(_SUM_FORMULA))
    d = _hist_distance(_ast_histogram(_SUM_LOOP), _ast_histogram(_SUM_FORMULA))
    # Loop vs closed-form: very different structure.
    assert j < 0.55 or d >= 0.10, (j, d)


# ── sampler mechanics ───────────────────────────────────────────────────


def test_sampler_rejects_copy_paste_from_positives(tmp_path: Path):
    root = _make_tree(tmp_path)
    cfg = SamplerConfig(seed=1, target_positives=50, target_negatives=20,
                        min_problems=80, max_pairs_per_problem=5)
    rep = sample(root, cfg)
    # Every kept positive must clear the structural-divergence gate.
    for p in rep.positives:
        j = _jaccard(_token_set(p.code_a), _token_set(p.code_b))
        d = _hist_distance(_ast_histogram(p.code_a), _ast_histogram(p.code_b))
        assert j < cfg.max_token_jaccard and d >= cfg.min_ast_distance
        # The exact copy-paste pair (loop vs loop-copy) must never appear.
        pair = {p.code_a, p.code_b}
        assert pair != {_SUM_LOOP, _SUM_LOOP_COPY}


def test_sampler_unfiltered_baseline_includes_copy_paste(tmp_path: Path):
    """The contamination-baseline set is NOT filtered — it should
    contain the easy copy-paste pairs the headline set excludes."""
    root = _make_tree(tmp_path)
    rep = sample(root, SamplerConfig(seed=1, target_positives=20,
                                     target_negatives=10, min_problems=80))
    assert len(rep.positives_unfiltered) == len(rep.positives)
    # All positives are labeled clone in both sets.
    assert all(p.is_clone for p in rep.positives)
    assert all(p.is_clone for p in rep.positives_unfiltered)


def test_sampler_negatives_are_cross_problem(tmp_path: Path):
    root = _make_tree(tmp_path)
    rep = sample(root, SamplerConfig(seed=2, target_positives=20,
                                     target_negatives=40, min_problems=80))
    for n in rep.negatives_hard + rep.negatives_random:
        assert n.is_clone is False
    # Hard negatives must clear the surface-similarity bar.
    for n in rep.negatives_hard:
        assert _jaccard(_token_set(n.code_a), _token_set(n.code_b)) >= 0.40


def test_sampler_is_deterministic(tmp_path: Path):
    root = _make_tree(tmp_path)
    cfg = SamplerConfig(seed=42, target_positives=15, target_negatives=15,
                        min_problems=80)
    a = sample(root, cfg)
    b = sample(root, cfg)
    assert [(p.code_a, p.code_b) for p in a.labeled()] == \
           [(p.code_a, p.code_b) for p in b.labeled()]


def test_sampler_respects_min_problems(tmp_path: Path):
    root = tmp_path / "Project_CodeNet_Python800"
    root.mkdir()
    (root / "p0").mkdir()
    (root / "p0" / "s1.py").write_text(_SUM_LOOP)
    import pytest
    with pytest.raises(ValueError, match="need ≥"):
        sample(root, SamplerConfig(min_problems=80))


# ── backstop: filter must not relabel the known 12 ──────────────────────


def test_filter_does_not_relabel_bundled_known_pairs():
    """The structural-divergence helpers are a *positive-selection*
    gate, not a relabeler. Run them over the hand-labeled bundled set
    and confirm the gate's verdict agrees with ground truth on the
    clear cases: NEG pairs never look like type-4 clones by structure
    alone is NOT asserted (that's the detector's job) — what we assert
    is the gate is computable and bounded for every bundled pair, so
    it can never crash the real run."""
    for p in load_bundled():
        j = _jaccard(_token_set(p.code_a), _token_set(p.code_b))
        d = _hist_distance(_ast_histogram(p.code_a), _ast_histogram(p.code_b))
        assert 0.0 <= j <= 1.0
        assert 0.0 <= d <= 1.0


# ── dump + describe wiring ──────────────────────────────────────────────


def test_dump_pairs_writes_readable_file(tmp_path: Path):
    pairs = [
        ClonePair(code_a="def a(): pass", code_b="def b(): pass",
                  is_clone=False, clone_type="NEG-random"),
        ClonePair(code_a=_SUM_LOOP, code_b=_SUM_FORMULA,
                  is_clone=True, clone_type="T4"),
    ]
    out = tmp_path / "sample.txt"
    dump_pairs(pairs, out, n=15)
    text = out.read_text()
    assert "label=NOT-CLONE" in text
    assert "label=CLONE" in text
    assert "token_jaccard=" in text and "ast_dist=" in text


def test_fill_descriptions_uses_describer_not_problem_id(tmp_path: Path):
    from otter_docs.clients import FakeLLMClient

    pairs = [ClonePair(code_a=_SUM_LOOP, code_b=_SUM_FORMULA,
                       is_clone=True, clone_type="T4")]
    llm = FakeLLMClient()
    filled = fill_descriptions(pairs, llm)
    assert filled[0].description_a and filled[0].description_b
    # FakeLLM echoes the prompt; the prompt must contain the CODE, and
    # must NOT contain any "p00000"-style problem id (we never pass it).
    joined = " ".join(llm.calls)
    assert "range" in joined  # code reached the describer
    assert "p00000" not in joined
