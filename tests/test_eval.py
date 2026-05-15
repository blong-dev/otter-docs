"""Phase 11 — eval harness + bench tests.

These validate the *math* (precision/recall/F1, threshold sweep,
loaders) with deterministic embedders. They do NOT measure model
quality — that needs a real embedder over the full GPTCloneBench and
is a documented local procedure, not a CI step.
"""

from __future__ import annotations

from otter_docs.clients import FakeEmbeddingClient
from otter_docs.eval import ClonePair, Metrics, evaluate, sweep_threshold
from otter_docs.eval_data import load_bundled, load_jsonl


# ── metrics math ────────────────────────────────────────────────────────


def test_metrics_precision_recall_f1():
    m = Metrics(threshold=0.9, true_positive=8, false_positive=2,
                true_negative=10, false_negative=4)
    assert abs(m.precision - 0.8) < 1e-9
    assert abs(m.recall - (8 / 12)) < 1e-9
    assert 0.0 < m.f1 < 1.0
    d = m.as_dict()
    assert d["tp"] == 8 and d["fn"] == 4


def test_metrics_zero_division_safe():
    m = Metrics(0.5, 0, 0, 0, 0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


# ── evaluate with a controllable embedder ──────────────────────────────


class _IdentityEmbedder:
    """Maps each distinct text to its own basis vector.

    Identical texts → identical vectors (cosine 1.0). Distinct texts
    → orthogonal vectors (cosine 0.0). Lets us construct exact
    clone/non-clone geometry: a pair whose two descriptions are the
    same string scores 1.0; different strings score 0.0.
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return 64

    def embed(self, texts):
        out = []
        for t in texts:
            idx = self._seen.setdefault(t, len(self._seen))
            v = [0.0] * 64
            v[idx % 64] = 1.0
            out.append(v)
        return out


def test_evaluate_perfect_separation():
    pairs = [
        ClonePair("a", "a", True, description_a="same", description_b="same"),
        ClonePair("b", "c", False, description_a="x", description_b="y"),
    ]
    m = evaluate(pairs, _IdentityEmbedder(), threshold=0.5)
    # Clone pair: identical description → cosine 1.0 ≥ 0.5 → TP.
    # Non-clone: orthogonal → cosine 0.0 < 0.5 → TN.
    assert m.true_positive == 1
    assert m.true_negative == 1
    assert m.false_positive == 0
    assert m.false_negative == 0
    assert m.f1 == 1.0


def test_evaluate_threshold_too_high_misses_clones():
    pairs = [ClonePair("a", "a", True, description_a="s", description_b="s")]
    # Even a perfect cosine of 1.0 is < threshold 1.01 → FN.
    m = evaluate(pairs, _IdentityEmbedder(), threshold=1.01)
    assert m.false_negative == 1
    assert m.true_positive == 0


def test_sweep_finds_best_f1():
    pairs = [
        ClonePair("a", "a", True, description_a="same", description_b="same"),
        ClonePair("b", "b", True, description_a="same2", description_b="same2"),
        ClonePair("c", "d", False, description_a="p", description_b="q"),
    ]
    sweep = sweep_threshold(pairs, _IdentityEmbedder(), start=0.1, stop=0.9, step=0.1)
    # Identical-description clones score 1.0, the negative scores 0.0,
    # so any threshold in (0, 1] separates perfectly → best F1 == 1.0.
    assert sweep.best.f1 == 1.0
    assert len(sweep.all) >= 8  # one Metrics per swept threshold


def test_sweep_embeds_each_pair_once():
    """The sweep must not re-embed per threshold (perf contract)."""
    class _CountingEmbedder(_IdentityEmbedder):
        calls = 0
        def embed(self, texts):
            type(self).calls += 1
            return super().embed(texts)

    pairs = [ClonePair("a", "a", True, description_a="s", description_b="s")] * 3
    emb = _CountingEmbedder()
    sweep_threshold(pairs, emb, start=0.1, stop=0.9, step=0.1)
    # 3 pairs, embedded once each (one embed() call per pair), regardless
    # of how many thresholds were swept.
    assert _CountingEmbedder.calls == 3


# ── loaders ─────────────────────────────────────────────────────────────


def test_load_bundled_fixture():
    pairs = load_bundled()
    assert len(pairs) >= 10
    kinds = {p.clone_type for p in pairs}
    assert "T3" in kinds
    assert "T4" in kinds
    assert "NEG" in kinds
    # Every NEG is labeled not-clone; every T3/T4 is labeled clone.
    for p in pairs:
        if p.clone_type == "NEG":
            assert p.is_clone is False
        else:
            assert p.is_clone is True


def test_load_jsonl_roundtrip(tmp_path):
    src = tmp_path / "pairs.jsonl"
    src.write_text(
        '{"code_a":"x","code_b":"y","is_clone":true,"clone_type":"T3"}\n'
        '\n'  # blank line tolerated
        '{"code_a":"p","code_b":"q","is_clone":false}\n'
    )
    pairs = load_jsonl(src)
    assert len(pairs) == 2
    assert pairs[0].is_clone is True
    assert pairs[0].clone_type == "T3"
    assert pairs[1].is_clone is False


def test_gptclonebench_missing_manifest_raises(tmp_path):
    from otter_docs.eval_data import load_gptclonebench
    import pytest

    with pytest.raises(FileNotFoundError, match="manifest"):
        list(load_gptclonebench(tmp_path))


# ── eval on the bundled fixture with FakeEmbedding (mechanics only) ─────


import os

import pytest


@pytest.mark.integration
def test_bundled_eval_with_real_embedder():
    """Real number, captured as a repeatable test.

    Opt-in: needs --run-integration AND OTTER_EMBED_URL pointing at an
    OpenAI-compatible embeddings endpoint. Asserts the mechanism still
    separates the bundled clones (F1 ≥ 0.9 at the best swept
    threshold) — a regression guard on the *approach*, not a
    production-scale claim.
    """
    url = os.environ.get("OTTER_EMBED_URL")
    if not url:
        pytest.skip("OTTER_EMBED_URL not set")
    from otter_docs.clients import OpenAICompatEmbeddingClient
    from otter_docs.eval import sweep_threshold

    emb = OpenAICompatEmbeddingClient(
        model=os.environ.get("OTTER_EMBED_MODEL", "nomic-embed-text"),
        base_url=url,
        dim=int(os.environ.get("OTTER_EMBED_DIM", "768")),
    )
    sweep = sweep_threshold(load_bundled(), emb, start=0.5, stop=0.95, step=0.025)
    assert sweep.best.f1 >= 0.9, sweep.best.as_dict()


def test_bundled_eval_runs_with_fake_embedder():
    """Smoke: the harness runs end-to-end on the bundled set.

    FakeEmbedding has no semantics, so we assert the harness produces
    a well-formed Metrics — NOT that F1 is good. Real quality numbers
    require a real embedder (documented local procedure).
    """
    pairs = load_bundled()
    m = evaluate(pairs, FakeEmbeddingClient(dim=32), threshold=0.8)
    total = m.true_positive + m.false_positive + m.true_negative + m.false_negative
    assert total == len(pairs)
    assert 0.0 <= m.f1 <= 1.0


# ── bench ───────────────────────────────────────────────────────────────


def test_benchmark_produces_phases(tmp_path):
    from otter_docs.bench import benchmark

    (tmp_path / "a.py").write_text(
        "def used():\n    return 1\n\n"
        "def caller():\n    return used()\n"
    )
    result = benchmark(tmp_path, name="benchtest")
    names = [p.name for p in result.phases]
    assert names == ["scan", "resolve", "findings"]
    assert result.total_seconds >= 0.0
    assert "benchtest" in result.report()
