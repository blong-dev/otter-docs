"""Clone-detection evaluation harness.

Measures the `redundancy.semantic_equivalence` approach against
labeled clone pairs. The similarity computation here is the *same*
one the detector uses (cosine over the description vector, optionally
gated by code-vector cosine) so the eval measures the real product,
not a proxy.

Honest scope:
  - CI runs this against a tiny bundled fixture with FakeEmbedding.
    That validates the harness math and guards against regressions in
    the similarity/threshold logic. It does NOT measure model quality
    — FakeEmbedding has no semantic understanding.
  - Real numbers (precision/recall/F1 vs C4's ~0.7 on GPTCloneBench)
    come from a local run against a real embedder over the actual
    GPTCloneBench dataset. That's a documented procedure, not a CI
    step — CI has neither the dataset nor a real embedder.

We publish whatever the local run produces, honestly, in the README.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from otter_docs.clients.base import EmbeddingClient


@dataclass(frozen=True)
class ClonePair:
    """One labeled pair. `is_clone` is ground truth.

    `clone_type` is metadata only (e.g. "T3", "T4") — used to break
    metrics down by difficulty, never as a feature.
    """

    code_a: str
    code_b: str
    is_clone: bool
    clone_type: str = ""
    description_a: str = ""
    description_b: str = ""


@dataclass(frozen=True)
class Metrics:
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
        }


@dataclass
class SweepResult:
    best: Metrics
    all: list[Metrics] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * y for x, y in zip(b, b, strict=True)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _pair_similarity(
    pair: ClonePair, embedder: EmbeddingClient,
) -> float:
    """Similarity score for a pair, mirroring the detector.

    The detector ranks on the *description* vector (HyDE-style: the
    LLM's prose abstracts away surface differences). When a pair
    carries explicit descriptions we embed those; otherwise we fall
    back to embedding the code directly (still a valid signal, just
    closer to a source-trained model's behavior).
    """
    if pair.description_a and pair.description_b:
        va, vb = embedder.embed([pair.description_a, pair.description_b])
    else:
        va, vb = embedder.embed([pair.code_a, pair.code_b])
    return _cosine(va, vb)


def evaluate(
    pairs: Iterable[ClonePair],
    embedder: EmbeddingClient,
    *,
    threshold: float,
) -> Metrics:
    """Score `pairs` at a fixed decision threshold.

    A pair is predicted clone iff similarity ≥ threshold.
    """
    tp = fp = tn = fn = 0
    for pair in pairs:
        sim = _pair_similarity(pair, embedder)
        predicted = sim >= threshold
        if predicted and pair.is_clone:
            tp += 1
        elif predicted and not pair.is_clone:
            fp += 1
        elif not predicted and pair.is_clone:
            fn += 1
        else:
            tn += 1
    return Metrics(threshold, tp, fp, tn, fn)


def sweep_threshold(
    pairs: Iterable[ClonePair],
    embedder: EmbeddingClient,
    *,
    start: float = 0.50,
    stop: float = 0.99,
    step: float = 0.01,
) -> SweepResult:
    """Sweep thresholds, return the F1-maximizing point plus the curve.

    Materializes `pairs` once (we score every pair at every threshold;
    re-embedding per threshold would be wasteful). Embeds each pair a
    single time, caches the similarity, then thresholds in-memory.
    """
    materialized = list(pairs)
    sims = [_pair_similarity(p, embedder) for p in materialized]

    results: list[Metrics] = []
    t = start
    while t <= stop + 1e-9:
        tp = fp = tn = fn = 0
        for pair, sim in zip(materialized, sims, strict=True):
            predicted = sim >= t
            if predicted and pair.is_clone:
                tp += 1
            elif predicted and not pair.is_clone:
                fp += 1
            elif not predicted and pair.is_clone:
                fn += 1
            else:
                tn += 1
        results.append(Metrics(round(t, 4), tp, fp, tn, fn))
        t += step

    best = max(results, key=lambda m: (m.f1, m.precision))
    return SweepResult(best=best, all=results)
