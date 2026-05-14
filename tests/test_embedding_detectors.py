"""Embedding-tier detector tests.

These use a custom EmbeddingClient that returns hand-crafted vectors
so we can construct exact similarity scenarios — fake but deterministic,
the same trick the rest of the suite leans on.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from otter_docs import Repo
from otter_docs.backends import SqliteBackend
from otter_docs.clients import FakeLLMClient
from otter_docs.detectors import registry
from otter_docs.detectors.description_divergence import DescriptionDivergenceDetector
from otter_docs.detectors.semantic_equivalence import SemanticEquivalenceDetector


class _ControlledEmbedder:
    """Returns vectors from a (text → vector) map.

    Unknown texts return a unit vector pointing along axis 0 — keeps
    them all clustered together so we can target which pairs we want
    to test without coincidental similarity tripping the assertions.
    """

    def __init__(self, mapping: dict[str, list[float]], dim: int) -> None:
        self._mapping = mapping
        self._dim = dim
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            v = self._mapping.get(t)
            if v is None:
                # Default unit vector along axis 0. The mapping uses
                # vectors orthogonal to this so unrelated texts don't
                # accidentally pair up.
                v = [1.0] + [0.0] * (self._dim - 1)
            out.append(_normalize(v))
        return out


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


# ── built-ins are registered ────────────────────────────────────────────


def test_embedding_detectors_registered():
    assert "redundancy.semantic_equivalence" in registry()
    assert "description.divergence" in registry()


# ── semantic equivalence ───────────────────────────────────────────────


def test_semantic_equivalence_pairs_similar_functions(tmp_path: Path):
    """Two functions whose description vectors are identical → 1 Finding."""
    (tmp_path / "a.py").write_text(
        "def alpha(x):\n    return x + 1\n\n"
        "def beta(x):\n    return x + 1  # duplicate\n\n"
        "def gamma(x):\n    return x * 1000  # different\n"
    )

    # Craft an embedder that maps the LLM descriptions of alpha+beta
    # to the same vector (axis 1) and gamma to axis 2. Code vectors
    # match descriptions for simplicity.
    pair_vec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    odd_vec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    backend = SqliteBackend(":memory:", vector_dim=8)
    llm = FakeLLMClient()

    # Pre-scan to learn what descriptions the FakeLLM produces, so we
    # can map exactly those strings to our chosen vectors.
    with Repo(tmp_path, backend=backend) as repo:
        repo.scan()
        # First enrich pass uses an embedder that returns near-identical
        # vectors for alpha & beta and a different one for gamma.
        # We can't know the exact LLM-generated description text until
        # the describer runs, so use a non-deterministic mapping by
        # falling back to "anything not in the map -> pair_vec". To
        # isolate gamma, map its function source explicitly.
        mapping: dict[str, list[float]] = {}
        # Build the mapping by sniffing which prompts FakeLLM would
        # produce when called — easier: just set odd_vec for any text
        # containing "gamma" or "1000", and pair_vec for everything
        # else.
        # We bypass this by making a controlled embedder that decides
        # at embed time based on whether the text contains 'gamma'.
        class _RuleEmbedder:
            @property
            def dim(self) -> int:
                return 8
            def embed(self, texts):
                out = []
                for t in texts:
                    if "gamma" in t or "1000" in t:
                        out.append(_normalize(odd_vec))
                    else:
                        out.append(_normalize(pair_vec))
                return out
        emb = _RuleEmbedder()
        repo.enrich(llm, emb)

        findings = repo.findings(kinds={"redundancy.semantic_equivalence"})
        # alpha+beta should be flagged exactly once.
        assert len(findings) == 1
        names = set(findings[0].evidence["function_names"])
        assert names == {"alpha", "beta"}
        # gamma should NOT appear in the pair.
        assert "gamma" not in names


def test_semantic_equivalence_skips_when_below_threshold(tmp_path: Path):
    """When description similarity is below the threshold, no Finding."""
    (tmp_path / "a.py").write_text(
        "def alpha(): return 1\n\n"
        "def beta(): return 2\n"
    )

    class _OrthogonalEmbedder:
        """Each *unique* text gets its own basis vector → orthogonal pairs."""
        def __init__(self) -> None:
            self._seen: dict[str, int] = {}
        @property
        def dim(self) -> int:
            return 8
        def embed(self, texts):
            out = []
            for t in texts:
                idx = self._seen.setdefault(t, len(self._seen))
                v = [0.0] * 8
                v[idx % 8] = 1.0
                out.append(v)
            return out

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _OrthogonalEmbedder())
        findings = repo.findings(kinds={"redundancy.semantic_equivalence"})
        assert findings == []


def test_semantic_equivalence_threshold_configurable():
    det = SemanticEquivalenceDetector(description_threshold=0.7, code_threshold=0.5)
    assert det.description_threshold == 0.7
    assert det.code_threshold == 0.5


# ── description divergence ─────────────────────────────────────────────


def test_description_divergence_flags_misaligned_function(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "def aligned(): return 1\n\n"
        "def diverged(): return 1\n"
    )

    class _DivergenceEmbedder:
        @property
        def dim(self) -> int:
            return 8
        def embed(self, texts):
            out: list[list[float]] = []
            for t in texts:
                # `aligned`: both description and code map to axis 0.
                # `diverged`: description→axis 0, code→axis 1 (orthogonal).
                if "diverged" in t and ("Source:" not in t):
                    # This is the *code* text for diverged.
                    v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                out.append(v)
            return out

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _DivergenceEmbedder())
        findings = repo.findings(kinds={"description.divergence"})
        flagged_names = {f.evidence["function_name"] for f in findings}
        assert "diverged" in flagged_names


def test_description_divergence_threshold_configurable():
    det = DescriptionDivergenceDetector(threshold=0.7)
    assert det.threshold == 0.7


def test_description_divergence_skips_when_no_vectors(tmp_path: Path):
    """Without an enrich pass, code_vec/description_vec are None — no findings."""
    (tmp_path / "a.py").write_text("def f(): return 1\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        # No enrich call — vectors stay None.
        findings = repo.findings(kinds={"description.divergence"})
        assert findings == []
