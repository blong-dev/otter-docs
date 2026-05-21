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
        # We can't know the exact LLM-generated description text until
        # the describer runs, so decide the vector at embed time by
        # content: odd_vec for anything mentioning "gamma"/"1000",
        # pair_vec otherwise — making alpha & beta collide and gamma
        # stand apart.
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


def _collide_all_embedder(dim: int = 8):
    """Embedder that maps every text to the same axis-1 unit vector.

    Forces every pair to look maximally similar — useful for
    exercising shape classification without juggling per-text vectors.
    """
    vec = _normalize([0.0] + [1.0] + [0.0] * (dim - 2))

    class _CollideEmbedder:
        @property
        def dim(self) -> int:
            return dim

        def embed(self, texts):
            return [list(vec) for _ in texts]

    return _CollideEmbedder()


def test_semantic_equivalence_lifecycle_hook_emits_informational(tmp_path: Path):
    """Two `__init__` methods on different classes → shape='lifecycle_hook'.

    Even when description + code vectors collide perfectly, the
    Finding should land at LIFECYCLE_CONFIDENCE (0.05) — well below
    any default subscriber threshold — and `evidence.shape` should be
    'lifecycle_hook' so downstream readers can recognize it.
    """
    (tmp_path / "a.py").write_text(
        "class Foo:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n\n"
        "class Bar:\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n"
    )

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _collide_all_embedder())
        findings = repo.findings(kinds={"redundancy.semantic_equivalence"})
        assert len(findings) == 1
        f = findings[0]
        assert f.evidence["shape"] == "lifecycle_hook"
        assert f.confidence == 0.05
        assert f.evidence["function_names"] == ["__init__", "__init__"]
        # Recommendation should not say "Consolidate"
        assert "Consolidate" not in f.recommendation.summary
        assert "lifecycle" in f.recommendation.summary.lower()


def test_semantic_equivalence_sibling_methods_scales_confidence(tmp_path: Path):
    """Two `process` methods on different classes (same basename, no
    lifecycle-hook match) → shape='sibling_methods', confidence
    scaled by SIBLING_CONFIDENCE_SCALE (0.7)."""
    (tmp_path / "a.py").write_text(
        "class Pipeline:\n"
        "    def process(self, x):\n"
        "        return x + 1\n\n"
        "class Worker:\n"
        "    def process(self, x):\n"
        "        return x + 1\n"
    )

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _collide_all_embedder())
        findings = repo.findings(kinds={"redundancy.semantic_equivalence"})
        assert len(findings) == 1
        f = findings[0]
        assert f.evidence["shape"] == "sibling_methods"
        # description_similarity == 1.0 (collide), confidence = 1.0 * 0.7
        assert f.confidence == pytest.approx(0.7, abs=1e-6)
        assert "Consolidate" not in f.recommendation.summary
        assert "sibling" in f.recommendation.summary.lower()


def test_semantic_equivalence_skips_test_file_pairs(tmp_path: Path):
    """Pairs where either side is a test file are filtered out — tests
    are duplicative by design (parameterized cases, shared fixtures)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def helper(): return 1\n"
    )
    (tmp_path / "lib.py").write_text(
        "def helper(): return 1\n"
    )

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _collide_all_embedder())
        findings = repo.findings(kinds={"redundancy.semantic_equivalence"})
        # Even with perfect description+code collision, the test-file
        # filter drops the pair entirely.
        assert findings == []


def test_semantic_equivalence_default_threshold_is_0_95():
    """Default description_threshold raised from 0.92 to 0.95 (2026-05-21
    calibration) — anything below was producing too many false positives
    from generic LLM descriptions."""
    det = SemanticEquivalenceDetector()
    assert det.description_threshold == 0.95


def test_semantic_equivalence_likely_duplicate_shape(tmp_path: Path):
    """Different-named module-level functions → shape='likely_duplicate'
    with the original Consolidate-style recommendation."""
    (tmp_path / "a.py").write_text(
        "def alpha(x):\n    return x + 1\n\n"
        "def beta(x):\n    return x + 1\n"
    )

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _collide_all_embedder())
        findings = repo.findings(kinds={"redundancy.semantic_equivalence"})
        assert len(findings) == 1
        f = findings[0]
        assert f.evidence["shape"] == "likely_duplicate"
        assert f.confidence == pytest.approx(1.0, abs=1e-6)
        assert "Consolidate" in f.recommendation.summary


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


def test_description_divergence_is_llm_direct_tier():
    """Tier moved to llm_direct (2026-05-21) so it doesn't fire on a
    default findings() call — cosine-only signal is too noisy without
    an LLM judge."""
    det = DescriptionDivergenceDetector()
    assert det.cost_tier == "llm_direct"


def test_description_divergence_excluded_from_default_findings(tmp_path: Path):
    """A default (unfiltered) findings() call must NOT include
    description.divergence, but an explicit kinds= request still runs it."""
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
                if "diverged" in t and ("Source:" not in t):
                    v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                out.append(v)
            return out

    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        repo.enrich(FakeLLMClient(), _DivergenceEmbedder())
        # Default call: no description.divergence findings.
        default = repo.findings()
        assert not any(f.kind == "description.divergence" for f in default)
        # Explicit kinds: it runs.
        explicit = repo.findings(kinds={"description.divergence"})
        assert any(f.kind == "description.divergence" for f in explicit)
        # Explicit llm_direct tier: also runs.
        tiered = repo.findings(cost_tiers={"llm_direct"})
        assert any(f.kind == "description.divergence" for f in tiered)


def test_description_divergence_skips_when_no_vectors(tmp_path: Path):
    """Without an enrich pass, code_vec/description_vec are None — no findings."""
    (tmp_path / "a.py").write_text("def f(): return 1\n")
    with Repo(tmp_path, backend=SqliteBackend(":memory:", vector_dim=8)) as repo:
        repo.scan()
        # No enrich call — vectors stay None.
        findings = repo.findings(kinds={"description.divergence"})
        assert findings == []
