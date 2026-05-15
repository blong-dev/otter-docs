"""Client tests — fakes always run; Ollama tests are opt-in."""

from __future__ import annotations

import os

import pytest

from otter_docs.clients import (
    FakeEmbeddingClient,
    FakeLLMClient,
    OllamaEmbeddingClient,
    OllamaLLMClient,
)

# ── fakes ──────────────────────────────────────────────────────────────


def test_fake_llm_records_prompt():
    llm = FakeLLMClient()
    out = llm.complete("describe foo")
    assert "describe foo" in out
    assert llm.calls == ["describe foo"]


def test_fake_llm_custom_prefix():
    llm = FakeLLMClient(prefix="MOCK")
    assert llm.complete("anything").startswith("MOCK:")


def test_fake_embedding_dim():
    emb = FakeEmbeddingClient(dim=16)
    vs = emb.embed(["a", "b"])
    assert len(vs) == 2
    assert all(len(v) == 16 for v in vs)


def test_fake_embedding_is_deterministic():
    emb = FakeEmbeddingClient(dim=8)
    a = emb.embed(["same text"])[0]
    b = emb.embed(["same text"])[0]
    assert a == b


def test_fake_embedding_distinguishes_texts():
    emb = FakeEmbeddingClient(dim=8)
    a, b = emb.embed(["text one", "text two"])
    assert a != b


def test_fake_embedding_returns_unit_vectors():
    emb = FakeEmbeddingClient(dim=8)
    (v,) = emb.embed(["any"])
    norm_sq = sum(x * x for x in v)
    assert 0.999 < norm_sq < 1.001


def test_fake_embedding_calls_history():
    emb = FakeEmbeddingClient(dim=4)
    emb.embed(["a"])
    emb.embed(["b", "c"])
    assert emb.calls == [["a"], ["b", "c"]]


# ── Ollama (opt-in) ────────────────────────────────────────────────────


pytestmark_ollama = pytest.mark.integration


@pytest.mark.integration
def test_ollama_llm_completes():
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")
    llm = OllamaLLMClient(
        base_url=os.environ["OLLAMA_BASE_URL"],
        model=os.environ.get("OLLAMA_LLM_MODEL", "qwen3.5:9b"),
    )
    out = llm.complete("Say 'ok'.", num_predict=10)
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.integration
def test_ollama_embedding_dim_matches():
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    emb = OllamaEmbeddingClient(
        base_url=os.environ["OLLAMA_BASE_URL"], model=model, dim=768
    )
    (v,) = emb.embed(["hello world"])
    assert len(v) == 768
    norm = sum(x * x for x in v) ** 0.5
    assert 0.99 < norm < 1.01
