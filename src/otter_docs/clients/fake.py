"""Deterministic test fakes for LLM and embedding clients.

Tests in otter-docs and downstream consumers should never need a live
model to verify wiring. These fakes are deterministic functions of
their input so test expectations are stable.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


class FakeLLMClient:
    """Returns a deterministic short summary based on the prompt hash.

    Designed for testing describer + cache wiring without spending
    tokens. The output references the first 80 characters of the
    prompt so tests can assert that prompts are reaching the client.
    """

    def __init__(self, prefix: str = "FAKE") -> None:
        self.prefix = prefix
        self.calls: list[str] = []  # prompts seen, for test assertions

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        snippet = " ".join(prompt.split())[:80]
        return f"{self.prefix}: {snippet}"


class FakeEmbeddingClient:
    """Hashes each text to a deterministic unit vector of the configured dim.

    The hash → float layout uses a uniform distribution from a SHA-256
    of the text. We then L2-normalize so similarity scoring works the
    same way it does with a real embedder. Identical texts → identical
    vectors. Distinct texts → effectively uncorrelated vectors.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.calls: list[list[str]] = []  # batches seen, for test assertions

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> list[float]:
        # Hash the text into enough bytes for `dim` floats. SHA-256 gives
        # us 32 bytes per round; we extend by suffixing the round index.
        floats: list[float] = []
        round_idx = 0
        while len(floats) < self._dim:
            digest = hashlib.sha256(f"{text}|{round_idx}".encode()).digest()
            for i in range(0, len(digest), 4):
                if len(floats) >= self._dim:
                    break
                # Convert 4 bytes → unsigned int → float in [-1, 1]
                u = int.from_bytes(digest[i : i + 4], "big")
                floats.append((u / 2**31) - 1.0)
            round_idx += 1
        # Normalize to unit length so similarity scores behave.
        norm = math.sqrt(sum(x * x for x in floats))
        if norm == 0:
            return floats
        return [x / norm for x in floats]
