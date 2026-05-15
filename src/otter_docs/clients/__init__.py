"""LLM + embedding client protocols and built-in adapters.

The two Protocols (LLMClient, EmbeddingClient) are the only surface
otter-docs reaches into for model-backed work. Anything that talks to
a model — describer, agent harness, future detectors — goes through
these.

Default adapters target Ollama (local-first). Anthropic / OpenAI
adapters are optional and live behind extras to keep the base install
free of those SDKs.

Test fakes (FakeLLMClient, FakeEmbeddingClient) live in
`otter_docs.clients.fake` so library tests don't need a live model.
"""

from __future__ import annotations

from otter_docs.clients.base import EmbeddingClient, LLMClient
from otter_docs.clients.fake import FakeEmbeddingClient, FakeLLMClient
from otter_docs.clients.ollama import OllamaEmbeddingClient, OllamaLLMClient
from otter_docs.clients.openai_compat import (
    OpenAICompatEmbeddingClient,
    OpenAICompatLLMClient,
)

__all__ = [
    "EmbeddingClient",
    "FakeEmbeddingClient",
    "FakeLLMClient",
    "LLMClient",
    "OllamaEmbeddingClient",
    "OllamaLLMClient",
    "OpenAICompatEmbeddingClient",
    "OpenAICompatLLMClient",
]
