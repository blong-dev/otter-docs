"""Ollama adapters for LLMClient and EmbeddingClient.

Stdlib-only — we use urllib so otter-docs's default install doesn't
pull in httpx/requests. Ollama runs as a local HTTP service, so a
synchronous request fits the protocol naturally.

Defaults match the user's stack: qwen3.5:9b for completion, the
nomic-embed-text model for embeddings (768-dim).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.5:9b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_DIM = 768


class OllamaError(RuntimeError):
    """Raised when Ollama isn't reachable or returns an error.

    Made distinct from generic exceptions so callers can decide whether
    a missing local model is fatal or recoverable (e.g., skip embedding
    enrichment and proceed with AST-only scan).
    """


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama request to {url} failed: {e.reason}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned non-JSON from {url}: {body[:120]!r}") from e


class OllamaLLMClient:
    """LLM client backed by Ollama's `/api/generate`.

    Parameters
    ----------
    model :
        Model tag in Ollama (e.g. "qwen3.5:9b"). Must already be pulled
        on the Ollama host.
    base_url :
        Where Ollama is listening. Default points at localhost.
    options :
        Extra options forwarded to Ollama (`temperature`, `num_predict`,
        etc). Kwargs to `.complete()` are merged on top.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        options: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.options = dict(options or {})
        self.timeout = timeout

    def complete(self, prompt: str, **kwargs: Any) -> str:
        opts = {**self.options, **kwargs}
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if opts:
            payload["options"] = opts
        body = _post_json(f"{self.base_url}/api/generate", payload, timeout=self.timeout)
        if "response" not in body:
            raise OllamaError(f"Ollama /api/generate returned no 'response' field: {body}")
        return body["response"]


class OllamaEmbeddingClient:
    """Embedding client backed by Ollama's `/api/embeddings`.

    Ollama's embedding endpoint takes one text at a time; we batch in
    Python with a single HTTP request per text. For local models on
    modern hardware this is plenty fast; if we ever hit a Phase-4 scan
    that's bound by embedding throughput, we'll switch to the
    `/api/embed` plural endpoint added in newer Ollama builds.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dim: int = DEFAULT_EMBEDDING_DIM,
        timeout: float = 60.0,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim = dim
        self.timeout = timeout
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            body = _post_json(
                f"{self.base_url}/api/embeddings",
                {"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            vec = body.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise OllamaError(
                    f"Ollama /api/embeddings returned no embedding for "
                    f"model={self.model}: {body}"
                )
            if len(vec) != self._dim:
                raise OllamaError(
                    f"Ollama returned embedding of dim {len(vec)} but client "
                    f"is configured for dim {self._dim}. Reconfigure or "
                    f"switch models."
                )
            if self.normalize:
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
            out.append(list(vec))
        return out
