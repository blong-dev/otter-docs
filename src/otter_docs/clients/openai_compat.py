"""OpenAI-compatible adapters.

Works with anything that speaks the OpenAI HTTP API: llama.cpp's
`llama-server`, vLLM, LM Studio, the real OpenAI/Anthropic-compatible
endpoints, and lots of self-hosted shims. The Ollama adapters in
`ollama.py` target Ollama's *native* `/api/generate` + `/api/embeddings`
shape; this module covers everything that landed on `/v1/...` instead.

Stdlib-only (urllib). Optional bearer token via `api_key`.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatError(RuntimeError):
    """Raised when the upstream returns an error or is unreachable."""


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        # 4xx/5xx responses still have a body worth surfacing.
        body = e.read() if hasattr(e, "read") else b""
        raise OpenAICompatError(
            f"HTTP {e.code} from {url}: {body[:300]!r}"
        ) from e
    except urllib.error.URLError as e:
        raise OpenAICompatError(f"request to {url} failed: {e.reason}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise OpenAICompatError(f"non-JSON response from {url}: {body[:120]!r}") from e


class OpenAICompatLLMClient:
    """LLM client over `/v1/chat/completions`.

    Sends the prompt as a single user message. If you need multi-turn
    or a system prompt, build it into the prompt string — the library
    only needs single-shot completion.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: float = 120.0,
        default_max_tokens: int = 256,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens

    def complete(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = kwargs.pop("max_tokens", None)
        # Ollama/llama.cpp call it `num_predict`; pass through whichever.
        if max_tokens is None:
            max_tokens = kwargs.pop("num_predict", self.default_max_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        for k in ("temperature", "top_p", "stop"):
            if k in kwargs:
                payload[k] = kwargs[k]
        body = _post_json(
            f"{self.base_url}/v1/chat/completions",
            payload, timeout=self.timeout, api_key=self.api_key,
        )
        choices = body.get("choices") or []
        if not choices:
            raise OpenAICompatError(f"no choices in response: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise OpenAICompatError(f"no content in choice[0].message: {choices[0]}")
        return content


class OpenAICompatEmbeddingClient:
    """Embedding client over `/v1/embeddings`.

    Sends the full batch in one request when the server supports it
    (real OpenAI, vLLM, llama-server with `--embeddings`). For servers
    that only accept a single input at a time we fall back transparently.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11435",
        api_key: str | None = None,
        dim: int = 768,
        timeout: float = 60.0,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._dim = dim
        self.timeout = timeout
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Try batched first; fall back to per-text if the server rejects
        # the list shape.
        try:
            body = _post_json(
                f"{self.base_url}/v1/embeddings",
                {"model": self.model, "input": texts},
                timeout=self.timeout, api_key=self.api_key,
            )
            return self._extract(body, expected=len(texts))
        except OpenAICompatError:
            # Per-text fallback. Common with llama-server builds that
            # only accept a string input.
            return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        body = _post_json(
            f"{self.base_url}/v1/embeddings",
            {"model": self.model, "input": text},
            timeout=self.timeout, api_key=self.api_key,
        )
        return self._extract(body, expected=1)[0]

    def _extract(self, body: dict[str, Any], *, expected: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise OpenAICompatError(
                f"expected {expected} embeddings, got {len(data) if isinstance(data, list) else type(data).__name__}"
            )
        out: list[list[float]] = []
        for item in data:
            vec = item.get("embedding")
            if not isinstance(vec, list) or len(vec) != self._dim:
                raise OpenAICompatError(
                    f"embedding dim mismatch: got {len(vec) if isinstance(vec, list) else type(vec).__name__}, expected {self._dim}"
                )
            if self.normalize:
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0:
                    vec = [x / norm for x in vec]
            out.append(list(vec))
        return out
