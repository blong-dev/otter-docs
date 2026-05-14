"""Client Protocols.

Keep these intentionally narrow. LLMClient is one method, EmbeddingClient
is one method plus the dim property. Anything fancier (streaming,
multi-turn, tool-calling) is the caller's responsibility — the library
itself only needs a black-box "give me text" and "give me vectors".
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """Single-shot text completion.

    Implementations should be synchronous and pure-ish — given the same
    prompt and kwargs, they should converge to consistent outputs.
    Async-only providers can wrap their calls in `asyncio.run` since
    otter-docs is sync end-to-end for v0.1.
    """

    def complete(self, prompt: str, **kwargs: Any) -> str: ...


class EmbeddingClient(Protocol):
    """Batch embedding.

    `embed` takes a list of texts and returns a parallel list of
    vectors. Implementations are expected to return unit-length vectors
    so similarity scoring works consistently across backends — both
    SqliteBackend and Neo4jBackend produce cosine-equivalent scores
    when inputs are unit-length.
    """

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
