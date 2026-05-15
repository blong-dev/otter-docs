"""otter-docs — polyglot codebase inspection library for agent-driven development."""

from otter_docs.backends import GraphBackend, SqliteBackend
from otter_docs.clients import (
    EmbeddingClient,
    FakeEmbeddingClient,
    FakeLLMClient,
    LLMClient,
    OllamaEmbeddingClient,
    OllamaLLMClient,
)
from otter_docs.describe import Describer, Description, SqliteDescriptionCache
from otter_docs.enrich import EnrichReport, Enricher
from otter_docs.findings import Finding, Recommendation
from otter_docs.llm_direct import Review
from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language,
    Location,
    ModuleRecord,
    SimilarityHit,
    VectorKind,
)
from otter_docs.repo import Repo, ScanReport
from otter_docs.resolvers.base import ResolveReport

__version__ = "0.1.0.dev0"

__all__ = [
    "ClassRecord",
    "Describer",
    "Description",
    "Edge",
    "EnrichReport",
    "Enricher",
    "Finding",
    "Recommendation",
    "EmbeddingClient",
    "FakeEmbeddingClient",
    "FakeLLMClient",
    "FunctionRecord",
    "GraphBackend",
    "LLMClient",
    "Language",
    "Location",
    "ModuleRecord",
    "OllamaEmbeddingClient",
    "OllamaLLMClient",
    "Repo",
    "ResolveReport",
    "Review",
    "ScanReport",
    "SimilarityHit",
    "SqliteBackend",
    "SqliteDescriptionCache",
    "VectorKind",
]
