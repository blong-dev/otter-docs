"""otter-docs — polyglot codebase inspection library for agent-driven development."""

from otter_docs.backends import GraphBackend, SqliteBackend
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
from otter_docs.repo import Repo

__version__ = "0.1.0.dev0"

__all__ = [
    "ClassRecord",
    "Edge",
    "FunctionRecord",
    "GraphBackend",
    "Language",
    "Location",
    "ModuleRecord",
    "Repo",
    "SimilarityHit",
    "SqliteBackend",
    "VectorKind",
]
