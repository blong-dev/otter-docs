"""Core data models for otter-docs.

Symbols, locations, and the three primary graph node types: Module, Function,
Class. These are what backends store and what detectors consume. The
description / code / docstring vectors are carried separately (on the node
records below) because they may or may not be populated depending on which
phase of indexing has run.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Language(str, Enum):
    """Languages whose AST otter-docs can parse."""

    PYTHON = "python"
    GO = "go"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    UNKNOWN = "unknown"


class Location(BaseModel):
    """A point in source: file path plus optional line span and stable GUID.

    Paths are repo-relative (e.g. `gnosis/agents/flow/__init__.py`), not
    absolute. The GUID is the stable identifier carried in `# guid:` comments
    for Python; for languages without comment-based GUIDs, it's a content-hash
    surrogate.
    """

    model_config = ConfigDict(frozen=True)

    repo: str  # repo name (e.g. "v3", "icarus")
    path: str  # repo-relative path
    guid: str | None = None
    line: int | None = None
    end_line: int | None = None


class Edge(BaseModel):
    """A typed directed edge between two symbols.

    `kind` is the relationship type. The library uses a small fixed set
    (DEFINED_IN, IMPORTS, CALLS, MEMBER_OF). Plugins may add their own.
    """

    kind: str  # "DEFINED_IN", "IMPORTS", "CALLS", "MEMBER_OF", ...
    src_id: str  # source node id (GUID or path)
    dst_id: str  # destination node id


class ModuleRecord(BaseModel):
    """A source file as a graph node."""

    repo: str
    path: str  # repo-relative; primary key with `repo`
    language: Language
    docstring: str = ""
    imports: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    # Vectors populated by the indexing pipeline; may be None pre-indexing.
    description_vec: list[float] | None = None
    code_vec: list[float] | None = None
    docstring_vec: list[float] | None = None


class FunctionRecord(BaseModel):
    """A function or method definition as a graph node.

    Methods carry their containing class via a MEMBER_OF edge separately;
    they are not embedded inside the class record.
    """

    repo: str
    guid: str  # stable id (comment GUID for Python; content-hash elsewhere)
    name: str
    module_path: str  # repo-relative path of defining module
    line: int
    end_line: int
    docstring: str = ""
    args: list[str] = Field(default_factory=list)
    returns: str = ""
    is_async: bool = False
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    description_vec: list[float] | None = None
    code_vec: list[float] | None = None
    docstring_vec: list[float] | None = None


class ClassRecord(BaseModel):
    """A class definition as a graph node."""

    repo: str
    guid: str
    name: str
    module_path: str
    line: int
    end_line: int
    docstring: str = ""
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    description_vec: list[float] | None = None
    code_vec: list[float] | None = None
    docstring_vec: list[float] | None = None


class SimilarityHit(BaseModel):
    """One result row from `find_similar`: a node identifier and a score."""

    node_kind: str  # "module" | "function" | "class"
    node_id: str  # GUID for fn/class, path for module
    similarity: float  # cosine similarity in [0, 1]; higher is more similar


class VectorKind(str, Enum):
    """Which of the three vectors a similarity query operates over."""

    DESCRIPTION = "description"
    CODE = "code"
    DOCSTRING = "docstring"


# Re-export the dataclasses for external consumers.
__all__ = [
    "ClassRecord",
    "Edge",
    "FunctionRecord",
    "Language",
    "Location",
    "ModuleRecord",
    "SimilarityHit",
    "VectorKind",
]
