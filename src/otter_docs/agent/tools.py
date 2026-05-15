"""Tool definitions — the callable surface agents and MCP hosts use.

Each `Tool` bundles:
  - name          stable identifier ("otter_docs.findings")
  - description    one-liner for the model
  - input_schema   JSON Schema for the arguments
  - call           a Python callable bound to a Repo

`build_tools(repo, llm=None, embedder=None)` returns the full set
bound to a specific Repo. `as_mcp_specs(tools)` emits the
`{name, description, inputSchema}` dicts an MCP server needs — we
produce that shape here so the separate `otter-docs-mcp` package is
a thin transport wrapper with zero analysis logic.

Tools that need an LLM/embedder raise a clear error if those weren't
provided, rather than being silently absent — an agent asking for a
consolidation should hear "no LLM configured", not get a missing tool.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from otter_docs.clients.base import EmbeddingClient, LLMClient
from otter_docs.findings import Finding
from otter_docs.repo import Repo


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    call: Callable[..., Any]


def _require(obj: Any, what: str) -> Any:
    if obj is None:
        raise RuntimeError(
            f"This tool needs {what}; pass it to build_tools(...) to enable."
        )
    return obj


def build_tools(
    repo: Repo,
    *,
    llm: LLMClient | None = None,
    embedder: EmbeddingClient | None = None,
) -> list[Tool]:
    """Construct the tool set bound to `repo`.

    LLM/embedder-dependent tools are still listed when those clients
    are absent, but invoking them raises a clear RuntimeError. This
    keeps the advertised tool surface stable regardless of config —
    an agent sees the same catalog, and only learns about a missing
    dependency if it actually tries to use that capability.
    """

    def _scan(reset: bool = False) -> dict[str, Any]:
        report = repo.scan(reset=reset)
        return {
            "files_parsed": report.files_parsed,
            "modules": report.modules,
            "functions": report.functions,
            "classes": report.classes,
            "edges": report.edges,
            "errors": len(report.errors),
        }

    def _resolve() -> dict[str, Any]:
        reports = repo.resolve()
        return {
            lang.value: {"edges_emitted": rep.edges_emitted}
            for lang, rep in reports.items()
        }

    def _enrich() -> dict[str, Any]:
        report = repo.enrich(_require(llm, "an LLM client"),
                             _require(embedder, "an embedding client"))
        return {
            "modules_enriched": report.modules_enriched,
            "functions_enriched": report.functions_enriched,
            "classes_enriched": report.classes_enriched,
            "cache_hits": report.cache_hits,
            "errors": len(report.errors),
        }

    def _findings(
        kinds: list[str] | None = None,
        cost_tiers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result = repo.findings(
            kinds=set(kinds) if kinds else None,
            cost_tiers=set(cost_tiers) if cost_tiers else None,  # type: ignore[arg-type]
        )
        return [f.model_dump() for f in result]

    def _callgraph(guid: str, direction: str = "callers") -> list[str]:
        if direction == "callers":
            return repo.graph.callers_of(repo.name, guid)
        edges = repo.graph.edges_from(repo.name, guid, kind="CALLS")
        return [e.dst_id for e in edges]

    def _propose_consolidation(finding: dict[str, Any]) -> dict[str, Any]:
        f = Finding.model_validate(finding)
        rec = repo.propose_consolidation(f, _require(llm, "an LLM client"))
        return rec.model_dump()

    def _review_change(diff: str) -> dict[str, Any]:
        review = repo.review_change(diff, _require(llm, "an LLM client"))
        return review.model_dump()

    def _describe(guid: str | None = None, path: str | None = None) -> dict[str, Any] | None:
        desc = repo.describe(_require(llm, "an LLM client"), guid=guid, path=path)
        return desc.__dict__ if desc is not None else None

    return [
        Tool(
            name="otter_docs.scan",
            description="Walk the repo, parse ASTs, populate the code graph. Returns counts.",
            input_schema={
                "type": "object",
                "properties": {"reset": {"type": "boolean", "default": False}},
            },
            call=_scan,
        ),
        Tool(
            name="otter_docs.resolve",
            description="Run cross-file call resolution (jedi/tsserver/gopls). Returns edge counts per language.",
            input_schema={"type": "object", "properties": {}},
            call=_resolve,
        ),
        Tool(
            name="otter_docs.enrich",
            description="Generate three-vector embeddings + LLM descriptions for every symbol. Needs an LLM and embedder.",
            input_schema={"type": "object", "properties": {}},
            call=_enrich,
        ),
        Tool(
            name="otter_docs.findings",
            description="Run detectors; return typed Findings. Filter by kinds and/or cost_tiers.",
            input_schema={
                "type": "object",
                "properties": {
                    "kinds": {"type": "array", "items": {"type": "string"}},
                    "cost_tiers": {"type": "array", "items": {"type": "string"}},
                },
            },
            call=_findings,
        ),
        Tool(
            name="otter_docs.callgraph",
            description="List callers (or callees) of a function by guid.",
            input_schema={
                "type": "object",
                "properties": {
                    "guid": {"type": "string"},
                    "direction": {"type": "string", "enum": ["callers", "callees"], "default": "callers"},
                },
                "required": ["guid"],
            },
            call=_callgraph,
        ),
        Tool(
            name="otter_docs.propose_consolidation",
            description="Given a redundancy Finding, ask the LLM for a consolidation unified diff. Needs an LLM.",
            input_schema={
                "type": "object",
                "properties": {"finding": {"type": "object"}},
                "required": ["finding"],
            },
            call=_propose_consolidation,
        ),
        Tool(
            name="otter_docs.review_change",
            description="Ask the LLM to review a unified diff. Returns a structured Review. Needs an LLM.",
            input_schema={
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
            call=_review_change,
        ),
        Tool(
            name="otter_docs.describe",
            description="Describe a single symbol by guid (function/class) or path (module). Needs an LLM.",
            input_schema={
                "type": "object",
                "properties": {
                    "guid": {"type": "string"},
                    "path": {"type": "string"},
                },
            },
            call=_describe,
        ),
    ]


def as_mcp_specs(tools: list[Tool]) -> list[dict[str, Any]]:
    """Emit MCP tool specs: `{name, description, inputSchema}` per tool.

    This is the entire contract the separate otter-docs-mcp package
    needs from the library — it wires these to an MCP transport and
    routes calls back to `Tool.call`. Keeping the spec shape here means
    the MCP package carries zero analysis logic.
    """
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_schema,
        }
        for t in tools
    ]
