"""Agent integration surface.

Four sub-modules, mirroring the spec:

  schemas   typed product + grading shapes + JSON Schema export
  prompts   drop-in system-prompt fragments
  tools     callable + MCP-spec-emittable tool definitions
  harness   the one-shot Harness that drives findings → grade report

The library is the source of truth; the separate `otter-docs-mcp`
package is a thin transport wrapper over `tools.as_mcp_specs`.
"""

from __future__ import annotations

from otter_docs.agent.harness import Harness
from otter_docs.agent.schemas import (
    Finding,
    Grade,
    GradeReport,
    Recommendation,
    Review,
    json_schema,
)
from otter_docs.agent.tools import Tool, as_mcp_specs, build_tools

__all__ = [
    "Finding",
    "Grade",
    "GradeReport",
    "Harness",
    "Recommendation",
    "Review",
    "Tool",
    "as_mcp_specs",
    "build_tools",
    "json_schema",
]
