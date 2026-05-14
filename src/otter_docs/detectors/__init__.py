"""Detector Protocol, registry, and built-in detectors.

A Detector takes a (repo, graph_backend) and returns a list of
Finding records. The framework is intentionally cost-tier-aware so
callers can budget which detectors run on which scan:

  static            — pure graph queries, no model calls. Cheap.
  embedding         — uses indexed vectors. Cheap if vectors exist.
  llm_direct        — calls the LLM directly. Expensive.

Detectors register at import time via `register()` so consumer code
can `from otter_docs.detectors import run_all` without needing to
know which ones live where. Library-defined detectors are imported
eagerly here; user/plugin detectors register via Python entry points
in a later phase (see spec).
"""

from __future__ import annotations

from otter_docs.detectors.base import Detector, register, registry, run_all

__all__ = ["Detector", "register", "registry", "run_all"]


def _bootstrap() -> None:
    # Importing the modules triggers their `register()` calls. Same
    # pattern as parsers/__init__.py.
    from otter_docs.detectors import dead_code as _dc  # noqa: F401
    from otter_docs.detectors import large_function as _lf  # noqa: F401
    from otter_docs.detectors import empty_module as _em  # noqa: F401


_bootstrap()
