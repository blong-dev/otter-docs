"""User-facing entry point.

The shape `from otter_docs import Repo; repo = Repo("/path/to/code")` is the
contract every consumer (humans, agents, MCP servers) uses. Phase 1 ships
the skeleton with the backend wired up; scanning and findings come in
later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from otter_docs.backends import GraphBackend, SqliteBackend

if TYPE_CHECKING:
    pass


class Repo:
    """An otter-docs view over a single git repository.

    Parameters
    ----------
    root :
        Filesystem path to the repository root. Will be resolved to an
        absolute path. Need not be a git repo for the skeleton — that
        constraint enters when hooks / git-blob caching land in a later phase.
    name :
        Logical repo name used as the `repo` column in the graph. Defaults
        to the directory's basename if not given.
    backend :
        Concrete `GraphBackend`. Defaults to a `SqliteBackend` storing the
        graph at `<root>/.otter-docs/graph.db`.

    Phase 1 contract
    ----------------
    Only the constructor and `.graph` property are functional. `.scan()`,
    `.findings()`, and `.render()` raise `NotImplementedError` with phase
    pointers so callers know what's coming.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        name: str | None = None,
        backend: GraphBackend | None = None,
    ):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Repo root does not exist: {self.root}")
        self.name = name or self.root.name

        if backend is None:
            data_dir = self.root / ".otter-docs"
            data_dir.mkdir(exist_ok=True)
            backend = SqliteBackend(data_dir / "graph.db")
        self._backend = backend
        self._backend.connect()

    # ── functional in phase 1 ────────────────────────────────────────

    @property
    def graph(self) -> GraphBackend:
        """The underlying graph backend.

        Use this for ad-hoc queries: `repo.graph.list_functions()`,
        `repo.graph.callers_of(...)`, `repo.graph.find_similar(...)`.
        """
        return self._backend

    def close(self) -> None:
        """Close the backend connection."""
        self._backend.close()

    def __enter__(self) -> Repo:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── stubs (later phases) ─────────────────────────────────────────

    def scan(self, *, reset: bool = False) -> None:
        """Walk the repo, parse ASTs, populate the graph + embeddings.

        Lands in phase 2 (tree-sitter + stack-graphs) and phase 4
        (three-vector indexing).
        """
        raise NotImplementedError(
            "Repo.scan() lands in phases 2–4. The graph backend is live now; "
            "you can manually add nodes via repo.graph.add_function(...) etc."
        )

    def findings(self, **_filters: object) -> list[object]:
        """Run detectors against the indexed graph; return list[Finding].

        Detectors land starting in phase 5 (static tier) and phase 6
        (embedding-augmented tier).
        """
        raise NotImplementedError("Repo.findings() lands in phases 5–7.")

    def render(self, _section: str) -> str:
        """Generate a markdown view for the named renderer.

        Renderers land in phase 9.
        """
        raise NotImplementedError("Repo.render() lands in phase 9.")
