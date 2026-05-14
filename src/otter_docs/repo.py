"""User-facing entry point.

The shape `from otter_docs import Repo; repo = Repo("/path/to/code")` is the
contract every consumer (humans, agents, MCP servers) uses. Phase 1 ships
the skeleton with the backend wired up; scanning and findings come in
later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from otter_docs.backends import GraphBackend, SqliteBackend
from otter_docs.discovery import is_tsx, iter_source_files
from otter_docs.models import Edge, Language
from otter_docs.parsers import parse_file
from otter_docs.parsers.typescript import TSX_PARSER


@dataclass
class ScanReport:
    """Summary of what `Repo.scan()` did. Returned to the caller.

    Useful for tests, logging, and "did anything actually get indexed?"
    sanity checks. Counts are post-deduplication via the backend's
    upsert semantics — so a re-scan of an unchanged repo reports the
    same numbers as the first scan, not zero.
    """

    files_seen: int = 0
    files_parsed: int = 0
    files_skipped: list[Path] = field(default_factory=list)
    modules: int = 0
    functions: int = 0
    classes: int = 0
    edges: int = 0
    errors: list[tuple[Path, str]] = field(default_factory=list)


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

    def scan(self, *, reset: bool = False) -> ScanReport:
        """Walk the repo, parse every supported source file, populate the graph.

        AST-only for now — no LLM, no embeddings. Phase 3/4 fill in
        descriptions and three-vector indexing on top of the nodes this
        method creates.

        Parameters
        ----------
        reset :
            If True, wipe this repo's existing rows before scanning so
            renames and deletes don't leave orphans behind. Default False
            keeps the data and relies on upsert semantics for changes.

        Returns
        -------
        ScanReport summarizing what was indexed.
        """
        report = ScanReport()
        if reset:
            self._backend.reset(repo=self.name)

        for abs_path, language in iter_source_files(self.root):
            report.files_seen += 1
            rel = abs_path.relative_to(self.root).as_posix()
            try:
                source = abs_path.read_bytes()
            except OSError as e:
                report.errors.append((abs_path, f"read failed: {e}"))
                continue

            try:
                if language is Language.TYPESCRIPT and is_tsx(abs_path):
                    # .tsx requires the TSX grammar; everything else
                    # routes through the standard parser registry.
                    result = TSX_PARSER.parse(
                        repo=self.name, path=rel, source=source
                    )
                else:
                    result = parse_file(
                        repo=self.name, path=rel, source=source, language=language
                    )
                    if result is None:
                        report.files_skipped.append(abs_path)
                        continue
            except Exception as e:  # noqa: BLE001 — parser bugs shouldn't fail the scan
                report.errors.append((abs_path, f"parse failed: {type(e).__name__}: {e}"))
                continue

            self._backend.add_module(result.module)
            for fn in result.functions:
                self._backend.add_function(fn)
            for cls in result.classes:
                self._backend.add_class(cls)
            for edge in result.edges:
                self._add_edge(edge)

            report.files_parsed += 1
            report.modules += 1
            report.functions += len(result.functions)
            report.classes += len(result.classes)
            report.edges += len(result.edges)

        return report

    def _add_edge(self, edge: Edge) -> None:
        """Write an Edge to the backend, scoped to this repo's name.

        The GraphBackend Protocol exposes `_add_edge_with_repo` because
        Edge itself doesn't carry a repo column; we keep the helper
        private since callers should reach for `repo.graph` for raw
        edge writes during exploration.
        """
        self._backend._add_edge_with_repo(edge, repo=self.name)

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
