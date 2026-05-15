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
from otter_docs.clients.base import EmbeddingClient, LLMClient
from otter_docs.describe import DescriptionCache, SqliteDescriptionCache
from otter_docs.discovery import is_tsx, iter_source_files
from otter_docs.detectors import run_all as _run_detectors
from otter_docs.detectors.base import CostTier
from otter_docs.enrich import EnrichReport, Enricher
from otter_docs.findings import Finding, Recommendation
from otter_docs.llm_direct import (
    Review,
    propose_consolidation as _propose_consolidation,
    review_change as _review_change,
)
from otter_docs.models import Edge, Language
from otter_docs.parsers import parse_file
from otter_docs.parsers.typescript import TSX_PARSER
from otter_docs.resolvers import resolve_repo as _resolve_repo
from otter_docs.resolvers.base import ResolveReport


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
        # Lazily-initialized description cache, scoped to this Repo
        # instance. When the backend is SqliteBackend we reuse its
        # connection so descriptions live in the same file as the
        # graph; for other backends we fall back to an in-memory dict.
        self._description_cache: DescriptionCache | None = None

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

    def resolve(
        self,
        *,
        languages: set[Language] | None = None,
    ) -> dict[Language, ResolveReport]:
        """Run cross-file resolvers and write CALLS edges.

        scan() emits intra-file CALLS only — a function calling a
        helper from another file looks dead. This step asks each
        language's mature name-resolver (jedi for Python; gopls / tsserver
        in later versions) to fill in those cross-file edges.

        Idempotent — backend edge inserts are upserts, so a second
        resolve() over an unchanged repo doesn't change the graph.

        Parameters
        ----------
        languages :
            Optional filter. Default runs every registered resolver.
        """
        return _resolve_repo(
            repo=self.name,
            repo_root=self.root,
            graph=self._backend,
            languages=languages,
        )

    def enrich(
        self,
        llm: LLMClient,
        embedder: EmbeddingClient,
        *,
        description_cache: DescriptionCache | None = None,
        max_embed_chars: int | None = None,
    ) -> EnrichReport:
        """Three-vector enrichment over every symbol the graph knows about.

        For each module/function/class:
          - Generate an LLM description (cached by source content hash).
          - Embed three texts: description, code slice, docstring.
          - Upsert the record back into the backend with the three vectors.

        Idempotent — re-running over unchanged code re-uses cached
        descriptions and reproduces the same vectors. Call this after
        `scan()` or pass a `embed=...`-shaped helper later when we
        consolidate the API surface.

        Parameters
        ----------
        llm :
            LLMClient implementation. Use FakeLLMClient for tests,
            OllamaLLMClient for local-model runs, or any custom client
            that follows the Protocol.
        embedder :
            EmbeddingClient. Its `.dim` must match this repo's backend
            vector_dim — otherwise the backend will raise.
        description_cache :
            Optional explicit description cache. If omitted, the
            describer uses an ephemeral in-memory cache. Pass a
            `SqliteDescriptionCache` bound to a long-lived connection
            for persistent caching across runs.
        """
        cache = description_cache or self._default_description_cache()
        kwargs: dict = {"description_cache": cache}
        if max_embed_chars is not None:
            kwargs["max_embed_chars"] = max_embed_chars
        enricher = Enricher(self._backend, llm, embedder, **kwargs)
        return enricher.enrich_repo(self.name, self.root)

    def _default_description_cache(self) -> DescriptionCache:
        """Return (and memoize) this repo's default description cache.

        When the backend is SqliteBackend, descriptions piggy-back on
        the same connection so they persist alongside the graph. For
        other backends (Neo4j, custom), we fall back to a per-Repo
        in-memory dict — callers who want persistence there should
        pass an explicit `description_cache=`.
        """
        if self._description_cache is not None:
            return self._description_cache
        if isinstance(self._backend, SqliteBackend):
            self._description_cache = SqliteDescriptionCache(self._backend.conn)
        else:
            from otter_docs.describe import _DictCache
            self._description_cache = _DictCache()
        return self._description_cache

    def findings(
        self,
        *,
        kinds: set[str] | None = None,
        cost_tiers: set[CostTier] | None = None,
    ) -> list[Finding]:
        """Run registered detectors against this repo's graph.

        Parameters
        ----------
        kinds :
            Optional set of Finding kinds to keep (e.g. {"dead_code"}).
            Other detectors won't be run. None means run everything.
        cost_tiers :
            Optional set of cost tiers (`"static"`, `"embedding"`,
            `"llm_direct"`). Use this to cap runtime cost when you
            don't have a budget to call the LLM-direct tier on every
            scan. None means all tiers.

        Returns
        -------
        list[Finding] — each carrying source_detector for provenance and
        an optional Recommendation for what to do about it.
        """
        return _run_detectors(
            self.name, self._backend, kinds=kinds, cost_tiers=cost_tiers
        )

    # ── LLM-direct tier (Phase 7) ────────────────────────────────────

    def propose_consolidation(
        self,
        finding: Finding,
        llm: LLMClient,
    ) -> Recommendation:
        """Ask the LLM to produce a unified diff consolidating a redundancy.

        Input must be a `redundancy.*` Finding with at least two
        function locations (the embedding-tier detector emits these).
        Returns a Recommendation whose `proposed_diff` is the
        generated diff, or None if the LLM declined.

        The library does NOT apply the diff — the harness owns
        implementation.
        """
        return _propose_consolidation(
            finding=finding,
            repo=self.name,
            repo_root=self.root,
            graph=self._backend,
            llm=llm,
        )

    def review_change(
        self,
        diff: str,
        llm: LLMClient,
        *,
        related_findings: list[Finding] | None = None,
    ) -> Review:
        """Ask the LLM to review a unified diff.

        Returns a `Review` carrying summary, overall verdict
        (approve / request_changes / comment), addresses_findings,
        new_risks, blockers. Used by the agent after writing a patch
        — call before applying.
        """
        return _review_change(
            diff=diff, related_findings=related_findings, llm=llm,
        )

    def describe(
        self,
        llm: LLMClient,
        *,
        guid: str | None = None,
        path: str | None = None,
    ):
        """Describe a single symbol on demand.

        Looks up the symbol by `guid` (function/class) or `path`
        (module), reads its source from disk, and runs the describer
        (with caching). Cheaper than running enrich() when you only
        want one symbol's prose.

        Returns the `Description` object or None if the symbol isn't
        found.
        """
        from otter_docs.describe import Describer
        from otter_docs.enrich import _slice_source
        if guid is None and path is None:
            raise ValueError("describe() needs guid= or path=")
        if guid is not None and path is not None:
            raise ValueError("describe() takes guid= OR path=, not both")

        describer = Describer(llm, self._default_description_cache())
        if path is not None:
            module = self._backend.get_module(self.name, path)
            if module is None:
                return None
            try:
                source = (self.root / path).read_bytes()
            except OSError:
                return None
            return describer.describe(
                kind="module", guid=path,
                language=module.language.value, source=source,
            )

        # guid path: look up function first, then class.
        fn = self._backend.get_function(self.name, guid)
        cls = None if fn else self._backend.get_class(self.name, guid)
        symbol = fn or cls
        if symbol is None:
            return None
        try:
            source = (self.root / symbol.module_path).read_bytes()
        except OSError:
            return None
        body = _slice_source(source, symbol.line, symbol.end_line)
        module = self._backend.get_module(self.name, symbol.module_path)
        language = module.language.value if module is not None else "text"
        kind = "function" if fn is not None else "class"
        return describer.describe(
            kind=kind, guid=guid, language=language, source=body,
        )

    def render(self, section: str) -> str:
        """Render one named section to a markdown fragment.

        Sections: system_overview, findings_summary, redundancy_report,
        dependency_graph, architecture_smells. Raises KeyError on an
        unknown name so a typo fails loudly.
        """
        from otter_docs.render import render_section
        return render_section(section, self)

    def render_document(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        sections: list[str] | None = None,
    ) -> str:
        """Write/update a generated document at `path`, return its text.

        If the file doesn't exist it's bootstrapped with marker pairs
        for every requested section (default: all registered
        renderers, in a stable order). On rerun, only the content
        between each section's BEGIN/END markers is replaced — human
        prose elsewhere is preserved byte-for-byte.
        """
        from otter_docs.render import (
            bootstrap_document,
            inject,
            registry,
        )

        target = Path(path)
        if sections is None:
            # Stable, readable order — overview first, smells last.
            order = [
                "system_overview",
                "findings_summary",
                "redundancy_report",
                "dependency_graph",
                "architecture_smells",
            ]
            known = set(registry())
            sections = [s for s in order if s in known]

        if target.exists():
            document = target.read_text(encoding="utf-8")
        else:
            document = bootstrap_document(
                title=title or self.name, sections=sections
            )

        for name in sections:
            document = inject(document, name=name, body=self.render(name))

        target.write_text(document, encoding="utf-8")
        return document
