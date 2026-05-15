"""Three-vector enrichment over a scanned Repo.

After `Repo.scan()` has populated the graph with AST-only records, the
enrichment pass:

1. Pulls each symbol's source slice from disk (using line/end_line).
2. Generates a structured description via the LLM (cached by content).
3. Embeds three texts per symbol:
   - description text   → `description_vec`
   - code slice         → `code_vec`
   - docstring (if any) → `docstring_vec`
4. Upserts the symbol back into the backend with the three vectors.

The enrichment pass is idempotent — re-running over an unchanged repo
hits the description cache for every node and produces identical
vectors (FakeEmbeddingClient is deterministic; real embedders should
be too at temp 0). The backend's add_module/function/class methods
upsert, so the second pass just replaces the existing rows in place.

For v0.1 we embed the raw code slice (no AST normalization). The
spec calls for normalized code in code_vec so semantically-equivalent
code clusters together; we'll layer that in once we have a static
detector pipeline (Phase 5) that needs it. Until then a real embedder
(nomic-embed-text) is robust enough to whitespace differences that the
unnormalized version is useful immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from otter_docs.backends.base import GraphBackend
from otter_docs.clients.base import EmbeddingClient, LLMClient
from otter_docs.describe import Describer, DescriptionCache
from otter_docs.models import ClassRecord, FunctionRecord, Language, ModuleRecord


# Cap on how much of a module file we feed to the describer for module-
# level descriptions. Past ~200 lines the description gets noisy and
# the model latency dominates. Functions/classes are scoped to their
# own line ranges already.
MAX_MODULE_LINES_FOR_DESCRIBE = 200

# Cap on how many characters we feed to the embedder per call.
#
# nomic-embed-text v1.5 maxes at 8192 tokens; code averages ~3 chars/
# token, so 20000 chars ≈ 6700 tokens stays comfortably under the
# model's hard limit with room for the model's own prefix overhead.
#
# Truncation matters because some embedders (and especially llama.cpp
# servers with default `--ubatch-size 512`) hard-fail beyond their
# batch size — operators running undersized batches should pass a
# smaller `max_embed_chars` to Repo.enrich() rather than expecting
# the library to second-guess their server config.
MAX_EMBED_CHARS = 20000


@dataclass
class EnrichReport:
    """Counts + errors from a single `enrich()` pass.

    Returned to the caller so they can decide whether to retry. Kept
    distinct from ScanReport because enrichment is its own phase and
    can be re-run without rescanning.
    """

    modules_enriched: int = 0
    functions_enriched: int = 0
    classes_enriched: int = 0
    cache_hits: int = 0
    embedding_calls: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def _slice_source(source: bytes, line: int, end_line: int) -> bytes:
    """Return the bytes between `line` (1-based inclusive) and `end_line`.

    Robust against off-by-one and trailing newlines: missing or out-of-
    range bounds clamp instead of raising. The describer doesn't need
    perfect bounds — it only needs enough text to characterize the symbol.
    """
    lines = source.splitlines(keepends=True)
    start = max(0, line - 1)
    end = min(len(lines), end_line)
    return b"".join(lines[start:end])


def _module_source_for_describe(source: bytes) -> bytes:
    """Trim a module's source to a budget the LLM can comfortably handle."""
    lines = source.splitlines(keepends=True)
    if len(lines) <= MAX_MODULE_LINES_FOR_DESCRIBE:
        return source
    head = lines[: MAX_MODULE_LINES_FOR_DESCRIBE // 2]
    tail = lines[-(MAX_MODULE_LINES_FOR_DESCRIBE // 2):]
    return b"".join(head) + b"\n# ... (truncated) ...\n" + b"".join(tail)


def _truncate_for_embed(text: str, *, max_chars: int = MAX_EMBED_CHARS) -> str:
    """Cap text at `max_chars`, keeping head + tail for context.

    Embedders have hard token limits and return errors (HTTP 500 in
    practice on nomic-embed-text) when exceeded. Head+tail preserves
    the most diagnostic regions of a file: imports + early definitions
    at the top, top-level invocations + main block at the bottom.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... (truncated for embedding) ...\n" + text[-half:]


def _language_tag(lang: Language) -> str:
    """Stringified Language for the prompt's fenced code block."""
    return lang.value if isinstance(lang, Language) else str(lang)


def _norm_dim(vec: list[float], dim: int) -> list[float]:
    """Validate that a returned embedding matches the expected dim."""
    if len(vec) != dim:
        raise ValueError(f"embedder returned dim {len(vec)}, expected {dim}")
    return vec


class Enricher:
    """Runs the three-vector pipeline over a scanned graph.

    Stateless across calls except for the optional description cache
    held by `describer`. Construct one per repo; call `.enrich_repo`
    to enrich every symbol the backend knows about.
    """

    def __init__(
        self,
        backend: GraphBackend,
        llm: LLMClient,
        embedder: EmbeddingClient,
        *,
        description_cache: DescriptionCache | None = None,
        max_embed_chars: int = MAX_EMBED_CHARS,
    ) -> None:
        self.backend = backend
        self.llm = llm
        self.embedder = embedder
        self.describer = Describer(llm, description_cache)
        self.max_embed_chars = max_embed_chars

    # ── high-level entry point ─────────────────────────────────────

    def enrich_repo(self, repo: str, repo_root: Path) -> EnrichReport:
        """Enrich every node (module / function / class) in `repo`.

        Reads source from `repo_root` on disk — the backend stores
        paths relative to the repo root, so we join them here.
        """
        report = EnrichReport()
        # Cache source per file to avoid re-reading once per symbol.
        source_cache: dict[str, bytes] = {}

        # Modules: enrich each, plus build a path → source map for the
        # function/class pass to reuse.
        for module in self.backend.list_modules(repo):
            try:
                source = self._read_source(repo_root, module.path, source_cache)
            except OSError as e:
                report.errors.append((module.path, f"read failed: {e}"))
                continue
            try:
                self._enrich_module(module, source, report)
            except Exception as e:  # noqa: BLE001
                report.errors.append(
                    (module.path, f"module enrich failed: {type(e).__name__}: {e}")
                )

        # Functions
        for fn in self.backend.list_functions(repo):
            try:
                source = self._read_source(repo_root, fn.module_path, source_cache)
            except OSError as e:
                report.errors.append((fn.module_path, f"read failed: {e}"))
                continue
            try:
                self._enrich_function(fn, source, report)
            except Exception as e:  # noqa: BLE001
                report.errors.append(
                    (fn.guid, f"function enrich failed: {type(e).__name__}: {e}")
                )

        # Classes
        for cls in self.backend.list_classes(repo):
            try:
                source = self._read_source(repo_root, cls.module_path, source_cache)
            except OSError as e:
                report.errors.append((cls.module_path, f"read failed: {e}"))
                continue
            try:
                self._enrich_class(cls, source, report)
            except Exception as e:  # noqa: BLE001
                report.errors.append(
                    (cls.guid, f"class enrich failed: {type(e).__name__}: {e}")
                )

        return report

    # ── per-kind enrichment ────────────────────────────────────────

    def _enrich_module(
        self, module: ModuleRecord, source: bytes, report: EnrichReport
    ) -> None:
        body = _module_source_for_describe(source)
        desc = self._describe(
            kind="module", guid=module.path, language=module.language, source=body,
            report=report,
        )
        vectors = self._embed_three(
            description_text=desc.text,
            code_text=source.decode("utf-8", errors="replace"),
            docstring_text=module.docstring,
            report=report,
        )
        updated = module.model_copy(update={
            "description_vec": vectors[0],
            "code_vec": vectors[1],
            "docstring_vec": vectors[2],
        })
        self.backend.add_module(updated)
        report.modules_enriched += 1

    def _enrich_function(
        self, fn: FunctionRecord, source: bytes, report: EnrichReport
    ) -> None:
        body = _slice_source(source, fn.line, fn.end_line)
        # Functions need a language tag; the backend doesn't carry it,
        # so we infer from the module the function belongs to.
        module = self.backend.get_module(fn.repo, fn.module_path)
        language = module.language if module is not None else Language.UNKNOWN
        desc = self._describe(
            kind="function", guid=fn.guid, language=language, source=body, report=report
        )
        vectors = self._embed_three(
            description_text=desc.text,
            code_text=body.decode("utf-8", errors="replace"),
            docstring_text=fn.docstring,
            report=report,
        )
        updated = fn.model_copy(update={
            "description_vec": vectors[0],
            "code_vec": vectors[1],
            "docstring_vec": vectors[2],
        })
        self.backend.add_function(updated)
        report.functions_enriched += 1

    def _enrich_class(
        self, cls: ClassRecord, source: bytes, report: EnrichReport
    ) -> None:
        body = _slice_source(source, cls.line, cls.end_line)
        module = self.backend.get_module(cls.repo, cls.module_path)
        language = module.language if module is not None else Language.UNKNOWN
        desc = self._describe(
            kind="class", guid=cls.guid, language=language, source=body, report=report
        )
        vectors = self._embed_three(
            description_text=desc.text,
            code_text=body.decode("utf-8", errors="replace"),
            docstring_text=cls.docstring,
            report=report,
        )
        updated = cls.model_copy(update={
            "description_vec": vectors[0],
            "code_vec": vectors[1],
            "docstring_vec": vectors[2],
        })
        self.backend.add_class(updated)
        report.classes_enriched += 1

    # ── shared helpers ─────────────────────────────────────────────

    def _describe(
        self, *, kind: str, guid: str, language: Language, source: bytes,
        report: EnrichReport,
    ):
        # We can't ask the cache directly without an extra method, so
        # we trust Describer to cache correctly and just count its
        # underlying LLM calls before vs after.
        before = len(getattr(self.llm, "calls", []))
        desc = self.describer.describe(
            kind=kind, guid=guid, language=_language_tag(language), source=source
        )
        after = len(getattr(self.llm, "calls", []))
        if after == before:
            report.cache_hits += 1
        return desc

    def _embed_three(
        self, *, description_text: str, code_text: str, docstring_text: str,
        report: EnrichReport,
    ) -> tuple[list[float], list[float], list[float] | None]:
        dim = self.embedder.dim
        # All three texts are truncated to the embedder's safe budget.
        # Code (especially whole-module source) is the one that
        # actually trips the limit in practice; description + docstring
        # are usually well under, but bounding all three keeps the
        # implementation uniform and the failure mode predictable.
        cap = self.max_embed_chars
        texts = [
            _truncate_for_embed(description_text, max_chars=cap),
            _truncate_for_embed(code_text, max_chars=cap),
        ]
        has_doc = bool(docstring_text)
        if has_doc:
            texts.append(_truncate_for_embed(docstring_text, max_chars=cap))
        vectors = self.embedder.embed(texts)
        report.embedding_calls += 1
        desc_vec = _norm_dim(vectors[0], dim)
        code_vec = _norm_dim(vectors[1], dim)
        doc_vec = _norm_dim(vectors[2], dim) if has_doc else None
        return desc_vec, code_vec, doc_vec

    @staticmethod
    def _read_source(
        root: Path, rel_path: str, cache: dict[str, bytes]
    ) -> bytes:
        cached = cache.get(rel_path)
        if cached is not None:
            return cached
        data = (root / rel_path).read_bytes()
        cache[rel_path] = data
        return data
