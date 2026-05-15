"""redundancy.semantic_equivalence — pairs of functions that look the same.

For every function with a description_vec, query the backend for its
nearest neighbors and emit Findings for pairs above the threshold.
Description-vector similarity is the primary signal (HyDE-style: the
LLM's description abstracts away surface-level differences). When
code_vec exists for both pair members, we also require it to clear a
slightly lower bar — that catches false positives where two functions
have similar descriptions but obviously different code.

This is the marquee detector of Phase 6: it's what the library does
that grep + radon can't.

Cost-tier: embedding. Free at query time if the index is warm.

Output deduplication: a pair (A, B) is emitted once; we keep the
canonical-pick metadata (longer function wins as "canonical") so the
recommendation can suggest a direction.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location, VectorKind

DEFAULT_DESCRIPTION_THRESHOLD = 0.92
DEFAULT_CODE_THRESHOLD = 0.85
# How many neighbors to ask the index for, per function. Larger k finds
# more pairs at the cost of latency; 8 is a good default for "find my
# clones" without being noisy.
DEFAULT_K = 8


class SemanticEquivalenceDetector:
    kind = "redundancy.semantic_equivalence"
    cost_tier = "embedding"

    def __init__(
        self,
        *,
        description_threshold: float = DEFAULT_DESCRIPTION_THRESHOLD,
        code_threshold: float = DEFAULT_CODE_THRESHOLD,
        k: int = DEFAULT_K,
    ) -> None:
        self.description_threshold = description_threshold
        self.code_threshold = code_threshold
        self.k = k

    def run(self, repo: str, graph: GraphBackend) -> list[Finding]:
        findings: list[Finding] = []
        seen_pairs: set[tuple[str, str]] = set()
        # Index by guid for O(1) lookups during pair building.
        by_guid = {fn.guid: fn for fn in graph.list_functions(repo)}

        for fn in by_guid.values():
            if fn.description_vec is None:
                continue
            hits = graph.find_similar(
                repo, fn.description_vec,
                vector_kind=VectorKind.DESCRIPTION,
                node_kind="function",
                k=self.k,
                min_similarity=self.description_threshold,
            )
            for hit in hits:
                if hit.node_id == fn.guid:
                    continue
                other = by_guid.get(hit.node_id)
                if other is None:
                    continue
                # Order the pair canonically so we only emit once.
                pair = tuple(sorted([fn.guid, hit.node_id]))
                if pair in seen_pairs:
                    continue

                # Secondary gate on code similarity if both sides have a code_vec.
                code_similarity: float | None = None
                if fn.code_vec is not None and other.code_vec is not None:
                    code_similarity = _cosine(fn.code_vec, other.code_vec)
                    if code_similarity < self.code_threshold:
                        continue

                seen_pairs.add(pair)
                canonical, redundant = _pick_canonical(fn, other)
                findings.append(Finding(
                    kind=self.kind,
                    confidence=hit.similarity,
                    locations=[
                        Location(
                            repo=repo, path=fn.module_path,
                            line=fn.line, end_line=fn.end_line, guid=fn.guid,
                        ),
                        Location(
                            repo=repo, path=other.module_path,
                            line=other.line, end_line=other.end_line,
                            guid=other.guid,
                        ),
                    ],
                    evidence={
                        "description_similarity": hit.similarity,
                        "code_similarity": code_similarity,
                        "function_names": [fn.name, other.name],
                        "canonical_guid": canonical.guid,
                    },
                    recommendation=Recommendation(
                        summary=(
                            f"Consolidate `{redundant.name}` into `{canonical.name}`"
                        ),
                        rationale=(
                            f"`{fn.name}` and `{other.name}` describe the same "
                            f"behavior (description similarity "
                            f"{hit.similarity:.2f}"
                            + (
                                f", code similarity {code_similarity:.2f}"
                                if code_similarity is not None else ""
                            )
                            + "). The longer/older implementation makes a better "
                            "canonical home. Confirm by reading both before "
                            "consolidating."
                        ),
                        blast_radius=[fn.guid, other.guid],
                        suggested_tests=[
                            f"git grep -n '\\b{redundant.name}\\b'",
                            "run callers of both functions after consolidation",
                        ],
                    ),
                    source_detector=self.kind,
                ))
        return findings


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity over already-unit-length vectors → dot product."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


def _pick_canonical(a, b):
    """Prefer the longer function; tiebreak by lower line number (older)."""
    a_len = a.end_line - a.line
    b_len = b.end_line - b.line
    if a_len > b_len:
        return a, b
    if b_len > a_len:
        return b, a
    # Same length — prefer the one with the lower starting line as a
    # proxy for "earlier in the file / closer to canonical".
    if a.line <= b.line:
        return a, b
    return b, a


register(SemanticEquivalenceDetector())
