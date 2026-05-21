"""redundancy.semantic_equivalence — pairs of functions that look the same.

For every function with a description_vec, query the backend for its
nearest neighbors and emit Findings for pairs above the threshold.
Description-vector similarity is the primary signal (HyDE-style: the
LLM's description abstracts away surface-level differences). When
code_vec exists for both pair members, we also require it to clear a
secondary bar — that catches false positives where two functions
have similar descriptions but obviously different code.

Cost-tier: embedding. Free at query time if the index is warm.

Output deduplication: a pair (A, B) is emitted once; we keep the
canonical-pick metadata (longer function wins as "canonical") so the
recommendation can suggest a direction.

Shape classification (evidence.shape, 2026-05-21 calibration). Same
`kind`, different downstream treatment:

  - `likely_duplicate` — real consolidation candidate. Confidence =
    description similarity (current behavior).
  - `sibling_methods` — same bare name on different classes (e.g. two
    `process` methods). Code-similarity gate raised to 0.93;
    confidence scaled down by 0.7. The recommendation reframes the
    pair as siblings: review for a shared base / protocol, not
    consolidation.
  - `lifecycle_hook` — same name is a Python lifecycle hook
    (`__init__`, `setUp`, `__repr__`, …). Categorically informational
    — same hook on different classes is structurally expected. Emit
    at confidence=0.05 so it shows up on the bus + in graph queries
    but never cards under the default threshold.

The shape choices keep one `kind` (so subscriber-side `external_id`
dedup stays stable across the recalibration); downstream readers can
branch on `evidence.shape` if they want shape-specific handling.
"""

from __future__ import annotations

from otter_docs.backends.base import GraphBackend
from otter_docs.detectors.base import register
from otter_docs.findings import Finding, Recommendation
from otter_docs.models import Location, VectorKind

DEFAULT_DESCRIPTION_THRESHOLD = 0.95
DEFAULT_CODE_THRESHOLD = 0.85
# Raised gate for pairs that already share a basename — same name +
# very similar body has to be very similar to be more than coincidence.
SIBLING_CODE_THRESHOLD = 0.93
# Confidence discount for sibling_methods: similar in shape but
# different contracts; not a deletion candidate.
SIBLING_CONFIDENCE_SCALE = 0.7
# Informational confidence for lifecycle hooks — kept above 0 so the
# Finding is observable, kept well below subscriber thresholds so it
# never cards by default.
LIFECYCLE_CONFIDENCE = 0.05
# How many neighbors to ask the index for, per function. Larger k finds
# more pairs at the cost of latency; 8 is a good default for "find my
# clones" without being noisy.
DEFAULT_K = 8

# Path segments + filename patterns that identify test files. Pairs
# where either side is a test file are skipped — tests are duplicative
# by design (parameterized cases, shared fixtures), and consolidating
# them is rarely what the user wants. (2026-05-21 calibration: 524 of
# v3's 1354 likely_duplicate findings involved tests.)
_TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__", "spec", "__test__"})
_TEST_FILE_SUFFIXES = (
    "_test.py", "_test.go",
    ".test.ts", ".test.tsx", ".test.js", ".test.jsx",
    ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
)


def _is_test_path(path: str) -> bool:
    """True if `path` looks like a test file or lives under a tests dir."""
    parts = path.replace("\\", "/").split("/")
    if any(p in _TEST_DIR_NAMES for p in parts):
        return True
    base = parts[-1] if parts else path
    if base.startswith("test_"):
        return True
    return any(base.endswith(suf) for suf in _TEST_FILE_SUFFIXES)


# Python lifecycle hooks + common test-framework hooks. Same name on
# different classes is structurally expected (the runtime / unittest
# calls each class's own hook); flagging them as redundant is wrong.
LIFECYCLE_HOOKS = frozenset({
    "__init__", "__post_init__", "__new__", "__del__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
    "__repr__", "__str__", "__bool__", "__hash__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__len__", "__iter__", "__next__", "__contains__",
    "__getitem__", "__setitem__", "__delitem__",
    "__getattr__", "__setattr__", "__delattr__",
    "__getattribute__", "__missing__",
    "__call__", "__add__", "__sub__", "__mul__", "__truediv__",
    "__floordiv__", "__mod__", "__pow__",
    "__copy__", "__deepcopy__", "__reduce__",
    "setUp", "tearDown", "setUpClass", "tearDownClass",
})


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
        # Pre-compute function_guid → class_guid via CONTAINS edges.
        # Backend-protocol-safe (one edges_from call per class), and
        # makes the per-pair shape classification O(1).
        class_of: dict[str, str] = {}
        for cls in graph.list_classes(repo):
            for e in graph.edges_from(repo, cls.guid, kind="CONTAINS"):
                class_of[e.dst_id] = cls.guid

        findings: list[Finding] = []
        seen_pairs: set[tuple[str, str]] = set()
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
                pair = tuple(sorted([fn.guid, hit.node_id]))
                if pair in seen_pairs:
                    continue
                # Test files: duplicative by design (parameterized cases,
                # shared fixtures). Skip pairs where either side is a test.
                if _is_test_path(fn.module_path) or _is_test_path(other.module_path):
                    continue

                shape = _classify_shape(fn, other, class_of)

                # Code-similarity secondary gate. Bar raised for
                # sibling_methods (same basename → higher prior on FP);
                # lifecycle_hook bypasses the gate because it's
                # informational regardless of code similarity.
                code_threshold = (
                    SIBLING_CODE_THRESHOLD
                    if shape == "sibling_methods"
                    else self.code_threshold
                )
                code_similarity: float | None = None
                if fn.code_vec is not None and other.code_vec is not None:
                    code_similarity = _cosine(fn.code_vec, other.code_vec)
                    if shape != "lifecycle_hook" and code_similarity < code_threshold:
                        continue

                seen_pairs.add(pair)
                canonical, redundant = _pick_canonical(fn, other)
                confidence, summary, rationale, tests = _render_for_shape(
                    shape, fn, other, canonical, redundant,
                    hit.similarity, code_similarity,
                )

                findings.append(Finding(
                    kind=self.kind,
                    confidence=confidence,
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
                        "shape": shape,
                    },
                    recommendation=Recommendation(
                        summary=summary,
                        rationale=rationale,
                        blast_radius=[fn.guid, other.guid],
                        suggested_tests=tests,
                    ),
                    source_detector=self.kind,
                ))
        return findings


def _classify_shape(fn, other, class_of: dict[str, str]) -> str:
    """Decide which of the three semantic shapes this pair fits.

    Same bare name + lifecycle hook → categorically informational.
    Same bare name + different classes (CONTAINS evidence) → siblings.
    Anything else → real consolidation candidate.
    """
    if fn.name == other.name:
        if fn.name in LIFECYCLE_HOOKS:
            return "lifecycle_hook"
        cls_a = class_of.get(fn.guid)
        cls_b = class_of.get(other.guid)
        if cls_a is not None and cls_b is not None and cls_a != cls_b:
            return "sibling_methods"
    return "likely_duplicate"


def _render_for_shape(
    shape: str, fn, other, canonical, redundant,
    desc_sim: float, code_sim: float | None,
) -> tuple[float, str, str, list[str]]:
    sim_str = f"{desc_sim:.2f}" + (
        f", code {code_sim:.2f}" if code_sim is not None else ""
    )
    if shape == "lifecycle_hook":
        return (
            LIFECYCLE_CONFIDENCE,
            f"`{fn.name}` defined on multiple classes (lifecycle hook)",
            (
                f"Both names are `{fn.name}` — a Python lifecycle hook "
                f"(description sim {sim_str}). The runtime / framework "
                "calls each class's own hook; same name on different "
                "classes is structurally expected. Informational only, "
                "not a consolidation candidate."
            ),
            [
                "no action — this is an informational pairing",
            ],
        )
    if shape == "sibling_methods":
        return (
            desc_sim * SIBLING_CONFIDENCE_SCALE,
            f"Review `{fn.name}` siblings across classes",
            (
                f"`{fn.name}` is defined as a method on two distinct "
                f"classes with similar shape ({sim_str}). Different "
                "classes carry different contracts even when bodies look "
                "alike; treat this as a sibling pair, not a duplicate. "
                "Consider whether a shared base class or protocol expresses "
                "the commonality better than consolidation."
            ),
            [
                "grep -rn 'class .*:' to confirm the owning classes",
                "check whether callers depend on the class-specific behavior",
            ],
        )
    # likely_duplicate (original detector behavior)
    return (
        desc_sim,
        f"Consolidate `{redundant.name}` into `{canonical.name}`",
        (
            f"`{fn.name}` and `{other.name}` describe the same behavior "
            f"({sim_str}). The longer/older implementation makes a better "
            "canonical home. Confirm by reading both before consolidating."
        ),
        [
            f"git grep -n '\\b{redundant.name}\\b'",
            "run callers of both functions after consolidation",
        ],
    )


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
