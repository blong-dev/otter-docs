"""Loaders for clone-pair datasets.

Two sources:

  load_jsonl(path)         our bundled fixture format — one JSON object
                           per line with the ClonePair fields. Used by
                           CI (the small hand-labeled set) and for
                           any custom labeled set you create.

  load_gptclonebench(dir)  the real GPTCloneBench layout. NOT shipped
                           with the package (37K pairs, large). This
                           loader lets a local run point at a checkout
                           of the dataset so the full eval is one call.
                           Documented procedure, not a CI step — CI
                           has neither the dataset nor a real embedder.

GPTCloneBench reference: arxiv.org/html/2505.04311v1 (the audit that
established BigCloneBench is corrupted and GPTCloneBench is the
credible replacement).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from otter_docs.eval import ClonePair

# Where the bundled fixture lives relative to the installed package.
# tests/ isn't packaged, so we resolve from the repo root when present
# and fall back to the package dir for an installed wheel.
_FIXTURE_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "clone_pairs.jsonl",
    Path(__file__).resolve().parent / "data" / "clone_pairs.jsonl",
]


def bundled_fixture_path() -> Path:
    """Return the path to the bundled labeled set, or raise if absent."""
    for c in _FIXTURE_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "bundled clone_pairs.jsonl not found; expected at "
        + " or ".join(str(c) for c in _FIXTURE_CANDIDATES)
    )


def load_jsonl(path: str | Path) -> list[ClonePair]:
    """Load ClonePairs from a JSONL file (our fixture format)."""
    out: list[ClonePair] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(ClonePair(
                code_a=obj["code_a"],
                code_b=obj["code_b"],
                is_clone=bool(obj["is_clone"]),
                clone_type=obj.get("clone_type", ""),
                description_a=obj.get("description_a", ""),
                description_b=obj.get("description_b", ""),
            ))
    return out


def load_bundled() -> list[ClonePair]:
    """Convenience: load the package's bundled labeled set."""
    return load_jsonl(bundled_fixture_path())


def load_gptclonebench(root: str | Path) -> Iterator[ClonePair]:
    """Stream ClonePairs from a GPTCloneBench checkout.

    GPTCloneBench publishes clone pairs as paired source files plus a
    manifest of (file_a, file_b, type, is_clone). Layout varies by
    mirror; we support the common CSV/JSONL manifest at
    `<root>/manifest.{jsonl,csv}` pointing at files under `<root>`.

    Yields lazily so the 37K-pair set doesn't all sit in memory.
    Raises FileNotFoundError if no recognizable manifest is present —
    we don't guess at an unknown layout.
    """
    root = Path(root)
    jsonl = root / "manifest.jsonl"
    csv = root / "manifest.csv"
    if jsonl.exists():
        yield from _gcb_from_jsonl(root, jsonl)
    elif csv.exists():
        yield from _gcb_from_csv(root, csv)
    else:
        raise FileNotFoundError(
            f"No manifest.jsonl or manifest.csv under {root}. "
            "Point load_gptclonebench at a dataset checkout with a "
            "manifest, or convert it to our JSONL fixture format and "
            "use load_jsonl()."
        )


def _read(root: Path, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8", errors="replace")


def _gcb_from_jsonl(root: Path, manifest: Path) -> Iterator[ClonePair]:
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            yield ClonePair(
                code_a=_read(root, m["file_a"]),
                code_b=_read(root, m["file_b"]),
                is_clone=bool(m.get("is_clone", True)),
                clone_type=str(m.get("type", "")),
            )


def _gcb_from_csv(root: Path, manifest: Path) -> Iterator[ClonePair]:
    import csv as _csv

    with open(manifest, encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            yield ClonePair(
                code_a=_read(root, row["file_a"]),
                code_b=_read(root, row["file_b"]),
                is_clone=str(row.get("is_clone", "1")).strip() in ("1", "true", "True"),
                clone_type=row.get("type", ""),
            )
