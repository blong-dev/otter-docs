"""Sanity checks on the pydantic data models."""

from __future__ import annotations

from datetime import datetime

import pytest

from otter_docs.models import (
    ClassRecord,
    Edge,
    FunctionRecord,
    Language,
    Location,
    ModuleRecord,
    SimilarityHit,
    VectorKind,
)


def test_location_is_frozen():
    loc = Location(repo="r", path="p")
    with pytest.raises(Exception):
        loc.path = "other"  # type: ignore[misc]


def test_module_defaults():
    m = ModuleRecord(repo="v3", path="pkg/a.py", language=Language.PYTHON)
    assert m.docstring == ""
    assert m.imports == []
    assert m.tags == []
    assert m.description_vec is None
    assert m.code_vec is None
    assert m.docstring_vec is None


def test_function_serializes_round_trip():
    f = FunctionRecord(
        repo="v3", guid="g-1", name="hello", module_path="pkg/a.py",
        line=10, end_line=20, is_async=True, tags=["io"],
        updated_at=datetime(2026, 5, 14, 12, 0, 0),
    )
    data = f.model_dump()
    f2 = FunctionRecord.model_validate(data)
    assert f2 == f


def test_class_record_no_args_field():
    """Classes don't carry args/returns/is_async (those are function-only)."""
    c = ClassRecord(repo="v3", guid="g", name="Foo", module_path="a.py",
                    line=1, end_line=10)
    assert not hasattr(c, "args")
    assert not hasattr(c, "returns")
    assert not hasattr(c, "is_async")


def test_edge_minimal():
    e = Edge(kind="CALLS", src_id="a", dst_id="b")
    assert e.kind == "CALLS"


def test_similarity_hit_shape():
    h = SimilarityHit(node_kind="function", node_id="g-1", similarity=0.92)
    assert h.similarity == 0.92


def test_vector_kind_values():
    assert {v.value for v in VectorKind} == {"description", "code", "docstring"}


def test_language_unknown_default_for_unrecognized():
    """Language enum doesn't auto-fallback; unknown strings raise."""
    with pytest.raises(ValueError):
        Language("rust")  # not in our v0.1 set
