"""Shared pytest fixtures + helpers."""

from __future__ import annotations

import math

import pytest

from otter_docs.backends import SqliteBackend


def unit(values: list[float], dim: int) -> list[float]:
    """Pad to `dim` with zeros, then normalize to unit length.

    Used everywhere we need test vectors with predictable similarity
    relationships. Both backends expect unit-length vectors for
    consistent similarity scores; this helper produces them.
    """
    padded = values + [0.0] * (dim - len(values))
    n = math.sqrt(sum(x * x for x in padded))
    if n == 0:
        return padded
    return [x / n for x in padded]


@pytest.fixture
def vector_dim() -> int:
    """Small dim used in unit tests for speed.

    Real otter-docs runs default to 768 (nomic-embed-text). Tests use
    a tiny dim so vector ops are trivially fast and easy to reason about.
    """
    return 8


@pytest.fixture
def backend(vector_dim: int):
    """Fresh in-memory SqliteBackend for each test."""
    with SqliteBackend(":memory:", vector_dim=vector_dim) as be:
        yield be


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default; opt in with --run-integration."""
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="needs --run-integration (and external services)"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that need external services (Neo4j, etc.)",
    )
