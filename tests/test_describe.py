"""Describer + cache tests."""

from __future__ import annotations

import sqlite3

import pytest

from otter_docs.clients import FakeLLMClient
from otter_docs.describe import (
    Describer,
    Description,
    SqliteDescriptionCache,
    _DictCache,
    content_hash,
)


def test_content_hash_is_sha1():
    assert content_hash(b"def foo(): pass") == content_hash(b"def foo(): pass")
    assert content_hash(b"a") != content_hash(b"b")


def test_describe_calls_llm_on_miss_and_caches():
    llm = FakeLLMClient()
    d = Describer(llm)
    src = b"def adder(a, b): return a + b"
    out = d.describe(kind="function", guid="g1", language="python", source=src)
    assert isinstance(out, Description)
    assert out.text.startswith("FAKE:")
    assert out.content_hash == content_hash(src)
    assert len(llm.calls) == 1  # one LLM call so far

    # Second describe of identical source — must hit cache, no LLM call.
    again = d.describe(kind="function", guid="g1", language="python", source=src)
    assert again.text == out.text
    assert len(llm.calls) == 1  # still one


def test_describe_re_invokes_on_source_change():
    llm = FakeLLMClient()
    d = Describer(llm)
    d.describe(kind="function", guid="g", language="python", source=b"def f(): pass")
    d.describe(kind="function", guid="g", language="python", source=b"def f(): return 1")
    assert len(llm.calls) == 2


def test_describe_cache_keyed_only_by_content():
    """Same source, different guid → cache hit (description is about the code)."""
    llm = FakeLLMClient()
    d = Describer(llm)
    src = b"def helper(): return None"
    d.describe(kind="function", guid="guid-a", language="python", source=src)
    d.describe(kind="function", guid="guid-b", language="python", source=src)
    assert len(llm.calls) == 1


def test_sqlite_cache_persists_across_describer_instances():
    conn = sqlite3.connect(":memory:")
    cache = SqliteDescriptionCache(conn)
    llm = FakeLLMClient()
    Describer(llm, cache).describe(
        kind="function", guid="g", language="python", source=b"def x(): pass"
    )
    assert len(llm.calls) == 1
    # Brand-new describer, same connection / cache — must read existing entry.
    fresh_llm = FakeLLMClient()
    Describer(fresh_llm, cache).describe(
        kind="function", guid="g", language="python", source=b"def x(): pass"
    )
    assert fresh_llm.calls == []


def test_sqlite_cache_get_put_directly():
    conn = sqlite3.connect(":memory:")
    cache = SqliteDescriptionCache(conn)
    assert cache.get("h1") is None
    cache.put("h1", kind="function", guid="g", text="hi")
    assert cache.get("h1") == "hi"


def test_describer_prompt_includes_kind_language_source():
    llm = FakeLLMClient()
    d = Describer(llm)
    d.describe(kind="function", guid="g", language="go", source=b"func plain() {}")
    prompt = llm.calls[0]
    assert "function" in prompt
    assert "go" in prompt
    assert "plain" in prompt


def test_describe_many_returns_one_per_input():
    llm = FakeLLMClient()
    d = Describer(llm)
    items = [
        ("function", "g1", "python", b"def a(): pass"),
        ("function", "g2", "python", b"def b(): pass"),
        ("class", "c1", "python", b"class C: pass"),
    ]
    out = d.describe_many(items)
    assert len(out) == 3
    assert [o.guid for o in out] == ["g1", "g2", "c1"]


def test_dict_cache_basic():
    c = _DictCache()
    assert c.get("k") is None
    c.put("k", kind="function", guid="g", text="t")
    assert c.get("k") == "t"
