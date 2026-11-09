"""Embedder tests. The real model only runs where the `embed` group is installed (not CI)."""

from __future__ import annotations

import math

import pytest

from docs_qa import embed


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embedder_is_deterministic_and_normalized():
    e = embed.HashEmbedder()
    a, b, a_again = e.embed(["autovacuum", "work_mem", "autovacuum"])
    assert len(a) == embed.EMBEDDING_DIM
    assert a == a_again and a != b
    assert math.isclose(dot(a, a), 1.0, rel_tol=1e-9)
    assert e.embedded == ["autovacuum", "work_mem", "autovacuum"]


@pytest.fixture(scope="module")
def local():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        return embed.LocalEmbedder()
    except Exception as e:  # first run downloads the model; skip when offline
        pytest.skip(f"model unavailable: {e}")


def test_local_embedder_vectors(local):
    texts = [
        "autovacuum_naptime: minimum delay between autovacuum runs on a database",
        "How often does autovacuum run?",
        "The string concatenation operator joins two text values",
    ]
    setting, question, unrelated = local.embed(texts)
    assert local.dim == embed.EMBEDDING_DIM and len(setting) == embed.EMBEDDING_DIM
    assert math.isclose(dot(setting, setting), 1.0, rel_tol=1e-4)
    assert dot(question, setting) > dot(question, unrelated)


def test_local_embedder_keeps_input_order(local):
    texts = ["short", "a much longer piece of text about write-ahead logging " * 5, "mid size text"]
    together = local.embed(texts)
    one_by_one = [local.embed([t])[0] for t in texts]
    for a, b in zip(together, one_by_one, strict=True):
        assert dot(a, b) > 0.999  # same vectors whatever the batching or padding
