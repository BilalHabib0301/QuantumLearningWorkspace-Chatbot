"""Unit tests for hybrid (RRF) retrieval and its relevance gate."""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from hybrid_search import (  # noqa: E402
    get_bm25_index,
    hybrid_retrieve,
    is_hybrid_relevant,
    rrf_fuse,
    rrf_scores,
)


class _Vec:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeEmbedder:
    def encode(self, _text):
        return _Vec([0.1, 0.2])


class FakeCollection:
    rows = [
        ("doc1", "The calvin cycle happens inside the stroma.", {"source": "a.txt"}),
        ("doc_lexical", "calvin cycle thylakoid grana chloroplast photosystems", {"source": "b.txt"}),
        ("doc3", "ATP synthase proton gradient makes ATP", {"source": "c.txt"}),
        ("doc4", "marine phytoplankton produce half the oxygen", {"source": "yt.txt"}),
        ("doc_user", "photosystems grana photosystems grana", {"source": "u.txt", "user_id": "alice"}),
    ]

    name = "fake_collection"

    def __init__(self, rows=None):
        self._rows = list(rows if rows is not None else self.rows)

    def get(self, where=None, include=None, limit=None):
        rows = self._rows
        if where is not None:
            rows = [r for r in rows if r[2].get("user_id") == where.get("user_id")]
        if limit is not None:
            rows = rows[:limit]
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [r[2] for r in rows],
        }

    def query(self, query_embeddings=None, n_results=None, where=None):
        return {
            "ids": [["doc1", "doc3"]],
            "documents": [["The calvin cycle happens inside the stroma.", "ATP synthase proton gradient makes ATP"]],
            "distances": [[0.4, 0.6]],
            "metadatas": [[{"source": "a.txt"}, {"source": "c.txt"}]],
        }


class FakeEngine:
    def __init__(self, collection):
        self.collection = collection
        self.embedding_model = FakeEmbedder()


def _engine():
    return FakeEngine(FakeCollection())


def test_rrf_scores_and_fuse_combine_ranks():
    scores = rrf_scores(["a", "b", "c"], ["c", "d", "a"])
    assert set(scores) == {"a", "b", "c", "d"}
    assert scores["a"] > scores["b"]
    assert scores["c"] > scores["b"]
    fused = rrf_fuse(["a", "b", "c"], ["c", "d", "a"])
    assert fused[0] in ("a", "c")
    assert fused[1] in ("a", "c")
    assert set(fused) == {"a", "b", "c", "d"}


def test_hybrid_retrieve_returns_same_shape():
    results = hybrid_retrieve(_engine(), "calvin cycle thylakoid", n_results=3)
    assert set(results) == {"documents", "distances", "metadatas", "ids", "rrf_scores"}
    assert len(results["ids"]) == 3
    assert len(results["documents"]) == 3
    assert len(results["metadatas"]) == 3


def test_hybrid_includes_lexical_only_hit_with_none_distance():
    results = hybrid_retrieve(_engine(), "calvin cycle thylakoid", n_results=3)
    ids = results["ids"]
    assert "doc_lexical" in ids
    by_id = dict(zip(ids, results["distances"]))
    assert by_id["doc_lexical"] is None
    assert by_id["doc1"] == 0.4


def test_hybrid_result_text_available_for_lexical_only():
    results = hybrid_retrieve(_engine(), "calvin cycle thylakoid", n_results=3)
    idx = results["ids"].index("doc_lexical")
    assert "grana" in results["documents"][idx]


def test_get_bm25_index_cached_until_corpus_changes():
    collection = FakeCollection()
    engine = FakeEngine(collection)
    first = get_bm25_index(engine)
    second = get_bm25_index(engine)
    assert first is second
    collection._rows.append(("doc_new", "a brand new topic about history", {"source": "n.txt"}))
    rebuilt = get_bm25_index(engine)
    assert rebuilt is not first
    assert rebuilt.size == 6


def test_bm25_index_scoped_by_user():
    collection = FakeCollection()
    engine = _engine()
    index = get_bm25_index(engine, user_id="alice")
    assert index.bm25.top_ids("photosystems grana", k=5)  # doc in scope
    results = hybrid_retrieve(engine, "photosystems grana", n_results=5, user_id="alice")
    assert results["ids"]  # corpus filtered to alice = intersection


def test_is_hybrid_relevant_semantic_close():
    results = hybrid_retrieve(_engine(), "calvin cycle thylakoid", n_results=3)
    assert is_hybrid_relevant(results, max_distance=0.5) is True


def test_is_hybrid_relevant_lexical_only_counts():
    results = hybrid_retrieve(_engine(), "calvin cycle thylakoid", n_results=3)
    results["distances"] = [None, None, None]
    assert is_hybrid_relevant(results, max_distance=0.1) is True


def test_is_hybrid_relevant_empty_refuses():
    assert is_hybrid_relevant({"ids": [], "distances": []}, max_distance=1.2) is False