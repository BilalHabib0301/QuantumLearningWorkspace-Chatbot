"""Unit tests for the pure-Python BM25 lexical retriever."""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from bm25 import BM25Index, STOPWORDS, tokenize  # noqa: E402


DOCS = [
    ("pdf_chunk_0", "The Calvin cycle takes place in the stroma of the chloroplast."),
    ("yt_chunk_0", "ATP synthase spins using the proton gradient to make ATP."),
    ("conflict_chunk_0", "The Calvin cycle requires darkness to run."),
    ("pdf_chunk_1", "Thylakoid membranes hold the photosystems of the light reactions."),
    ("yt_chunk_1", "About 1000 ATP synthase complexes occupy each square micrometer."),
]


def test_tokenize_drops_stopwords():
    tokens = tokenize("What does the Calvin cycle require?")
    assert "what" not in tokens
    assert "the" not in tokens
    assert "does" not in tokens
    assert "calvin" in tokens
    assert "cycle" in tokens
    assert "require" in tokens


def test_tokenize_lowercases_and_keeps_numbers():
    tokens = tokenize("ATP density 1000 per SQUARE micrometer")
    assert "atp" in tokens
    assert "1000" in tokens
    assert "square" in tokens


def test_stopwords_are_a_frozenset():
    assert isinstance(STOPWORDS, frozenset)
    assert "the" in STOPWORDS


def test_bm25_ranks_matching_docs_first():
    index = BM25Index.build(DOCS)
    ranked = index.top_ids("Calvin cycle stroma darkness", k=3)
    # Only documents sharing a query term get a score; both Calvin-cycle docs match.
    assert set(ranked) == {"conflict_chunk_0", "pdf_chunk_0"}
    assert ranked[0].startswith("conflict_chunk_0") or ranked[0].startswith("pdf_chunk_0")


def test_bm25_no_overlap_returns_empty():
    index = BM25Index.build(DOCS)
    assert index.top_ids("FIFA world cup football", k=5) == []


def test_bm25_respects_k():
    index = BM25Index.build(DOCS)
    assert len(index.top_ids("ATP synthase proton gradient", k=1)) == 1
    # Large k still only returns documents that share at least one query term.
    assert len(index.top_ids("ATP synthase proton gradient", k=100)) == 2


def test_bm25_scored_only_shared_terms():
    index = BM25Index.build(DOCS)
    scores = index.score("ATP synthase proton gradient")
    assert scores
    for doc_id in scores:
        assert scores[doc_id] > 0


def test_bm25_empty_corpus_safe():
    index = BM25Index.build([])
    assert index.size == 0
    assert index.score("anything at all") == {}
    assert index.top_ids("anything at all", k=3) == []