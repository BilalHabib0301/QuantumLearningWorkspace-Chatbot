"""Tests for the clarifying-question flow (Phase 11 Task 1).

Covers the three required behaviours:
  1. a genuinely vague question triggers clarification
  2. a short-but-clear question does NOT
  3. a valid follow-up resolvable from history does NOT
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from clarify import needs_clarification  # noqa: E402
from rag_service import (  # noqa: E402
    CLARIFICATION_MESSAGE,
    RagEngine,
    ask,
    prepare_ask,
)

RICH_CONTEXT = [
    {"role": "user", "content": "What is photosynthesis?"},
    {
        "role": "assistant",
        "content": (
            "Photosynthesis has two stages. The light-dependent reactions split water "
            "and release oxygen; the Calvin cycle fixes carbon dioxide into sugar in "
            "the stroma."
        ),
    },
]

VAGUE_CONTEXT = [
    {"role": "user", "content": "hmm"},
    {"role": "assistant", "content": "ok"},
]


def _engine() -> RagEngine:
    return RagEngine(
        collection=MagicMock(),
        embedding_model=MagicMock(),
        chunks_indexed=10,
    )


def test_vague_question_triggers_clarification():
    assert needs_clarification("tell me more", None) is True
    assert needs_clarification("what about that?", None) is True
    assert needs_clarification("what about it?", []) is True


def test_short_but_clear_question_no_clarification():
    assert needs_clarification("What is photosynthesis?", None) is False
    assert needs_clarification("what is ATP?", None) is False
    assert needs_clarification("Where does the Calvin cycle take place?", None) is False


def test_question_with_concrete_subject_no_clarification():
    # Concrete content words present => no clarification even if referential phrasing
    assert needs_clarification("What about the second stage?", None) is False
    assert needs_clarification("What about the Calvin cycle?", None) is False
    # ...but a pure pronoun reference with no history stays ambiguous
    assert needs_clarification("what about it?", None) is True


def test_followup_resolvable_from_history_no_clarification():
    assert needs_clarification("what about that?", RICH_CONTEXT) is False
    assert needs_clarification("tell me more", RICH_CONTEXT) is False


def test_followup_with_vague_history_still_clarifies():
    assert needs_clarification("what about that?", VAGUE_CONTEXT) is True


def test_prepare_ask_returns_clarification_for_vague_question():
    prepared = prepare_ask(_engine(), "tell me more", history=None)
    assert prepared.clarification_required is True
    assert prepared.clarification_message == CLARIFICATION_MESSAGE
    assert not prepared.refused


def test_ask_returns_clarifying_message_without_llm():
    result = ask(_engine(), "what about that?", history=None)
    assert result.needed_clarification is True
    assert result.refused is False
    assert result.answer == CLARIFICATION_MESSAGE
    assert result.source_ids == []


def test_prepare_ask_clear_question_not_clarification(monkeypatch):
    engine = _engine()

    def fake_retrieve(engine, client, query, k, do_rerank, user_id=None, retrieval_method=None):
        return {
            "documents": ["ATP is the energy currency of cells."],
            "distances": [0.5],
            "ids": ["pdf_chunk_0"],
            "metadatas": [{"source": "a.txt"}],
        }, None

    monkeypatch.setattr("rag_service._retrieve_round", fake_retrieve)
    prepared = prepare_ask(engine, "what is ATP?", history=None, multi_hop=False)
    assert prepared.clarification_required is False
    assert prepared.refused is False
    assert prepared.accumulated["ids"] == ["pdf_chunk_0"]