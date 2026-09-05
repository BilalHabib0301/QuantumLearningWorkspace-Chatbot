"""Structured request-logging tests: a real /ask and /ask/stream call
must append one JSONL entry with the documented fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

# Point the JSONL log at a temp file before the app handlers call it.
import request_logger  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    mock_engine = MagicMock()
    mock_engine.chunks_indexed = 7
    mock_engine.embedding_model_name = "all-MiniLM-L6-v2"
    mock_engine.default_top_k = 4
    mock_engine.max_distance = 1.2
    mock_engine.auto_reconnect = False
    log_file = tmp_path / "requests.jsonl"
    request_logger.set_log_path(log_file)

    with patch("main.create_engine", return_value=mock_engine):
        import main
        from auth import get_current_user_email

        main._engine = mock_engine
        main._engine_ready = True
        main.app.dependency_overrides[get_current_user_email] = lambda: "alice@example.com"
        with TestClient(main.app) as test_client:
            yield test_client, log_file
        main.app.dependency_overrides.clear()
        main._engine = None
        main._engine_ready = False


def _read_log(log_file: Path) -> list[dict]:
    return [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]


@patch("main.prepare_ask")
@patch("main.finalize_ask")
@patch("main.generate_answer_sync")
def test_ask_writes_structured_log_entry(mock_gen, mock_finalize, mock_prepare, client):
    from rag_service import AskResult, PreparedAsk, SourceInfo

    test_client, log_file = client
    question = "Where does photosynthesis happen?"
    prepared = PreparedAsk(
        question=question,
        history=[],
        top_k=4,
        rewritten_question=question,
        hop_queries=[question],
        retrieved_text="<<<UNTRUSTED_DOCUMENT>>>...",
        accumulated={
            "documents": ["x"],
            "ids": ["pdf_chunk_1"],
            "distances": [0.5],
            "metadatas": [{}],
        },
        refused=False,
        include_sources=True,
    )
    mock_prepare.return_value = prepared
    mock_gen.return_value = "In the chloroplasts."
    mock_finalize.return_value = AskResult(
        answer="In the chloroplasts.",
        refused=False,
        top_k=4,
        sources=[SourceInfo(id="pdf_chunk_1", distance=0.5, preview="chloroplasts")],
        source_ids=["pdf_chunk_1"],
        rewritten_question=question,
        grounded=True,
        retrieval_rounds=1,
        hop_queries=[question],
    )

    resp = test_client.post(
        "/ask",
        json={"question": question, "skip_cache": True},
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200

    entries = _read_log(log_file)
    assert len(entries) == 1
    entry = entries[0]
    assert "timestamp" in entry
    assert entry["endpoint"] == "/ask"
    assert entry["question_length"] == len(question)
    assert entry["retrieval_ms"] is not None
    assert entry["llm_ms"] is not None
    assert entry["grounded"] is True
    assert entry["cached"] is False
    # user_id is SHA-256 hashed, never the raw email
    assert entry["user_id"] is not None
    assert entry["user_id"] != "alice@example.com"
    assert len(entry["user_id"]) == 16
    assert "alice" not in json.dumps(entry)


@patch("main.prepare_ask")
@patch("main.finalize_ask")
@patch("main.stream_answer_tokens")
def test_ask_stream_writes_structured_log_entry(mock_stream, mock_finalize, mock_prepare, client):
    from rag_service import AskResult, PreparedAsk

    test_client, log_file = client
    question = "What is ATP?"
    prepared = PreparedAsk(
        question=question,
        history=[],
        top_k=4,
        rewritten_question=question,
        hop_queries=[question],
        retrieved_text="energy",
        accumulated={
            "documents": ["x"],
            "ids": ["yt_chunk_0"],
            "distances": [0.4],
            "metadatas": [{}],
        },
        refused=False,
        include_sources=True,
        client=MagicMock(),
    )
    mock_prepare.return_value = prepared
    mock_stream.return_value = iter(["ATP ", "is energy."])
    mock_finalize.return_value = AskResult(
        answer="ATP is energy.",
        refused=False,
        top_k=4,
        source_ids=["yt_chunk_0"],
        grounded=True,
        retrieval_rounds=1,
        hop_queries=[question],
    )

    with test_client.stream(
        "POST",
        "/ask/stream",
        json={"question": question, "skip_cache": True},
        headers={"X-User-Id": "bob"},
    ) as resp:
        lines = [json.loads(line) for line in resp.iter_lines() if line]
    assert lines[-1]["type"] == "done"

    entries = _read_log(log_file)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["endpoint"] == "/ask/stream"
    assert entry["question_length"] == len(question)
    assert entry["retrieval_ms"] is not None
    assert entry["llm_ms"] is not None
    assert entry["grounded"] is True
    assert entry["user_id"] not in {"bob", "alice@example.com"}
    assert "alice" not in json.dumps(entry)


def test_log_entry_fields_match_schema(client):
    """Log shape is stable: only the documented keys, same across endpoints."""
    from rag_service import AskResult, PreparedAsk
    import main as main_mod

    test_client, log_file = client
    question = "Schema?  "
    prepared = PreparedAsk(
        question=question.strip(),
        history=[],
        top_k=4,
        rewritten_question=question.strip(),
        hop_queries=[question.strip()],
        retrieved_text="",
        accumulated={"documents": [], "ids": [], "distances": [], "metadatas": []},
        refused=True,
        refusal_answer="unavailable",
        include_sources=True,
    )
    with patch.object(main_mod, "prepare_ask", return_value=prepared):
        test_client.post(
            "/ask",
            json={"question": question, "skip_cache": True},
            headers={"X-User-Id": "carol"},
        )

    entries = _read_log(log_file)
    assert len(entries) == 1
    assert set(entries[0].keys()) == {
        "timestamp",
        "endpoint",
        "user_id",
        "question_length",
        "retrieval_ms",
        "llm_ms",
        "grounded",
        "cached",
    }
    assert entries[0]["question_length"] == len(question.strip())
    assert entries[0]["grounded"] is None