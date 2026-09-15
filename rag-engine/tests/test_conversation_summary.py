"""Unit + contract tests for conversation summarization and history compaction.

Phase 11 Part C:
- summarize_conversation(): dedicated summarization prompt, no retrieval.
- condense_history() / recent_history(): summary-block compaction instead of
  hard-dropping turns past HISTORY_TURN_CAP.
- POST /conversations/summarize contract + summary caching.
- Token-savings regression: compacted context < full raw history for the
  same conversation while preserving early-turn facts.

The live LLM/server behavior (baseline degradation + compaction recovery) is
covered separately in tests/test_conversation_quality_integration.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from cache import SummaryCache  # noqa: E402
from rag_service import (  # noqa: E402
    EARLIER_SUMMARY_PREFIX,
    HISTORY_TURN_CAP,
    condense_history,
    recent_history,
    summarize_conversation,
)


def _turns(n: int, prefix: str = "turn") -> list[dict]:
    """n user/assistant turn pairs."""
    out: list[dict] = []
    for i in range(n):
        out.append({"role": "user", "content": f"{prefix} {i} question"})
        out.append({"role": "assistant", "content": f"{prefix} {i} answer"})
    return out


def _stub_client(summary_text: str = "Early turns summarized here.") -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = summary_text
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# summarize_conversation()
# ---------------------------------------------------------------------------

def test_summarize_conversation_uses_dedicated_prompt_no_retrieval():
    client = _stub_client("  A dense study summary.  ")
    history = [
        {"role": "user", "content": "What is ATP?"},
        {"role": "assistant", "content": "ATP is the energy currency."},
    ]
    summary = summarize_conversation(client, history)
    assert summary == "A dense study summary."

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "study-summarizer" in messages[0]["content"]
    assert "What is ATP?" in messages[1]["content"]
    assert "ATP is the energy currency." in messages[1]["content"]
    # Dedicated prompt must not look like the RAG answer prompt / retrieval path.
    assert "UNTRUSTED_DOCUMENT" not in str(messages)
    assert "Reference documents" not in messages[0]["content"]


def test_summarize_conversation_empty_history_short_circuits():
    client = MagicMock()
    assert summarize_conversation(client, []) == ""
    client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# condense_history()
# ---------------------------------------------------------------------------

def test_condense_history_returns_none_when_within_cap():
    assert condense_history(MagicMock(), _turns(4)) is None
    assert condense_history(MagicMock(), _turns(3)) is None


def test_condense_history_summarizes_older_keeps_recent_verbatim():
    history = _turns(6)  # 12 messages; cap keeps last 4 turns (8 messages)
    client = _stub_client("Older turns: topic A.")
    result = condense_history(client, history, max_turns=HISTORY_TURN_CAP)

    assert result is not None
    assert len(result) == 1 + HISTORY_TURN_CAP * 2
    assert result[0]["content"].startswith(EARLIER_SUMMARY_PREFIX)
    assert result[1:] == history[-(HISTORY_TURN_CAP * 2):]
    # Only the older turns were sent to the summarizer.
    summarized = client.chat.completions.create.call_args.kwargs["messages"][1]
    assert "turn 0 question" in summarized["content"]
    assert "turn 5 question" not in summarized["content"]


def test_condense_history_falls_back_to_none_when_summary_empty():
    client = _stub_client("   ")
    result = condense_history(client, _turns(6))
    assert result is None


# ---------------------------------------------------------------------------
# recent_history()
# ---------------------------------------------------------------------------

def test_recent_history_plain_cap_without_client_drops_old_turns():
    history = _turns(6)
    got = recent_history(history)
    assert got == history[-(HISTORY_TURN_CAP * 2):]  # hard drop (baseline behavior)
    assert got != history


def test_recent_history_with_client_condenses_overdue_turns():
    history = _turns(12)
    client = _stub_client("exam code XKCD-42, topic hemoglobin.")
    got = recent_history(history, client=client)
    assert got[0]["content"].startswith(EARLIER_SUMMARY_PREFIX)
    assert "XKCD-42" in got[0]["content"]  # early-turn fact preserved in the block
    assert got[1:] == history[-(HISTORY_TURN_CAP * 2):]
    assert len(got) == 1 + HISTORY_TURN_CAP * 2


def test_recent_history_toggle_disables_compaction(monkeypatch):
    """ENABLE_HISTORY_COMPACTION=false must restore the raw hard-drop baseline."""
    monkeypatch.setenv("ENABLE_HISTORY_COMPACTION", "false")
    history = _turns(12)
    got = recent_history(history, client=_stub_client("should never be used"))
    assert got == history[-(HISTORY_TURN_CAP * 2):]
    assert not any(
        (m.get("content") or "").startswith(EARLIER_SUMMARY_PREFIX) for m in got
    )


def test_recent_history_within_cap_never_calls_llm():
    history = _turns(4)
    client = MagicMock()
    got = recent_history(history, client=client)
    assert got == history
    client.chat.completions.create.assert_not_called()


def test_recent_history_preserves_existing_summary_block():
    history = [
        {"role": "user", "content": f"{EARLIER_SUMMARY_PREFIX}: old stuff"},
    ] + _turns(6)
    got = recent_history(history)  # no client — but summary block must survive
    assert got[0]["content"].startswith(EARLIER_SUMMARY_PREFIX)
    assert got[1:] == history[-(HISTORY_TURN_CAP * 2):]


def test_recent_history_falls_back_to_cap_when_summary_fails():
    history = _turns(6)
    client = _stub_client("")
    got = recent_history(history, client=client)
    assert got == history[-(HISTORY_TURN_CAP * 2):]


# ---------------------------------------------------------------------------
# Token savings regression (unit-level mechanics)
# ---------------------------------------------------------------------------

def _est_tokens(messages: list[dict]) -> int:
    """Rough token estimate (~4 chars/token) for ASCII-heavy study text."""
    text = " ".join((m.get("content") or "") for m in messages)
    return max(1, len(text) // 4)


def test_compaction_uses_fewer_tokens_than_full_history():
    """Compaction must cost less than shipping the whole raw history every turn."""
    full = _turns(12)
    client = _stub_client("Early study session: hemoglobin, oxygen transport.")
    compacted = recent_history(full, client=client)
    assert compacted is not None

    full_cost = _est_tokens(full)
    compacted_cost = _est_tokens(compacted)
    # The compacted context must be smaller than the full 12-turn history...
    assert compacted_cost < full_cost, (
        f"compacted {compacted_cost} tokens should beat full history {full_cost}"
    )
    # ...but strictly larger than the raw-drop baseline (which loses the facts).
    baseline = _turns(12)[-(HISTORY_TURN_CAP * 2):]
    assert compacted_cost > _est_tokens(baseline)
    # ...and the early-turn fact must survive only in the compacted form.
    assert "hemoglobin" in compacted[0]["content"]
    assert "hemoglobin" not in " ".join(b["content"] for b in baseline)


# ---------------------------------------------------------------------------
# POST /conversations/summarize contract
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    mock_engine = MagicMock()
    mock_engine.chunks_indexed = 7
    mock_engine.embedding_model_name = "all-MiniLM-L6-v2"
    mock_engine.default_top_k = 4
    mock_engine.max_distance = 1.2
    mock_engine.auto_reconnect = False

    with patch("main.create_engine", return_value=mock_engine):
        import main
        from auth import get_current_user_email
        from rate_limiter import check_rate_limit

        main._engine = mock_engine
        main._engine_ready = True
        main.app.dependency_overrides[get_current_user_email] = (
            lambda: "test-user@example.com"
        )
        # Isolate contract tests from the shared per-user rate limiter; real
        # rate-limit behavior is covered by test_rate_limiting.py.
        main.app.dependency_overrides[check_rate_limit] = lambda: None
        with patch("main.summary_cache", new=SummaryCache()):
            from fastapi.testclient import TestClient

            with TestClient(main.app) as test_client:
                yield test_client
        main.app.dependency_overrides.clear()
        main._engine = None
        main._engine_ready = False


BODY = {
    "history": [
        {"role": "user", "content": "What is ATP?"},
        {"role": "assistant", "content": "ATP is the energy currency."},
        {"role": "user", "content": "And where is it made?"},
        {"role": "assistant", "content": "In mitochondria."},
    ]
}


@patch("main.summarize_conversation", return_value="Concise summary of the chat.")
def test_summarize_endpoint_returns_summary_and_caches(mock_summarize, client):
    resp = client.post("/conversations/summarize", json=BODY)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["summary"] == "Concise summary of the chat."
    assert data["turn_count"] == 2
    assert data["cached"] is False

    # Repeat call must come from cache, not a second LLM hit.
    resp2 = client.post("/conversations/summarize", json=BODY)
    assert resp2.status_code == 200
    assert resp2.json()["summary"] == "Concise summary of the chat."
    assert resp2.json()["cached"] is True
    assert mock_summarize.call_count == 1


def test_summarize_endpoint_empty_history_422(client):
    resp = client.post("/conversations/summarize", json={"history": []})
    assert resp.status_code == 422


def test_summarize_endpoint_empty_summary_502(client):
    with patch("main.summarize_conversation", return_value="   "):
        resp = client.post("/conversations/summarize", json=BODY)
    assert resp.status_code == 502