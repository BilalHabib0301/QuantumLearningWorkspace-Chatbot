"""Tests for the /feedback endpoint and FeedbackStore."""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Note: Tests assume rag-engine/ is in PYTHONPATH (configured in pyproject.toml)

@pytest.fixture()
def client():
    # Mock RAG engine
    mock_engine = MagicMock()
    mock_engine.chunks_indexed = 1
    mock_engine.embedding_model_name = "test-model"
    mock_engine.default_top_k = 4
    mock_engine.auto_reconnect = False
    
    # Mock JWT auth dependency
    from auth import get_current_user_email
    
    with patch("main.create_engine", return_value=mock_engine):
        import main
        from feedback_store import feedback_store
        
        main._engine = mock_engine
        main._engine_ready = True
        
        # Clear feedback store before/after tests
        feedback_store.clear()
        
        main.app.dependency_overrides[get_current_user_email] = lambda: "test-user@example.com"
        with TestClient(main.app) as test_client:
            yield test_client
        
        main.app.dependency_overrides.clear()
        feedback_store.clear()
        main._engine = None
        main._engine_ready = False

def test_feedback_flow_success(client):
    """Test submitting feedback for a valid response_id."""
    # 1. First, get a valid response_id from /ask
    with patch("main.prepare_ask") as mock_prepare, \
         patch("main.generate_answer_sync") as mock_gen, \
         patch("main.finalize_ask") as mock_finalize:
        
        from rag_service import AskResult, PreparedAsk, SourceInfo
        
        mock_prepare.return_value = PreparedAsk(
            question="What is ATP?",
            history=[], top_k=4, rewritten_question="What is ATP?",
            hop_queries=["What is ATP?"], retrieved_text="energy",
            accumulated={"documents": ["energy"], "ids": ["c1"], "distances": [0.1], "metadatas": [{"source": "test.txt"}]},
        )
        mock_gen.return_value = "ATP is energy."
        mock_finalize.return_value = AskResult(
            answer="ATP is energy.", refused=False, top_k=4,
            sources=[SourceInfo(id="c1", distance=0.1, preview="energy", source="test.txt")],
            source_ids=["c1"], rewritten_question="What is ATP?", grounded=True
        )

        resp = client.post("/ask", json={"question": "What is ATP?", "skip_cache": True})
        assert resp.status_code == 200
        data = resp.json()
        response_id = data["response_id"]
        assert response_id != ""

    # 2. Now submit feedback
    feedback_resp = client.post("/feedback", json={
        "response_id": response_id,
        "rating": "up"
    })
    
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["received"] is True
    
    # 3. Verify it was stored correctly
    from feedback_store import feedback_store
    record = feedback_store.get(response_id)
    assert record is not None
    assert record.rating == "up"
    assert record.question == "What is ATP?"

def test_feedback_nonexistent_response_id(client):
    """Submitting feedback for unknown id returns 404."""
    resp = client.post("/feedback", json={
        "response_id": "nonexistent-id",
        "rating": "down"
    })
    assert resp.status_code == 404
    assert "no answer found" in resp.json()["detail"].lower()
