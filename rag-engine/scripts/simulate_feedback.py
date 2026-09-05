"""Simulate user feedback scenarios to identify failure patterns.

Uses a mocked RAG pipeline so it runs without GROQ_API_KEY or a live
Chroma collection.  The point is to exercise the /ask → /feedback flow,
review what the store contains, and discover failure-pattern categories
that warrant regression tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

import main  # noqa: E402
from auth import get_current_user_email  # noqa: E402
from feedback_store import feedback_store  # noqa: E402
from rag_service import AskResult, PreparedAsk, SourceInfo  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


SCENARIOS = [
    {
        "name": "Extreme Ambiguity",
        "question": "What is the process?",
        "answer": "The process is photosynthesis.",
        "grounded": True,
        "source_ids": ["c1"],
        "refused": False,
        "pattern": "vague_query",
    },
    {
        "name": "Non-existent Content",
        "question": "How does StudyMind handle quantum gravity calculations?",
        "answer": "I don't have enough information to answer that.",
        "grounded": None,
        "source_ids": [],
        "refused": True,
        "pattern": "out_of_scope",
    },
    {
        "name": "Cross-document Conflict",
        "question": "What is the exact temperature for the reaction?",
        "answer": "The temperature is 25C.",
        "grounded": True,
        "source_ids": ["c2"],
        "refused": False,
        "pattern": "conflict_unreported",
    },
    {
        "name": "Complex Negation",
        "question": "What cell parts are NOT involved in photosynthesis?",
        "answer": "Chloroplasts are involved in photosynthesis.",
        "grounded": True,
        "source_ids": ["c3"],
        "refused": False,
        "pattern": "negation_ignored",
    },
]


def _build_mock(sc: dict):
    prepared = PreparedAsk(
        question=sc["question"],
        history=[],
        top_k=4,
        rewritten_question=sc["question"],
        hop_queries=[sc["question"]],
        retrieved_text="chunk text",
        accumulated={
            "documents": ["chunk text"],
            "ids": sc["source_ids"],
            "distances": [0.5],
            "metadatas": [{}],
        },
        refused=sc["refused"],
        refusal_answer=sc["answer"] if sc["refused"] else "",
        include_sources=True,
        client=MagicMock(),
    )
    result = AskResult(
        answer=sc["answer"],
        refused=sc["refused"],
        top_k=4,
        sources=[SourceInfo(id=s, distance=0.5, preview="") for s in sc["source_ids"]],
        source_ids=list(sc["source_ids"]),
        rewritten_question=sc["question"],
        grounded=sc["grounded"],
        retrieval_rounds=1,
        hop_queries=[sc["question"]],
    )
    return prepared, result


def run_simulation():
    print("--- Starting Feedback Simulation (mocked pipeline) ---")

    mock_engine = MagicMock()
    mock_engine.chunks_indexed = 10
    mock_engine.embedding_model_name = "test-model"
    mock_engine.default_top_k = 4
    mock_engine.auto_reconnect = False
    main._engine = mock_engine
    main._engine_ready = True
    main.app.dependency_overrides[get_current_user_email] = lambda: "sim@example.com"

    feedback_store.clear()

    with TestClient(main.app) as client:
        with patch("main.prepare_ask") as mp, \
             patch("main.generate_answer_sync") as mg, \
             patch("main.finalize_ask") as mf:

            mg.return_value = ""
            for sc in SCENARIOS:
                prepared, result = _build_mock(sc)
                mp.return_value = prepared
                mf.return_value = result

                resp = client.post(
                    "/ask",
                    json={"question": sc["question"], "skip_cache": True},
                )
                print(f"\n[{sc['name']}] status={resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"  answer : {data['answer'][:80]}")
                    print(f"  grounded: {data['grounded']}  refused: {data['refused']}")
                    feedback_store.submit(data["response_id"], "down")
                    print(f"  -> down-voted (store size: {feedback_store.count()})")

    main.app.dependency_overrides.clear()
    main._engine = None
    main._engine_ready = False

    print("\n=== Negative Feedback Review ===")
    for rec in feedback_store.by_rating("down"):
        print(f"\n  question     : {rec.question}")
        print(f"  answer       : {rec.answer[:80]}")
        print(f"  sources      : {rec.sources}")
        print(f"  created_at   : {rec.created_at}")

    return feedback_store.by_rating("down")


if __name__ == "__main__":
    run_simulation()
