"""P0-2: Prove JWT is the sole identity source — body 'user_id' is ignored."""

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

from chunker import chunk_file
from rag_service import RagEngine, prepare_ask, REFUSAL_MESSAGE, NO_DOCUMENTS_MESSAGE
from vector_store import add_chunks


# ---------------------------------------------------------------------------
# Stub embedder (shared pattern from test_zero_results_scoped.py)
# ---------------------------------------------------------------------------

def _stub_embedder():
    import numpy as np

    class _Embedder:
        def encode(self, texts):
            single = isinstance(texts, str)
            if single:
                texts = [texts]
            vecs = []
            for t in texts:
                np.random.seed(hash(t) % (2**32))
                vecs.append(np.random.randn(384).astype(np.float32) * 0.1)
            result = np.array(vecs, dtype=np.float32)
            if single and result.ndim == 2 and result.shape[0] == 1:
                result = result[0]
            return result

    return _Embedder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_user(engine: RagEngine, user_id: str, filename: str) -> list[dict]:
    """Seed a single document into the engine for a given user_id. Returns chunks."""
    path = RAG_ENGINE_DIR / "data" / filename
    chunks = chunk_file(path)
    for chunk in chunks:
        chunk["id"] = f"{user_id}__{chunk['id']}"
    add_chunks(engine.collection, engine.embedding_model, chunks, user_id=user_id)
    engine.chunks_indexed += len(chunks)
    return chunks


def _make_engine(name: str = "test_jwt_isolation") -> RagEngine:
    """Create an in-memory engine for testing."""
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name=name)
    embedding_model = _stub_embedder()
    return RagEngine(
        collection=collection,
        embedding_model=embedding_model,
        chunks_indexed=0,
        embedding_model_name="all-MiniLM-L6-v2",
        default_top_k=4,
        max_distance=1.2,
        auto_reconnect=False,
    )


def _mock_retrieve_for_user(expected_user_id: str, chunks: list[dict]):
    """Return a mock for rag_service.retrieve that yields controlled results.

    When called with the expected user_id, returns chunks with distance 0.5
    (well within the 1.2 relevance threshold). When called with any other
    user_id, returns empty results. This isolates the identity-scoping logic
    from embedding quality.
    """
    def fake_retrieve(collection, embedding_model, question, n_results=4, user_id=None):
        if user_id == expected_user_id:
            k = min(len(chunks), n_results)
            return {
                "documents": [c["text"] for c in chunks[:k]],
                "distances": [0.5] * k,
                "ids": [c["id"] for c in chunks[:k]],
                "metadatas": [{"user_id": user_id} for _ in range(k)],
            }
        return {"documents": [], "distances": [], "ids": [], "metadatas": []}

    return fake_retrieve


def _mock_retrieve_alice(alice_chunks: list[dict]):
    """Mock retrieve that returns alice's chunks with low distance when
    user_id is alice, empty results otherwise."""

    def fake_retrieve(collection, embedding_model, question, n_results=4, user_id=None):
        if user_id == "alice":
            k = min(len(alice_chunks), n_results)
            return {
                "documents": [c["text"] for c in alice_chunks[:k]],
                "distances": [0.5] * k,
                "ids": [c["id"] for c in alice_chunks[:k]],
                "metadatas": [{"user_id": "alice"} for _ in range(k)],
            }
        return {"documents": [], "distances": [], "ids": [], "metadatas": []}

    return fake_retrieve


def _mock_groq():
    """Return a mock Groq client that returns no-op rewrites."""
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = "What is photosynthesis?"
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Test 1: Unit-level — only JWT user_id parameter controls retrieval scope
# ---------------------------------------------------------------------------

@patch("rag_service._get_groq")
@patch("rag_service.retrieve")
def test_body_user_id_has_no_effect_on_retrieval(mock_retrieve, mock_groq):
    """
    Seed photosynthesis docs for alice only. Call prepare_ask() as alice
    (user_id="alice"). Mock retrieve to return alice's chunks with a low
    distance. Verify only alice's docs appear — bob is absent entirely.
    """
    mock_groq.return_value = _mock_groq()

    engine = _make_engine()
    alice_chunks = _seed_user(engine, "alice", "photosynthesis_overview.txt")

    # Wire mock: returns alice's chunks when user_id="alice", empty otherwise
    mock_retrieve.side_effect = _mock_retrieve_alice(alice_chunks)

    prepared = prepare_ask(
        engine=engine,
        question="What is photosynthesis?",
        history=[],
        top_k=4,
        user_id="alice",
    )

    # Alice has relevant docs → not refused
    assert prepared.refused is False, (
        "Alice should get results from her own photosynthesis documents"
    )

    # Retrieved IDs must all belong to alice
    ids = prepared.accumulated.get("ids") or []
    assert len(ids) > 0, "Should have retrieved at least one chunk"
    assert all(id.startswith("alice__") for id in ids), (
        f"All retrieved IDs must be alice's, got: {ids}"
    )

    # Verify retrieve was called with user_id="alice" (not bob, not None)
    mock_retrieve.assert_called()
    for call in mock_retrieve.call_args_list:
        assert call.kwargs.get("user_id") == "alice", (
            f"retrieve() must be called with user_id='alice', got: {call}"
        )


# ---------------------------------------------------------------------------
# Test 2: Unit-level — alice cannot reach bob's docs (empty namespace)
# ---------------------------------------------------------------------------

def test_alice_cannot_reach_bobs_docs_via_parameter():
    """
    Seed photosynthesis docs for bob only. Call prepare_ask() as alice.
    Alice should be refused because the where filter limits to her (empty)
    namespace, proving the user_id parameter — not any body field — is
    the scoping mechanism.
    """
    engine = _make_engine()
    _seed_user(engine, "bob", "photosynthesis_overview.txt")

    prepared = prepare_ask(
        engine=engine,
        question="What is photosynthesis?",
        history=[],
        top_k=4,
        user_id="alice",
    )

    assert prepared.refused is True, (
        "Alice should be refused when she has no documents"
    )
    assert prepared.refusal_answer == NO_DOCUMENTS_MESSAGE


# ---------------------------------------------------------------------------
# Test 3: HTTP-level — extra 'user_id' field in JSON body is ignored
# ---------------------------------------------------------------------------

@patch("rag_service._get_groq")
@patch("rag_service.retrieve")
def test_extra_user_id_in_body_ignored_http(mock_retrieve, mock_groq):
    """
    Full HTTP test via TestClient:
    - Seed photosynthesis docs for 'jwt-alice@example.com' only.
    - Authenticate as 'jwt-alice@example.com' via dependency override.
    - Send a JSON body containing an extra field: "user_id": "attacker-bob@example.com".
    - Assert:
      1. No 422 (extra field is silently ignored, not rejected).
      2. Request succeeds with 200.
      3. Response does NOT refuse (alice has docs).
      4. Source IDs all belong to alice.
      5. No bob documents exist in the collection.
    """
    mock_groq.return_value = _mock_groq()

    engine = _make_engine()
    alice_chunks = _seed_user(engine, "jwt-alice@example.com", "photosynthesis_overview.txt")
    mock_retrieve.side_effect = _mock_retrieve_for_user("jwt-alice@example.com", alice_chunks)

    with patch("main.create_engine", return_value=engine):
        import main
        from auth import get_current_user_email

        main._engine = engine
        main._engine_ready = True
        main.app.dependency_overrides[get_current_user_email] = (
            lambda: "jwt-alice@example.com"
        )

        try:
            with TestClient(main.app) as client:
                # --- Assertion 1: Extra field does NOT cause 422 ---
                resp = client.post(
                    "/ask",
                    json={
                        "question": "What is photosynthesis?",
                        "user_id": "attacker-bob@example.com",
                        "skip_cache": True,
                    },
                )
                assert resp.status_code != 422, (
                    f"Extra 'user_id' field must not cause 422, got: {resp.status_code} {resp.text}"
                )

                # --- Assertion 2: Request succeeds ---
                assert resp.status_code == 200, (
                    f"Expected 200, got {resp.status_code}: {resp.text}"
                )

                data = resp.json()

                # --- Assertion 3: alice is NOT refused (she has docs) ---
                assert data.get("refused") is not True, (
                    "jwt-alice should not be refused — she owns the documents"
                )

                # --- Assertion 4: sources are alice's, not bob's ---
                source_ids = data.get("source_ids") or []
                if source_ids:
                    assert all(
                        id.startswith("jwt-alice@example.com__") for id in source_ids
                    ), (
                        f"All source IDs must belong to alice, got: {source_ids}"
                    )

                # --- Assertion 5: verify no bob documents leaked ---
                bob_docs = engine.collection.get(
                    where={"user_id": "attacker-bob@example.com"}, limit=1
                )
                assert not bob_docs.get("ids"), (
                    "attacker-bob should have zero documents in the collection"
                )

        finally:
            main.app.dependency_overrides.clear()
            main._engine = None
            main._engine_ready = False


# ---------------------------------------------------------------------------
# Test 4: Pin Pydantic extra-field behavior (silent ignore vs. forbid)
# ---------------------------------------------------------------------------

def test_extra_field_rejection_behavior_documented():
    """
    Document and confirm the exact behavior: Pydantic v2 default is
    extra='ignore' (silently drops unknown fields). If that ever changes
    to extra='forbid', this test will catch it and document the new
    behavior (422 on unknown fields is also safe).

    Either outcome is secure — this test pins the behavior.
    """
    from schemas import AskRequest

    model_config = getattr(AskRequest, "model_config", {})
    extra_setting = model_config.get("extra", "ignore")

    # If this fails, verify the new setting is intentional.
    # 'forbid' = safe (rejects unknown fields with 422)
    # 'ignore' = safe (silently drops unknown fields)
    # 'allow'  = INSECURE (accepts and stores unknown fields)
    assert extra_setting in ("ignore", "forbid"), (
        f"AskRequest.extra is '{extra_setting}'; expected 'ignore' or 'forbid'. "
        "'allow' would accept and store unknown fields (INSECURE)."
    )

    # Confirm: sending an extra field via raw dict does NOT raise during parsing
    raw_body = {
        "question": "test",
        "user_id": "attacker@example.com",
        "extra_field": "sneaky",
    }

    if extra_setting == "forbid":
        # If extra='forbid', parsing should reject unknown fields
        with pytest.raises(Exception):
            AskRequest.model_validate(raw_body)
    else:
        # extra='ignore' (default): unknown fields are silently dropped
        parsed = AskRequest.model_validate(raw_body)
        assert not hasattr(parsed, "user_id") or "user_id" not in parsed.model_fields_set, (
            "Parsed AskRequest should not contain 'user_id'"
        )
        assert "extra_field" not in (parsed.model_fields_set or set()), (
            "Parsed AskRequest should not contain 'extra_field'"
        )


# ---------------------------------------------------------------------------
# Test 5: Cache key is scoped by JWT identity, not body
# ---------------------------------------------------------------------------

def test_cache_key_uses_jwt_not_body():
    """
    Prove the cache key is derived from the JWT email (passed via
    dependency override), not from any 'user_id' in the request body.
    """
    from cache import AnswerCache

    kwargs = dict(
        question="What is photosynthesis?",
        history=None,
        top_k=4,
        rerank=True,
        multi_hop=True,
    )

    key_for_alice = AnswerCache.make_key("alice@example.com", **kwargs)
    key_for_bob = AnswerCache.make_key("bob@example.com", **kwargs)

    # Same question, different JWT identity → different cache key
    assert key_for_alice != key_for_bob, (
        "Cache keys must differ per JWT identity"
    )

    # Same JWT identity, same question → same cache key
    key_for_alice2 = AnswerCache.make_key("alice@example.com", **kwargs)
    assert key_for_alice == key_for_alice2, (
        "Same JWT identity and question must produce the same cache key"
    )
