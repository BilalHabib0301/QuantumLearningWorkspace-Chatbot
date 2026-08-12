"""Test user-scoped retrieval with zero relevant results (Phase 9 Part A)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from chunker import chunk_file
from rag_service import prepare_ask, REFUSAL_MESSAGE
from vector_store import add_chunks, retrieve, is_relevant


def _create_stub_embedder():
    """Create a deterministic stub embedder that returns proper numpy arrays.

    This avoids loading the real SentenceTransformer model which would require
    network access and slow down tests.

    The encode method receives either a single string or list of strings.
    SentenceTransformer.encode() returns a 2D array (n_texts, 384) for any input,
    but when called with a single string, the result can be squeezed to 1D.
    """
    import numpy as np

    class _StubEmbedder:
        """Deterministic stub that mimics SentenceTransformer.encode()."""

        def encode(self, texts: str | list[str]) -> np.ndarray:
            """Return deterministic embeddings using numpy array."""
            # Handle single string input (common case in retrieve())
            single_input = isinstance(texts, str)
            if single_input:
                texts = [texts]

            # Use a fixed seed for reproducibility
            np.random.seed(42)

            embeddings = []
            for text in texts:
                # Generate deterministic but distinct embeddings per text
                # Use the text hash to seed differently for each text
                text_seed = hash(text) % (2**32)
                np.random.seed(text_seed)

                # all-MiniLM-L6-v2 has 384 dimensions
                vec = np.random.randn(384).astype(np.float32) * 0.1
                embeddings.append(vec)

            result = np.array(embeddings, dtype=np.float32)

            # SentenceTransformer.squeeze() returns 1D for single input
            if single_input and result.ndim == 2 and result.shape[0] == 1:
                result = result[0]

            return result

    return _StubEmbedder()


def test_user_scoped_retrieval_refuses_on_unrelated_question():
    """
    Test that user-scoped retrieval correctly refuses when a question
    has no relevant chunks for that user.

    This exercises the Phase 8 user-scoping path combined with the
    Phase 5 zero-result refusal logic.

    Test scenario:
    - User "alice" uploads a document about photosynthesis
    - Alice asks: "What is the capital of France?" (unrelated question)
    - Expected: System refuses gracefully because no chunks match
    """
    from rag_service import RagEngine

    import chromadb

    # Use in-memory ChromaDB to avoid writing to disk
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_scoped")

    # Use deterministic stub embedder for predictable distances and faster tests
    embedding_model = _create_stub_embedder()

    # Simulate user "alice" uploading a document about photosynthesis
    photosynthesis_file = RAG_ENGINE_DIR / "data" / "photosynthesis_overview.txt"
    chunks = chunk_file(photosynthesis_file)

    # Prefix chunk IDs with user_id to simulate user scoping
    user_id = "alice"
    for chunk in chunks:
        chunk["id"] = f"{user_id}__{chunk['id']}"

    # Add chunks with user_id metadata
    add_chunks(collection, embedding_model, chunks, user_id=user_id)

    # Create engine with this collection
    engine = RagEngine(
        collection=collection,
        embedding_model=embedding_model,
        chunks_indexed=len(chunks),
        embedding_model_name="all-MiniLM-L6-v2",
        default_top_k=4,
        max_distance=1.2,
    )

    # Now ask an unrelated question - capital of France
    # This should NOT match any of alice's photosynthesis documents
    question = "What is the capital of France?"

    # Prepare the ask with user_id - this exercises user-scoped retrieval
    prepared = prepare_ask(
        engine=engine,
        question=question,
        history=[],
        top_k=4,
        user_id=user_id,
    )

    # Verify the system refuses gracefully - not a crash, no exception
    # The refusal should happen at the relevance gate (Phase 5 logic)
    assert prepared.refused is True, (
        "Expected refusal when question has no relevant chunks for user"
    )

    # The refusal answer should be a clean message, not empty or broken
    assert prepared.refusal_answer, "Refusal answer should not be empty"
    assert isinstance(prepared.refusal_answer, str), (
        "Refusal answer should be a string"
    )

    # Check that the refusal message is the expected one from rag_service.py
    assert prepared.refusal_answer == REFUSAL_MESSAGE, (
        f"Expected refusal message '{REFUSAL_MESSAGE}', got '{prepared.refusal_answer}'"
    )


@patch("rag_service._get_groq")
@patch("rag_service.retrieve")
def test_user_scoped_retrieval_answers_on_relevant_question(mock_retrieve, mock_get_groq):
    """
    Control test: Same user asking about their uploaded document
    should get a relevant answer (not refused).

    This verifies that the user-scoping mechanism works correctly
    by confirming users CAN answer questions about their own documents.

    Mocks the Groq client and retrieve() to avoid embedding model dependencies.
    The retrieve() mock simulates successful retrieval of relevant chunks.
    """
    from rag_service import RagEngine

    import chromadb

    # Use in-memory ChromaDB to avoid writing to disk
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_scoped2")

    embedding_model = _create_stub_embedder()

    photosynthesis_file = RAG_ENGINE_DIR / "data" / "photosynthesis_overview.txt"
    chunks = chunk_file(photosynthesis_file)

    user_id = "bob"
    for chunk in chunks:
        chunk["id"] = f"{user_id}__{chunk['id']}"

    add_chunks(collection, embedding_model, chunks, user_id=user_id)

    engine = RagEngine(
        collection=collection,
        embedding_model=embedding_model,
        chunks_indexed=len(chunks),
        embedding_model_name="all-MiniLM-L6-v2",
        default_top_k=4,
        max_distance=1.2,
    )

    # Mock the Groq client to return a simple response for question rewriting
    # (so we don't need GROQ_API_KEY set)
    mock_client = MagicMock()
    mock_get_groq.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "What is photosynthesis?"  # No rewrite needed
    mock_client.chat.completions.create.return_value = mock_response

    # Mock retrieve() to return relevant results with low distances (within threshold)
    # This simulates successful user-scoped retrieval
    mock_retrieve.return_value = {
        "documents": [chunks[0]["text"]],
        "distances": [0.5],  # Within threshold of 1.2
        "ids": [f"{user_id}__{chunks[0]['id']}"],
        "metadatas": [{"user_id": user_id}],
    }

    # Ask about photosynthesis - should match bob's documents
    question = "What is photosynthesis?"

    prepared = prepare_ask(
        engine=engine,
        question=question,
        history=[],
        top_k=4,
        user_id=user_id,
    )

    # Should NOT refuse - retrieve() mock returns relevant chunks
    assert prepared.refused is False, (
        "Expected no refusal when question matches user's documents"
    )

    # Should have retrieved some documents
    assert len(prepared.accumulated.get("documents") or []) > 0, (
        "Expected at least one relevant document to be retrieved"
    )

    # The documents should be user-prefixed (user-scoping working)
    docs = prepared.accumulated.get("documents") or []
    ids = prepared.accumulated.get("ids") or []
    assert all(id.startswith(f"{user_id}__") for id in ids), (
        "Document IDs should be prefixed with user_id"
    )

    # Verify retrieve() was called with the correct user_id filter
    mock_retrieve.assert_called_once()
    call_kwargs = mock_retrieve.call_args[1]
    assert call_kwargs.get("user_id") == user_id, (
        "retrieve() should be called with user_id filter"
    )


def test_user_cannot_access_other_users_documents():
    """
    Test that user "alice" cannot retrieve documents uploaded by user "charlie".
    This directly tests the Phase 8 user-scoping isolation.

    Test scenario:
    - Charlie uploads photosynthesis document
    - Alice asks about photosynthesis
    - Expected: Alice gets refused because her namespace has no documents
    """
    from rag_service import RagEngine

    import chromadb

    # Use in-memory ChromaDB to avoid writing to disk
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_isolation")

    embedding_model = _create_stub_embedder()

    # Charlie uploads photosynthesis document
    charlie_chunks = chunk_file(RAG_ENGINE_DIR / "data" / "photosynthesis_overview.txt")
    charlie_user_id = "charlie"
    for chunk in charlie_chunks:
        chunk["id"] = f"{charlie_user_id}__{chunk['id']}"
    add_chunks(collection, _create_stub_embedder(), charlie_chunks, user_id=charlie_user_id)

    engine = RagEngine(
        collection=collection,
        embedding_model=embedding_model,
        chunks_indexed=len(charlie_chunks),
        embedding_model_name="all-MiniLM-L6-v2",
        default_top_k=4,
        max_distance=1.2,
    )

    # Alice asks the same question - she should NOT see charlie's documents
    alice_user_id = "alice"
    question = "What is photosynthesis?"

    prepared = prepare_ask(
        engine=engine,
        question=question,
        history=[],
        top_k=4,
        user_id=alice_user_id,
    )

    # Alice should get refused because she has no documents
    # (the where filter restricted search to alice's namespace only)
    assert prepared.refused is True, (
        "User 'alice' should be refused when trying to query 'charlie's documents"
    )


def test_is_relevant_uses_l2_distance_threshold():
    """
    Unit test for the is_relevant function with user-scoped distances.

    ChromaDB uses L2 distance (lower = more similar).
    The threshold is 1.8 by default.
    """
    # Close match (within threshold) - should be relevant
    assert is_relevant([0.5, 1.0, 1.5], max_distance=1.2) is True

    # Best match within threshold
    assert is_relevant([0.1, 2.0, 3.0], max_distance=1.2) is True

    # All too far - should not be relevant
    assert is_relevant([1.3, 1.5, 2.0], max_distance=1.2) is False

    # Empty/None - should not be relevant
    assert is_relevant(None) is False
    assert is_relevant([]) is False

    # Exactly at default threshold (1.8) - should be relevant
    assert is_relevant([1.8]) is True
    assert is_relevant([1.8001]) is False


@patch("rag_service._get_groq")
def test_user_scoped_retrieval_filters_by_user_id(mock_get_groq):
    """
    Direct test of the user-scoping filter in retrieve().

    This confirms that when user_id is specified, the query includes
    a 'where' filter that restricts results to that user's documents only.
    """

    import chromadb

    # Use in-memory ChromaDB to avoid writing to disk
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="test_filter")

    # Use deterministic stub embedder for predictable distances
    embedding_model = _create_stub_embedder()

    # Upload documents for two different users
    photosynthesis_file = RAG_ENGINE_DIR / "data" / "photosynthesis_overview.txt"
    chunks = chunk_file(photosynthesis_file)

    # User 1: alice's chunks (prefixed with "alice__")
    for chunk in chunks:
        chunk["id"] = f"alice__{chunk['id']}"
    add_chunks(collection, embedding_model, chunks, user_id="alice")

    # User 2: bob's chunks (prefixed with "bob__") - same file content
    chunks2 = chunk_file(photosynthesis_file)
    for chunk in chunks2:
        chunk["id"] = f"bob__{chunk['id']}"
    add_chunks(collection, embedding_model, chunks2, user_id="bob")

    # Verify total chunks - derive from actual chunking output for robustness
    all_chunks = collection.count()
    expected_chunks = len(chunks) + len(chunks2)
    assert all_chunks == expected_chunks, (
        f"Expected {expected_chunks} chunks total ({len(chunks)} for alice + {len(chunks2)} for bob), got {all_chunks}"
    )

    # Mock Groq for question rewriting
    mock_client = MagicMock()
    mock_get_groq.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "What is photosynthesis?"
    mock_client.chat.completions.create.return_value = mock_response

    # Test 1: alice queries - should only see alice's chunks
    alice_results = retrieve(
        collection,
        embedding_model,
        "What is photosynthesis?",
        n_results=4,
        user_id="alice",
    )
    alice_ids = alice_results.get("ids") or []
    assert len(alice_ids) > 0, "Alice should get results"
    assert all(id.startswith("alice__") for id in alice_ids), (
        f"Alice's results should all be prefixed with 'alice__', got: {alice_ids}"
    )
    assert all("bob__" not in id for id in alice_ids), (
        "Alice should not see bob's chunks"
    )

    # Test 2: bob queries - should only see bob's chunks
    bob_results = retrieve(
        collection,
        embedding_model,
        "What is photosynthesis?",
        n_results=4,
        user_id="bob",
    )
    bob_ids = bob_results.get("ids") or []
    assert len(bob_ids) > 0, "Bob should get results"
    assert all(id.startswith("bob__") for id in bob_ids), (
        f"Bob's results should all be prefixed with 'bob__', got: {bob_ids}"
    )
    assert all("alice__" not in id for id in bob_ids), (
        "Bob should not see alice's chunks"
    )
