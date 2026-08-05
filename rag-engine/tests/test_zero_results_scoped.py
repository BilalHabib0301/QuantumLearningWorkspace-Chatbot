"""Test user-scoped retrieval with zero relevant results (Phase 9 Part A)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from chunker import chunk_file
from rag_service import prepare_ask, create_engine, REFUSAL_MESSAGE, PreparedAsk
from vector_store import (
    add_chunks,
    create_collection,
    load_embedding_model,
    retrieve,
    is_relevant,
)


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

    # Use a unique collection name for this test
    collection_name = f"test_scoped_{uuid.uuid4().hex[:8]}"

    # Load embedding model first (loaded once for all tests in this file)
    embedding_model = load_embedding_model()

    # Get persistent client (same as the main engine uses)
    from vector_store import DEFAULT_CHROMA_PATH
    client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    # Simulate user "alice" uploading a document about photosynthesis
    # (This mimics what add_user_document does in rag_service.py)
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

    # Cleanup: delete the collection
    client.delete_collection(name=collection_name)


@patch("rag_service._get_groq")
def test_user_scoped_retrieval_answers_on_relevant_question(mock_get_groq):
    """
    Control test: Same user asking about their uploaded document
    should get a relevant answer (not refused).

    This verifies that the user-scoping mechanism works correctly
    by confirming users CAN answer questions about their own documents.

    Mocks the Groq client since we only need to test the retrieval path,
    not the actual LLM answer generation.
    """
    from rag_service import RagEngine

    import chromadb

    # Use a unique collection name for this test
    collection_name = f"test_scoped_{uuid.uuid4().hex[:8]}"

    # Get persistent client
    from vector_store import DEFAULT_CHROMA_PATH
    client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    embedding_model = load_embedding_model()

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

    # Ask about photosynthesis - should match bob's documents
    question = "What is photosynthesis?"

    prepared = prepare_ask(
        engine=engine,
        question=question,
        history=[],
        top_k=4,
        user_id=user_id,
    )

    # Should NOT refuse - there are relevant chunks for this user
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

    # Cleanup: delete the collection
    client.delete_collection(name=collection_name)


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

    # Use a unique collection name for this test
    collection_name = f"test_isolation_{uuid.uuid4().hex[:8]}"

    # Get persistent client
    from vector_store import DEFAULT_CHROMA_PATH
    client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    embedding_model = load_embedding_model()

    # Charlie uploads photosynthesis document
    charlie_chunks = chunk_file(RAG_ENGINE_DIR / "data" / "photosynthesis_overview.txt")
    charlie_user_id = "charlie"
    for chunk in charlie_chunks:
        chunk["id"] = f"{charlie_user_id}__{chunk['id']}"
    add_chunks(collection, embedding_model, charlie_chunks, user_id=charlie_user_id)

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

    # Cleanup: delete the collection
    client.delete_collection(name=collection_name)


def test_is_relevant_uses_l2_distance_threshold():
    """
    Unit test for the is_relevant function with user-scoped distances.

    ChromaDB uses L2 distance (lower = more similar).
    The threshold is 1.2 by default.
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

    # Exactly at threshold - should be relevant
    assert is_relevant([1.2]) is True
    assert is_relevant([1.2001]) is False


@patch("rag_service._get_groq")
def test_user_scoped_retrieval_filters_by_user_id(mock_get_groq):
    """
    Direct test of the user-scoping filter in retrieve().

    This confirms that when user_id is specified, the query includes
    a 'where' filter that restricts results to that user's documents only.
    """
    import chromadb

    # Use a unique collection name for this test
    collection_name = f"test_filter_{uuid.uuid4().hex[:8]}"

    # Get persistent client
    from vector_store import DEFAULT_CHROMA_PATH
    client = chromadb.PersistentClient(path=DEFAULT_CHROMA_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    embedding_model = load_embedding_model()

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

    # Verify total chunks
    all_chunks = collection.count()
    assert all_chunks == 8, f"Expected 8 chunks total (4 for alice + 4 for bob), got {all_chunks}"

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

    # Cleanup: delete the collection
    client.delete_collection(name=collection_name)
