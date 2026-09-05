"""Reproducible diagnostic: semantic search handles exact terms well.

This script supports the architecture.md claim that a keyword-based
hybrid search adds minimal value for our current scope.

It ingests simple documents containing exact terms (acronyms and numbers)
and verifies that the semantic search (SentenceTransformer + ChromaDB)
retrieves them correctly, even with rephrased queries.

Usage:
    rag_venv\Scripts\python.exe rag-engine\scripts\verify_retrieval_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import create_engine, add_user_document, prepare_ask, load_env  # noqa: E402

TEST_COLLECTION = "verify_retrieval_evidence"
TEST_USER = "diagnostic_user"
DATA_DIR = RAG_ENGINE_DIR / "data" / "diagnostic_terms"

# 1. Setup & Ingestion
def setup():
    load_env()
    engine = create_engine(collection_name=TEST_COLLECTION)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    docs = {
        "acronym.txt": "The acronym for Adenosine Triphosphate is ATP.",
        "constant.txt": "The speed of light in a vacuum is 299,792,458 meters per second.",
        "photo_nadph.txt": "NADPH is produced during the light-dependent reactions in the thylakoid membrane.",
    }
    for name, text in docs.items():
        p = DATA_DIR / name
        p.write_text(text, encoding="utf-8")
        add_user_document(engine, p, TEST_USER)
    return engine

# 2. Query & Verification
def run_evidence(engine):
    print("--- Retrieval Evidence Test ---")
    tests = [
        ("Acronym (ATP)", "What is ATP?", "acronym.txt"),
        ("Number (Speed of Light)", "What is the speed of light?", "constant.txt"),
        ("Exact Number (Rephrased)", "How fast is light in a vacuum?", "constant.txt"),
        ("Specific Mechanism (NADPH)", "How is NADPH produced?", "photo_nadph.txt"),
    ]
    
    all_passed = True
    for name, query, expected_source in tests:
        print(f"\nQuery: {query}")
        prepared = prepare_ask(engine, query, user_id=TEST_USER)
        sources = [m.get('source') for m in (prepared.accumulated.get('metadatas') or [])]
        
        passed = expected_source in sources
        print(f"  Status: {'PASS' if passed else 'FAIL'} (Expected: {expected_source}, Got: {sources})")
        if not passed:
            all_passed = False
            
    return all_passed

if __name__ == "__main__":
    engine = setup()
    success = run_evidence(engine)
    print("\n" + ("✅ All exact-term retrieval tests passed." if success else "❌ Some tests failed."))
    sys.exit(0 if success else 1)
