"""Check if conflicting docs are retrieved together."""
from pathlib import Path
import sys

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

import main
from rag_service import create_engine, prepare_ask
from dotenv import load_dotenv
from pathlib import Path

# Load env
load_dotenv(RAG_ENGINE_DIR / '.env')

# Ingest conflicting files for a test user
def ingest_conflicts():
    print("Creating test conflicting documents...")
    p1 = RAG_ENGINE_DIR / "data" / "conflicts" / "temp_25.txt"
    p2 = RAG_ENGINE_DIR / "data" / "conflicts" / "temp_40.txt"
    
    # Ensure main engine exists
    main._engine = create_engine(collection_name="test_conflict_db")
    main._engine_ready = True
    
    # Use a specific test user
    user = "conflict_test_user"
    from rag_service import add_user_document
    
    add_user_document(main._engine, p1, user)
    add_user_document(main._engine, p2, user)
    print("Ingested both conflicting documents.")

def test_retrieval():
    print("\nTesting retrieval for conflicting documents...")
    user = "conflict_test_user"
    prepared = prepare_ask(
        main._engine,
        "What is the exact temperature for the reaction?",
        user_id=user
    )
    print(f"Retrieved text:\n{prepared.retrieved_text}")
    
    if "25C" in prepared.retrieved_text and "40C" in prepared.retrieved_text:
        print("\nSUCCESS: Both conflicting facts retrieved in the same context window.")
        return True
    else:
        print("\nFAILURE: Only one conflicting fact retrieved.")
        return False

if __name__ == "__main__":
    ingest_conflicts()
    success = test_retrieval()
    if not success:
        sys.exit(1)
