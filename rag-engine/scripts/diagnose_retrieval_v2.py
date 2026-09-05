"""
Minimal diagnostic to evidence retrieval effectiveness (semantic search for exact terms).
Run this to reproduce the claim: 'semantic search handles exact terms well'.
"""

from pathlib import Path
import sys
import os

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import create_engine, add_user_document, prepare_ask, load_env

# Setup engine
load_env()
# Using a dedicated collection
engine = create_engine(collection_name="evidence_collection")
user = "evidence_user"

# Ingest dummy terms
data_dir = RAG_ENGINE_DIR / "data" / "evidence"
data_dir.mkdir(parents=True, exist_ok=True)

(data_dir / "acronym.txt").write_text("The acronym for Adenosine Triphosphate is ATP.", encoding="utf-8")
(data_dir / "constant.txt").write_text("The speed of light in a vacuum is 299,792,458 meters per second.", encoding="utf-8")

add_user_document(engine, data_dir / "acronym.txt", user)
add_user_document(engine, data_dir / "constant.txt", user)

def check_evidence(q, term):
    print(f"\n--- Diagnostic: Querying '{q}' ---")
    prepared = prepare_ask(engine, q, user_id=user)
    found = term in prepared.retrieved_text
    print(f"Term '{term}' found: {found}")
    print(f"Text snippet: {prepared.retrieved_text[:100]}...")

check_evidence("What is ATP?", "ATP")
check_evidence("What is 299,792,458 m/s?", "299,792,458")
check_evidence("Speed of light value?", "299,792,458")
