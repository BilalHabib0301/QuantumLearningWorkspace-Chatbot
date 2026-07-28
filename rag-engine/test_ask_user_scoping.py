"""
End-to-end test: confirm user_id scoping works through the full ask() pipeline,
including the is_relevant() threshold check.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from rag_service import ask, add_user_document, create_engine, load_env

DATA_DIR = Path(__file__).resolve().parent / "data"

load_env()

print("Creating engine (loads demo corpus)...")
engine = create_engine(collection_name="user_scoping_ask_test")
print(f"Demo corpus indexed: {engine.chunks_indexed} chunks (no user_id, shared demo data)\n")

print("Adding User A's documents...")
n = add_user_document(engine, DATA_DIR / "user_a_docs" / "photosynthesis_overview.txt", user_id="user_a")
print(f"  Added {n} chunks for user_a")

print("Adding User B's documents...")
n = add_user_document(engine, DATA_DIR / "user_b_docs" / "python_history.txt", user_id="user_b")
print(f"  Added {n} chunks for user_b")

print(f"\nTotal chunks now indexed: {engine.chunks_indexed}\n")

# ---- Test 1: User A asks about photosynthesis ----
print("=" * 80)
print("TEST 1: User A asks 'Where does the Calvin cycle occur?'")
result = ask(engine, "Where does the Calvin cycle occur?", user_id="user_a")
print(f"  refused={result.refused}, grounded={result.grounded}")
print(f"  answer: {result.answer[:150]}...")

# ---- Test 2: User B asks about Python ----
print("\n" + "=" * 80)
print("TEST 2: User B asks 'Who created Python?'")
result = ask(engine, "Who created Python?", user_id="user_b")
print(f"  refused={result.refused}, grounded={result.grounded}")
print(f"  answer: {result.answer[:150]}...")

# ---- Test 3: CRITICAL - User B asks about photosynthesis ----
print("\n" + "=" * 80)
print("TEST 3 (CRITICAL): User B asks 'Where does the Calvin cycle occur?'")
print("Expected: system should REFUSE, since user_b has no photosynthesis data")
result = ask(engine, "Where does the Calvin cycle occur?", user_id="user_b")
print(f"  refused={result.refused}, grounded={result.grounded}")
print(f"  answer: {result.answer}")
if result.refused:
    print("  PASS: correctly refused (no cross-user leakage, no false answer)")
else:
    print("  FAIL: system answered instead of refusing - check for leakage!")
    for s in result.sources:
        print(f"    source used: {s.source} (id={s.id})")