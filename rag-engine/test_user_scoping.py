"""
Standalone test: confirm user_id scoping in vector_store.py works correctly.
Run this BEFORE touching rag_service.py/main.py, to isolate any bugs.
"""

from pathlib import Path

from chunker import chunk_file
from vector_store import add_chunks, create_collection, load_embedding_model, retrieve

DATA_DIR = Path(__file__).resolve().parent / "data"

print("Loading embedding model...")
embedding_model = load_embedding_model()
collection = create_collection(name="user_scoping_test")

# ---- Index User A's documents (photosynthesis-related) ----
print("\nIndexing User A's documents...")
user_a_files = [
    DATA_DIR / "user_a_docs" / "photosynthesis_overview.txt",
    DATA_DIR / "user_a_docs" / "youtube_lecture_energy.txt",
]
for path in user_a_files:
    chunks = chunk_file(path, min_words=40)  # small min_words so files stay separate chunks
    add_chunks(collection, embedding_model, chunks, user_id="user_a")
    print(f"  Indexed {len(chunks)} chunks from {path.name} for user_a")

# ---- Index User B's documents (Python history) ----
print("\nIndexing User B's documents...")
user_b_files = [
    DATA_DIR / "user_b_docs" / "python_history.txt",
]
for path in user_b_files:
    chunks = chunk_file(path, min_words=40)
    add_chunks(collection, embedding_model, chunks, user_id="user_b")
    print(f"  Indexed {len(chunks)} chunks from {path.name} for user_b")

# ---- Test 1: User A asks a photosynthesis question ----
print("\n" + "=" * 80)
print("TEST 1: User A asks about photosynthesis (should find their own data)")
results = retrieve(collection, embedding_model, "Where does the Calvin cycle occur?", user_id="user_a")
print(f"Retrieved {len(results['documents'])} chunk(s) for user_a")
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"  - source={meta.get('source')}, user_id={meta.get('user_id')}")
    print(f"    preview: {doc[:80]}...")

# ---- Test 2: User B asks about Python (should find their own data) ----
print("\n" + "=" * 80)
print("TEST 2: User B asks about Python (should find their own data)")
results = retrieve(collection, embedding_model, "Who created Python?", user_id="user_b")
print(f"Retrieved {len(results['documents'])} chunk(s) for user_b")
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"  - source={meta.get('source')}, user_id={meta.get('user_id')}")
    print(f"    preview: {doc[:80]}...")

# ---- Test 3: CRITICAL - User B asks a photosynthesis question ----
# User B has NO photosynthesis data. This should return ZERO results,
# proving User B cannot see User A's data even if they ask a related question.
print("\n" + "=" * 80)
print("TEST 3 (CRITICAL): User B asks about photosynthesis")
results = retrieve(collection, embedding_model, "Where does the Calvin cycle occur?", user_id="user_b")
print(f"Retrieved {len(results['documents'])} chunk(s) for user_b")

# The real check: did we get User A's data, or just User B's own (irrelevant) data?
leaked = False
for doc, meta in zip(results["documents"], results["metadatas"]):
    owner = meta.get("user_id")
    print(f"  - source={meta.get('source')}, user_id={owner}, distance={results['distances']}")
    if owner != "user_b":
        leaked = True
        print(f"    LEAKED: this chunk belongs to '{owner}', not user_b!")

if not leaked:
    print("  PASS: every returned chunk belongs to user_b (isolation works, "
          "even though the match quality is poor since user_b has no relevant data)")
else:
    print("  FAIL: User B retrieved another user's data!")

# ---- Test 4: No user_id filter (legacy behavior, should see everyone's data) ----
print("\n" + "=" * 80)
print("TEST 4: Query WITHOUT user_id filter (legacy/admin behavior)")
results = retrieve(collection, embedding_model, "Calvin cycle Python", n_results=6)
print(f"Retrieved {len(results['documents'])} chunk(s) total (no filter)")
seen_users = {meta.get("user_id") for meta in (results["metadatas"] or [])}
print(f"  user_ids present in results: {seen_users}")