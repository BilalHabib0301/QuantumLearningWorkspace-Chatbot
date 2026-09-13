
## Retrieval Strategy (Phase 10)

Our retrieval pipeline uses a **semantic-first approach** with **LLM re-ranking**.

### 1. Semantic Search
We use `all-MiniLM-L6-v2` to embed questions and retrieve the top 10 candidates from ChromaDB. Our diagnostic tests (Phase 10 Part B) confirmed that semantic search is already highly effective at capturing exact terms, acronyms, and numbers — even with rephrased queries (see `scripts/diagnose_retrieval_v2.py`). This made keyword-based hybrid search unnecessary for our current scope.

### 2. LLM Re-ranking
We use an LLM call to pick the top 3-4 most relevant chunks from the 10 candidates. The re-ranking prompt instructs the LLM to prioritize chunks that cover different aspects of a question or reveal contradictions.

**Important finding from multi-run testing (5 runs x 2 prompts):** For this specific tested conflict scenario (temperature contradiction), the Groq model (GPT-OSS-120B) already included both conflicting chunks in 100% of runs, even with the simpler "relevance only" prompt. This indicates the base model handles this specific type of two-source numeric conflict reasonably well; the "conflict-aware" prompt is added as explicit reinforcement and documentation of architectural intent, rather than a fix for a previously reproducible failure in this exact scenario. Its broader value lies in:
- Guarding against potential regression if the base model changes
- Providing explicit guidance for more ambiguous or subtle conflict scenarios
- Documenting architectural intent clearly in the prompt itself

### Rationale for choosing Re-ranking over Hybrid Search
We considered adding keyword-based hybrid search to catch "missed" exact terms. However, our diagnostic tests showed that semantic search was already retrieving those terms correctly. The primary failures we observed (negation being ignored, conflicts being missed) were issues of **LLM comprehension and heuristic scope**, not initial retrieval gaps. Enhancing the LLM re-ranking step to be more "aware" of multi-source contexts was deemed a higher-impact improvement — primarily as a safeguard and explicit instruction layer.

---

## Hybrid Search A/B (Phase 11 Task 1)

Phase 11 added a **hybrid retrieval path** (semantic + BM25 fused via reciprocal rank fusion) alongside the existing semantic-only path, so both can be compared on the same question set while holding LLM re-ranking constant.

### Architecture

- **`bm25.py`**: Pure-Python Okapi BM25 index (no external dependencies). Tokenizes on lowercase alphanumeric, removes a small English stopword set + content-free terms. Builds a static index over the collection's documents via `collection.get()`.
- **`hybrid_search.py`**: Fuses ChromaDB semantic search (`all-MiniLM-L6-v2` top-20) with BM25 lexical top-20 using **Reciprocal Rank Fusion** (k=60). Returns the same dict shape as `vector_store.retrieve()` so downstream LLM re-ranking, grounding, and merging are unchanged.
- **Toggle**: `RETRIEVAL_METHOD=semantic|hybrid` env var (default `semantic`). Also available as `retrieval_method=` kwarg through `ask()`, `prepare_ask()`, `_retrieve_round()`.
- **Hybrid relevance gate**: `is_hybrid_relevant()` accepts when (a) any fused chunk has L2 distance <= max_distance, OR (b) at least one fused chunk is lexical-only (real keyword overlap — off-topic queries share no content words with the corpus, so BM25 contributes nothing).
- **Lexical index caching**: Built lazily on first hybrid query, cached on the engine keyed by sorted document id set. Invalidated automatically when new documents are added. First-call overhead: ~10 ms on demo corpus.

### Comparison Results (15 shared eval cases)

**Retrieval-level** (top-10 candidate pool, no LLM reranking):

| metric | semantic | hybrid |
|---|---|---|
| mean anchor hit@4 | 0.98 | 0.96 |
| mean anchor hit@10 | 1.00 | 1.00 |
| mean prefix hit@4 | 1.00 | 1.00 |
| mean prefix hit@10 | 1.00 | 1.00 |
| steady-state latency (ms) | ~65 | ~70 |
| gate refusals | 4 | 3 |

Per-case anchor winner: **13 ties, 0 wins either way** (2 refusal cases have no anchors).

**Notable gate difference:** `followup_conflict_reference` ("Which one was wrong?") — semantic gate refuses, hybrid gate passes. BM25 catches keyword overlap that semantic embedding distance alone misses on vague follow-ups. In the live pipeline, the rewrite step resolves this for both methods, so it is not a real answer-quality gap — but it demonstrates that BM25 provides a slightly more permissive first-pass gate on referential queries.

**Answer-level** (full pipeline, rerank=False, 5-case clean sample — accepted as final):

| method | passed | grounded |
|---|---|---|
| semantic | 5/5 | 5 |
| hybrid | 5/5 | 5 |

Parity was confirmed on a clean 5-case sample (5/5 PASS, 5/5 grounded for **both** methods). The full 15-case answer-level run (30 `ask()` calls, ~90 sequential LLM calls: rewrite/answer/grounding per case) was attempted **three times**, each a few hours apart, and every attempt was blocked by Groq free-tier rate limits. Even with proactive throttling (~3s spacing, ~20 RPM ceiling) and fallback retry, the API returned "busy right now" on essentially every non-refusal case. This is a **structural free-tier limit** (~30 RPM ceiling cannot sustain ~90 sequential calls), not a transient deployment issue. The answer-level comparison is documented for future re-run should the API tier change, via `python scripts/compare_hybrid_vs_semantic.py --with-answers` (proactive throttle + fallback retry are built into the script).

### Recommendation

**Keep semantic-only as the default; hybrid is available as an optional path but does not justify a default switch on the current corpus.**

Rationale:
1. Retrieval quality is statistically indistinguishable — the demo corpus is small (7 chunks) and semantically diverse enough that MiniLM embeddings already retrieve all relevant content at top-10. This was consistently observed across the retrieval-level evaluation (15 cases, all ties on anchor coverage) and the accepted 5-case answer-level sample (5/5 pass, 5/5 grounded for both).
2. Latency is equivalent (~65-70 ms steady-state for both retrieval methods).
3. BM25 hybrid adds complexity (index building, caching, RRF fusion, a hybrid relevance gate) with no measurable answer-quality gain on this corpus. The one gate-behavior difference (hybrid lets through a vague follow-up that semantic refuses) is mitigated by the existing query-rewrite step.
4. Hybrid may become valuable as the corpus grows with documents where semantic similarity fails to capture exact terms (e.g., code, acronyms, part numbers) — the toggle is already wired for future re-evaluation. To re-run: `python scripts/compare_hybrid_vs_semantic.py --with-answers`.

The comparison script (`scripts/compare_hybrid_vs_semantic.py`) and the final detailed report (`eval/hybrid_vs_semantic_report.md`) are kept for future re-evaluation as the corpus evolves.

---

## Clarifying-Question Flow (Phase 11 Task 1)

When a user asks a question that is too vague to retrieve or answer meaningfully (e.g., "tell me more", "what about that?", pronoun-only follow-ups with no resolvable referent), the chatbot now responds with a **clarifying question** instead of guessing or retrieving blindly.

### Design

- **`clarify.py`**: Lightweight heuristic detection — no ML model. A question is flagged as vague when it contains zero content tokens (stopwords and discourse terms filtered out) **and** the conversation history does not provide enough concrete context to resolve the referent.
- **Short-circuit**: Runs before query rewriting and retrieval in `prepare_ask()`. If clarification is required, the pipeline returns a `PreparedAsk` with `clarification_required=True` and a templated message, skipping all LLM and retrieval calls.
- **Conservative bias**: Only triggers on clearly substance-free queries. A short-but-clear question like "what is ATP?" has a concrete subject and passes through. A history-resolvable follow-up like "what about that?" after a substantive assistant reply also passes through.

### Test Coverage

| test | scenario | expected |
|---|---|---|
| `test_vague_question_triggers_clarification` | "tell me more", "what about that?" with no history | triggers clarification |
| `test_short_but_clear_question_no_clarification` | "What is photosynthesis?" | no clarification |
| `test_followup_resolvable_from_history_no_clarification` | "what about that?" after a rich assistant answer | no clarification |
| `test_prepare_ask_returns_clarification_for_vague_question` | prepare_ask integration | clarification_required=True |
| `test_ask_returns_clarifying_message_without_llm` | full ask() pipeline | needed_clarification=True, correct message |
| `test_ask_clarification_returns_clarifying_message` | /ask endpoint mock | CLARIFICATION_MESSAGE in response |
| `test_ask_stream_clarification_metadata_done` | /ask/stream endpoint mock | metadata with is_clarification=True |
