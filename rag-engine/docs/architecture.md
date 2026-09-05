
## Retrieval Strategy (Phase 10)

Our retrieval pipeline uses a **semantic-first approach** with **LLM re-ranking**.

### 1. Semantic Search
We use `all-MiniLM-L6-v2` to embed questions and retrieve the top 10 candidates from ChromaDB. Our diagnostic tests (Phase 10 Part B) confirmed that semantic search is already highly effective at capturing exact terms, acronyms, and numbers — even with rephrased queries (see `scripts/diagnose_retrieval_v2.py`). This made keyword-based hybrid search unnecessary for our current scope.

### 2. LLM Re-ranking
We use an LLM call to pick the top 3-4 most relevant chunks from the 10 candidates. The re-ranking prompt instructs the LLM to prioritize chunks that cover different aspects of a question or reveal contradictions.

**Important finding from multi-run testing (5 runs x 2 prompts):** The Groq model (GPT-OSS-120B) already includes both conflicting chunks in 100% of runs even with the simpler "relevance only" prompt for our test scenario. This means the "conflict-aware" prompt is a **safety reinforcement**, not a fix for a reproducible bug. Its value lies in:
- Guarding against potential regression if the base model changes
- Providing explicit guidance for more ambiguous or subtle conflict scenarios
- Documenting architectural intent clearly in the prompt itself

### Rationale for choosing Re-ranking over Hybrid Search
We considered adding keyword-based hybrid search to catch "missed" exact terms. However, our diagnostic tests showed that semantic search was already retrieving those terms correctly. The primary failures we observed (negation being ignored, conflicts being missed) were issues of **LLM comprehension and heuristic scope**, not initial retrieval gaps. Enhancing the LLM re-ranking step to be more "aware" of multi-source contexts was deemed a higher-impact improvement — primarily as a safeguard and explicit instruction layer.
