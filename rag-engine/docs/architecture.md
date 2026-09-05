
## Retrieval Strategy (Phase 10)

Our retrieval pipeline uses a **semantic-first approach** with **LLM re-ranking**.

1.  **Semantic Search:** We use `all-MiniLM-L6-v2` to embed questions and retrieve the top 10 candidates from ChromaDB. Our diagnostic tests (Phase 10 Part B) confirmed that semantic search is already highly effective at capturing exact terms, acronyms, and numbers, making a keyword-based hybrid search unnecessary for our current scope.
2.  **LLM Re-ranking:** We use an LLM call to pick the top 3-4 most relevant chunks from the 10 candidates. We recently upgraded this prompt to be "conflict-aware," instructing the LLM to prioritize chunks that cover different aspects of a question or reveal contradictions, rather than just picking the most semantically similar ones. This helps address the "conflict unreported" pattern identified in our feedback loop.

### Rationale for choosing Re-ranking over Hybrid Search
We considered adding keyword-based hybrid search to catch "missed" exact terms. However, our diagnostic tests showed that semantic search was already retrieving those terms correctly. The primary failures we observed (negation being ignored, conflicts being missed) were issues of **LLM comprehension and heuristic scope**, not initial retrieval gaps. Enhancing the LLM re-ranking step to be more "aware" of multi-source contexts was deemed a higher-impact improvement.
