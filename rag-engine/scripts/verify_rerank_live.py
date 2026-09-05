"""
Side-by-side comparison of the old vs. new rerank prompt using a live Groq call.
Demonstrates the 'conflict-aware' improvement (Phase 10, Part B).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import load_env, _get_groq, GROQ_MODEL, _call_groq_safe

load_env()

# 1. Simulated Conflict Candidates
# (Simulating the retrieval results before the final reranking step)
CANDIDATES = """
[temp_25_chunk_0]
Photosynthesis temperature is 25C.

[temp_40_chunk_0]
Photosynthesis temperature is 40C.

[general_chunk_0]
Photosynthesis is the process by which plants make food.
"""
QUESTION = "What is the exact temperature for the reaction?"
TOP_K = 2

OLD_PROMPT = (
    "You re-rank retrieved document chunks for a search query. "
    f"Return ONLY a comma-separated list of the best {TOP_K} chunk ids "
    "in order of relevance (most relevant first). "
    "Use the exact ids shown in brackets. No other text."
)

NEW_PROMPT = (
    "You re-rank retrieved document chunks for a search query. "
    f"Return ONLY a comma-separated list of the best {TOP_K} chunk ids "
    "in order of relevance (most relevant first). "
    "CRITICAL: Prioritize chunks that cover DIFFERENT aspects of the question "
    "or reveal CONTRADICTIONS between sources. If multiple chunks provide "
    "conflicting facts about the same topic, include them all so the "
    "final answer can report the disagreement. "
    "Use the exact ids shown in brackets. No other text."
)

def run_live_rerank(name, prompt_content):
    print(f"\n=== {name} ===")
    print(f"Question: {QUESTION}\nCandidates: {CANDIDATES}")
    
    messages = [
        {"role": "system", "content": prompt_content},
        {"role": "user", "content": f"Query: {QUESTION}\n\nCandidates:{CANDIDATES}\n\nBest chunk ids:"},
    ]
    
    # Live Groq call (mocked if no key available)
    try:
        client = _get_groq(type("Engine", (), {"_groq": None, "auto_reconnect": False}))
        response = _call_groq_safe(client, model=GROQ_MODEL, messages=messages, temperature=0.0)
        result = response.choices[0].message.content or ""
        print(f"LLM Selected IDs: {result.strip()}")
        
        # Check if it included both conflicting chunks
        if "temp_25" in result and "temp_40" in result:
            print("SUCCESS: Successfully identified and preserved both conflicting sources.")
        else:
            print("FAIL: Failed to preserve both conflicting sources (selected only one or none).")
        return result
    except RuntimeError as e:
        print(f"⚠️ Skipped live call (Groq key not found): {e}")
        return None

if __name__ == "__main__":
    run_live_rerank("OLD PROMPT (Relevance Only)", OLD_PROMPT)
    run_live_rerank("NEW PROMPT (Conflict-Aware)", NEW_PROMPT)
