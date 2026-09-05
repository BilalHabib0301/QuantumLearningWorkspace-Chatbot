"""Multi-run comparison of old vs new rerank prompt for conflict handling.

Runs each prompt 5 times against the real Groq API and reports the pass rate
(i.e. did the LLM include BOTH conflicting chunks in its top-K selection).

Usage:
    rag_venv\Scripts\python.exe rag-engine\scripts\verify_rerank_reliability.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import load_env, _get_groq, GROQ_MODEL, _call_groq_safe

load_env()

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
NUM_RUNS = 5

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


def run_one(prompt_label: str, prompt_content: str, run_num: int) -> bool:
    """Run a single rerank call. Returns True if both conflicting chunks selected."""
    messages = [
        {"role": "system", "content": prompt_content},
        {"role": "user", "content": f"Query: {QUESTION}\n\nCandidates:{CANDIDATES}\n\nBest chunk ids:"},
    ]
    try:
        client = _get_groq(type("Engine", (), {"_groq": None, "auto_reconnect": False}))
        response = _call_groq_safe(client, model=GROQ_MODEL, messages=messages, temperature=0.0)
        result = response.choices[0].message.content or ""
        has_both = "temp_25" in result and "temp_40" in result
        print(f"  Run {run_num}: {result.strip()}")
        return has_both
    except RuntimeError as e:
        print(f"  Run {run_num}: ERROR - {e}")
        return False


def main():
    print(f"Question: {QUESTION}")
    print(f"Runs per prompt: {NUM_RUNS}\n")

    # --- Old prompt ---
    print(f"--- OLD PROMPT ({NUM_RUNS} runs) ---")
    old_pass = 0
    for i in range(1, NUM_RUNS + 1):
        if run_one("OLD", OLD_PROMPT, i):
            old_pass += 1
            print(f"    -> PASS (both chunks)")
        else:
            print(f"    -> FAIL (missing a chunk)")
    print(f"  Old prompt pass rate: {old_pass}/{NUM_RUNS}\n")

    # --- New prompt ---
    print(f"--- NEW PROMPT ({NUM_RUNS} runs) ---")
    new_pass = 0
    for i in range(1, NUM_RUNS + 1):
        if run_one("NEW", NEW_PROMPT, i):
            new_pass += 1
            print(f"    -> PASS (both chunks)")
        else:
            print(f"    -> FAIL (missing a chunk)")
    print(f"  New prompt pass rate: {new_pass}/{NUM_RUNS}\n")

    # --- Summary ---
    print("=" * 50)
    print(f"OLD prompt: {old_pass}/{NUM_RUNS} passed")
    print(f"NEW prompt: {new_pass}/{NUM_RUNS} passed")
    if new_pass > old_pass:
        print("CONCLUSION: New prompt improved reliability.")
    elif new_pass == old_pass:
        print("CONCLUSION: Both prompts equally reliable for this scenario.")
    else:
        print("CONCLUSION: Old prompt was more reliable (unexpected).")


if __name__ == "__main__":
    main()
