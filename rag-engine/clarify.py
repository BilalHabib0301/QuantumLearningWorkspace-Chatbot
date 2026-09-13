"""Lightweight vague-question detection (Phase 11 Task 1 clarifying-question flow).

No ML: pure heuristics. A question with no concrete content words is treated as
vague — unless the conversation history provides enough substance to resolve a
referential follow-up (e.g. "what about that?" right after a real answer).

Intentional bias: only trigger on clearly substance-free queries so we never
interrupt legitimately terse but answerable questions ("what is X").
"""

from __future__ import annotations

import re

from bm25 import STOPWORDS, _TOKEN_RE

# Minimum number of distinct informative tokens the conversation context must
# contain before a vague follow-up is considered resolvable.
CONTEXT_MIN_INFORMATIVE = 2


def informative_tokens(text: str) -> list[str]:
    """Content tokens after removing stopwords / near-content-free terms."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS]


def has_resolvable_context(
    history: list[dict] | None,
    min_informative: int = CONTEXT_MIN_INFORMATIVE,
) -> bool:
    """Does the conversation contain enough concrete content to resolve a vague referent?"""
    if not history:
        return False
    context = " ".join((m.get("content") or "").strip() for m in history)
    return len(set(informative_tokens(context))) >= min_informative


def needs_clarification(question: str, history: list[dict] | None) -> bool:
    """
    Return True when the query is too vague to answer and shouldn't be guessed
    at. Resolvable context in history suppresses clarification.
    """
    cleaned = (question or "").strip()
    if not cleaned:
        return False
    if informative_tokens(cleaned):
        return False
    return not has_resolvable_context(history)