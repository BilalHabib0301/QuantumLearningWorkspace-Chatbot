"""Okapi BM25 lexical retrieval (pure Python, no external dependencies).

Predates the Phase 10 decision where keyword search was rejected because the
semantic path already recovered exact terms. Phase 11 Task 1 revives it as an
*alternative retrieval method* fused with semantic search (see hybrid_search.py)
so we can A/B it against the semantic-only path.
"""

from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small English stopword set. Aggressive removal is fine here: the demo corpus
# is technical notes, and BM25 is only one side of a hybrid fuse.
STOPWORDS: frozenset[str] = frozenset(
    """
    a and are as at be been but by can could did do does down for from had has
    have having he her here hers herself him his how i if in into is it its
    itself just let me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves
    """.split()
)

# Near-content-free terms for study questions: question words, pronouns,
# demonstratives, discourse fillers. Filtering these is what lets the language
# clarify flow (clarify.py) detect pronoun-only follow-ups.
NON_INFORMATIVE: frozenset[str] = frozenset(
    """
    about above after again all also am any anyone anything as at because before
    being below between both but come could did do does doing down each else few
    for from further get got had has have having he her here herself him his i
    in into is it its itself just let more most my myself no nor not now of off
    on one only or other our out over same she should so some such than that the
    their them there these they this through to too under until up very we well
    what whatever when where while who whom whose why will you your yours
    tell say said ask asking know think mean says please ok okay oh yes yeah
    thing things stuff something anything someone somebody
    """.split()
)
STOPWORDS = STOPWORDS | NON_INFORMATIVE


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, dropping stopwords."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS]


class BM25Index:
    """Static BM25 index over a fixed document set (no incremental updates)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._terms: list[list[str]] = []
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_dl = 0.0
        self._num_docs = 0
        self._doc_freq: dict[str, int] = {}

    @classmethod
    def build(
        cls,
        documents: list[tuple[str, str]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "BM25Index":
        index = cls(k1=k1, b=b)
        index._num_docs = len(documents)
        for doc_id, text in documents:
            terms = tokenize(text)
            index._ids.append(doc_id)
            index._terms.append(terms)
            index._doc_lengths.append(len(terms))
            freqs: dict[str, int] = {}
            for term in terms:
                freqs[term] = freqs.get(term, 0) + 1
            index._term_freqs.append(freqs)
            for term in set(terms):
                index._doc_freq[term] = index._doc_freq.get(term, 0) + 1
        index._avg_dl = (
            sum(index._doc_lengths) / index._num_docs if index._num_docs else 0.0
        )
        return index

    @property
    def size(self) -> int:
        return self._num_docs

    def _idf(self, term: str) -> float:
        n = self._doc_freq.get(term, 0)
        return math.log(1.0 + (self._num_docs - n + 0.5) / (n + 0.5))

    def score(self, query: str) -> dict[str, float]:
        """Return {doc_id: bm25_score} for every document sharing a query term."""
        if self._num_docs == 0 or self._avg_dl <= 0:
            return {}
        q_terms = tokenize(query)
        if not q_terms:
            return {}
        scores: dict[str, float] = {}
        k1, b = self.k1, self.b
        for term in set(q_terms):
            idf = self._idf(term)
            for i, freqs in enumerate(self._term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * (1.0 - b + b * (self._doc_lengths[i] / self._avg_dl))
                scores[self._ids[i]] = scores.get(self._ids[i], 0.0) + idf * (
                    tf * (k1 + 1.0)
                ) / denom
        return scores

    def top_ids(self, query: str, k: int) -> list[str]:
        """Top-k doc ids by BM25 score (most relevant first)."""
        ranked = sorted(self.score(query).items(), key=lambda kv: kv[1], reverse=True)
        return [cid for cid, _score in ranked[:k]]