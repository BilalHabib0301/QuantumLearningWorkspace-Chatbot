"""In-memory feedback store keyed by response_id (Phase 10 Part A).

/ask snapshots each completed response (question, answer, sources) into
this store keyed by a fresh response_id, which the frontend echoes back
to /feedback. Kept in memory (no disk writes) to match the existing
in-memory cache/rate-limiter infra and to avoid persisting question and
answer text. The user_id is SHA-256 hashed via request_logger._anonymize_user
so the record never stores a raw email.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from request_logger import _anonymize_user

Rating = Literal["up", "down"]


@dataclass
class FeedbackRecord:
    response_id: str
    user_id: str | None
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    rated_at: str | None = None
    rating: Rating | None = None

    def to_dict(self) -> dict:
        return {
            "response_id": self.response_id,
            "user_id": self.user_id,
            "question": self.question,
            "answer": self.answer,
            "sources": list(self.sources),
            "created_at": self.created_at,
            "rated_at": self.rated_at,
            "rating": self.rating,
        }


def record_from_response(
    *,
    response_id: str,
    user_id: str | None,
    question: str,
    answer: str,
    source_ids: list[str],
) -> FeedbackRecord:
    """Snapshot an /ask response so a later /feedback can attach a rating."""
    return FeedbackRecord(
        response_id=response_id,
        user_id=_anonymize_user(user_id),
        question=(question or "").strip(),
        answer=answer or "",
        sources=list(source_ids or []),
    )


class FeedbackStore:
    """In-memory map of response_id -> FeedbackRecord."""

    def __init__(self) -> None:
        self._records: dict[str, FeedbackRecord] = {}

    def put(self, record: FeedbackRecord) -> None:
        self._records[record.response_id] = record

    def has(self, response_id: str) -> bool:
        return response_id in self._records

    def get(self, response_id: str) -> FeedbackRecord | None:
        return self._records.get(response_id)

    def submit(self, response_id: str, rating: Rating) -> FeedbackRecord | None:
        """Attach a rating if the response_id is known; None if unknown."""
        record = self._records.get(response_id)
        if record is None:
            return None
        record.rating = rating
        record.rated_at = datetime.now(timezone.utc).isoformat()
        return record

    def all(self) -> list[FeedbackRecord]:
        return list(self._records.values())

    def rated(self) -> list[FeedbackRecord]:
        return [r for r in self._records.values() if r.rating is not None]

    def by_rating(self, rating: Rating) -> list[FeedbackRecord]:
        return [r for r in self._records.values() if r.rating == rating]

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()


feedback_store = FeedbackStore()