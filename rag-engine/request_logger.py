"""Structured per-request logging for /ask and /ask/stream.

Writes one JSON object per line (JSONL/NDJSON) to a local file so each
request is easy to parse and analyze later (Phase 10 Part A). Reuses
TimingRecord.to_dict() for the retrieval/LLM durations instead of
duplicating timing logic.

PII note: user_ids are SHA-256 hashed before being written to disk so a
raw email address never touches the log file. The question text itself
is deliberately NOT logged — only its length.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from timing_logger import TimingRecord

RAG_ENGINE_DIR = Path(__file__).resolve().parent
LOG_DIR = RAG_ENGINE_DIR / "logs"
LOG_PATH = Path(os.environ.get("REQUEST_LOG_PATH", str(LOG_DIR / "requests.jsonl")))


def set_log_path(path: str | Path) -> None:
    """Override the JSONL log file location (mainly for tests)."""
    global LOG_PATH
    LOG_PATH = Path(path)


def _anonymize_user(user_id: str | None) -> str | None:
    """Hash user_id to keep raw PII (email addresses) out of the log."""
    if not user_id:
        return None
    return hashlib.sha256(user_id.strip().encode("utf-8")).hexdigest()[:16]


def log_request(
    *,
    endpoint: str,
    user_id: str | None,
    question: str,
    timing: TimingRecord | None,
    grounded: bool | None,
    cached: bool = False,
) -> list[str]:
    """
    Append one JSONL entry for a completed /ask or /ask/stream request.

    Returns the list of lines that were written (useful for tests).
    """
    timing_dict = timing.to_dict() if timing is not None else {}
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "user_id": _anonymize_user(user_id),
        "question_length": len((question or "").strip()),
        "retrieval_ms": timing_dict.get("retrieval_ms"),
        "llm_ms": timing_dict.get("llm_ms"),
        "grounded": grounded,
        "cached": cached,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return [line.strip()]