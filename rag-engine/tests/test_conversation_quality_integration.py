"""Live-server integration tests: long-conversation quality (Phase 11 Part C).

Two opt-in, real-LLM experiments against a real uvicorn server (no mocks):

* BASELINE  (server started with ENABLE_HISTORY_COMPACTION=false): proves the
  HISTORY_TURN_CAP=4 hard-drop means a fact introduced turn 1 is NOT recallable
  from turn 8+ — degradation confirmed-by-design, measured end to end.
* COMPACTION (server started with ENABLE_HISTORY_COMPACTION=true, the default):
  proves the same 12-turn conversation now recalls the turn-1 token via the
  earlier-conversation summary block.

Also exercises POST /conversations/summarize against the live server (incl.
cache hit on repeat) and records per-turn latency/grounding so the observed
behavior is documented, not just asserted.

Run (PowerShell):
    $env:RUN_LIVE_CONVERSATION_TESTS="1"
    python -m pytest rag-engine/tests/test_conversation_quality_integration.py -v

Requirements: GROQ_API_KEY + JWT_SECRET_KEY (loaded from rag-engine/.env),
network access for Groq and (on first run) HF model download.
Expect several minutes: 12 turns x 2 experiments, ~3 LLM calls/turn (rewrite,
answer, grounding), paced to respect Groq free-tier rate limits.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from dotenv import load_dotenv

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import HISTORY_TURN_CAP  # noqa: E402

load_dotenv(RAG_ENGINE_DIR / ".env")

USER_EMAIL = "phase11-conversation-quality@example.com"
TEST_JWT_SECRET = "phase11-cq-test-secret-change-me"
PROBE_TOKEN = "TOKEN-7741"
PACE_SECONDS = float(os.environ.get("CQ_PACE_SECONDS", "5"))
MAX_TURNS = 12
FINDINGS_PATH = RAG_ENGINE_DIR / "logs" / "conversation_quality_live.json"

_required = os.environ.get("RUN_LIVE_CONVERSATION_TESTS") == "1" and bool(
    os.environ.get("GROQ_API_KEY")
    and os.environ.get("JWT_SECRET_KEY")
)

pytestmark = pytest.mark.skipif(
    not _required,
    reason=(
        "requires RUN_LIVE_CONVERSATION_TESTS=1 plus GROQ_API_KEY and "
        "JWT_SECRET_KEY in env (real LLM + live server)"
    ),
)

# Turn script: 1-indexed. Mixes topic switches, follow-ups, and references
# back to early turns. The private code in turn 1 appears NOWHERE in the
# seeded documents, so it is recallable only from conversation context.
TURN_QUESTIONS = [
    "To keep track of us: my private study code is TOKEN-7741. Now, where "
    "exactly do the light-dependent reactions of photosynthesis take place?",
    "So in the thylakoid: what mechanically drives ATP production during the "
    "light reactions, according to the energy lecture?",
    "Pigment question: which wavelengths does chlorophyll absorb best and why "
    "do leaves look green?",
    "Switch to the Calvin cycle: where does it run, and what is the role of "
    "the enzyme RuBisCO?",
    "And in the energy lecture, what concrete number was cited for ATP "
    "synthase density in the grana?",
    "Which stage releases oxygen, and what happens during photolysis at "
    "Photosystem II?",
    "Switch topic: roughly what share of Earth's photosynthetic oxygen comes "
    "from marine phytoplankton, per the lecture?",
    "Now recall: what exact private study code did I give you at the very "
    "start of this conversation? Reply with the code only.",
    "Switch again: what is the difference between where the light-dependent "
    "reactions and the Calvin cycle occur?",
    "Last follow-up course: what was the code from the first message? Reply "
    "with the code only.",
    "How do C4 and CAM plants deal with photorespiration in hot, arid "
    "environments?",
    "Final retention check: please repeat the exact study code from the "
    "first message verbatim.",
]


def _free_port() -> int:
    """Grab an ephemeral port, then release it for uvicorn to bind."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _jwt_token() -> str:
    import jwt

    return jwt.encode({"sub": USER_EMAIL}, TEST_JWT_SECRET, algorithm="HS256")


def _seed_user_docs(chroma_path: Path) -> int:
    """Index demo docs into an isolated Chroma store for the test user."""
    from rag_service import add_user_document, create_engine
    from vector_store import DEFAULT_COLLECTION_NAME

    old = os.environ.get("CHROMA_DB_PATH")
    os.environ["CHROMA_DB_PATH"] = str(chroma_path)
    try:
        engine = create_engine(
            collection_name=DEFAULT_COLLECTION_NAME, seed_demo_data=False
        )
        added = 0
        for name in ("photosynthesis_overview.txt", "youtube_lecture_energy.txt"):
            added += add_user_document(
                engine, RAG_ENGINE_DIR / "data" / name, user_id=USER_EMAIL
            )
        return added
    finally:
        if old is None:
            os.environ.pop("CHROMA_DB_PATH", None)
        else:
            os.environ["CHROMA_DB_PATH"] = old


def _wait_ready(base_url: str, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("ready") is True:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"server at {base_url} did not become ready in {timeout}s")


def _post_ask(base_url: str, *, question: str, history: list[dict]) -> dict:
    headers = {"Authorization": f"Bearer {_jwt_token()}"}
    body = {
        "question": question,
        "history": history or None,
        "skip_cache": True,
        "rerank": False,
        "multi_hop": False,
        "top_k": 4,
        "include_sources": True,
    }
    for _ in range(15):
        try:
            resp = requests.post(f"{base_url}/ask", json=body, headers=headers, timeout=240)
        except requests.ConnectionError:
            time.sleep(10)
            continue
        if resp.status_code in (429, 503):
            time.sleep(25)  # Groq free-tier busy window
            continue
        if resp.status_code != 200:
            time.sleep(5)
            continue
        return resp.json()
    raise RuntimeError(f"/ask never succeeded for question: {question[:60]!r}")


def _run_conversation(base_url: str) -> list[dict]:
    """Run the 12-turn script, returning per-turn observations."""
    history: list[dict] = []
    turns: list[dict] = []
    for idx, question in enumerate(TURN_QUESTIONS, start=1):
        data = _post_ask(base_url, question=question, history=history)
        answer = (data.get("answer") or "").strip()
        timing = data.get("timing") or {}
        turn = {
            "turn": idx,
            "question": question,
            "answer": answer,
            "refused": bool(data.get("refused")),
            "no_documents": bool(data.get("no_documents")),
            "grounded": data.get("grounded"),
            "retrieval_rounds": data.get("retrieval_rounds"),
            "total_ms": timing.get("total_ms"),
            "llm_ms": timing.get("llm_ms"),
            "token_in_answer": PROBE_TOKEN in answer,
        }
        turns.append(turn)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if idx < MAX_TURNS:
            time.sleep(PACE_SECONDS)
    return turns


def _summarize(base_url: str, history: list[dict]) -> dict:
    headers = {"Authorization": f"Bearer {_jwt_token()}"}
    resp = requests.post(
        f"{base_url}/conversations/summarize",
        json={"history": history},
        headers=headers,
        timeout=240,
    )
    assert resp.status_code == 200, f"summarize failed: {resp.status_code} {resp.text}"
    return resp.json()


@pytest.fixture(scope="module", params=[False, True], ids=["baseline-drop", "compaction-on"])
def live_server(request) -> SimpleNamespace:
    compaction_on: bool = request.param
    tmp_root = Path(tempfile.mkdtemp(prefix="rag_cq_"))
    chroma_dir = tmp_root / "chroma"
    seeded = _seed_user_docs(chroma_dir)
    port = _free_port()

    env = dict(os.environ)
    env.update(
        {
            "CHROMA_DB_PATH": str(chroma_dir),
            "JWT_SECRET_KEY": TEST_JWT_SECRET,
            "RATE_LIMIT_MAX": "1000",
            "RATE_LIMIT_WINDOW_SECONDS": "60",
            "ENABLE_HISTORY_COMPACTION": str(compaction_on).lower(),
            "REQUEST_LOG_PATH": str(tmp_root / "requests.jsonl"),
        }
    )
    out = (tmp_root / "server.out.log").open("w")
    err = (tmp_root / "server.err.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=RAG_ENGINE_DIR,
        env=env,
        stdout=out,
        stderr=err,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(base_url)
        turns = _run_conversation(base_url)
        ns = SimpleNamespace(
            compaction_on=compaction_on,
            base_url=base_url,
            tmp_root=tmp_root,
            seeded=seeded,
            turns=turns,
        )
        yield ns
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        out.close()
        err.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


def _probe_turns(turns: list[dict]) -> list[dict]:
    return [t for t in turns if PROBE_TOKEN in t["question"]]


def _write_findings(summary: dict, overwrite: bool = True) -> None:
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FINDINGS_PATH.exists() and not overwrite:
        data = json.loads(FINDINGS_PATH.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update(summary)
    FINDINGS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def test_turn1_recall_depends_on_compaction(live_server):
    """The load-bearing assertion: baseline drops the turn-1 token, compaction keeps it."""
    probes = _probe_turns(live_server.turns)
    assert probes, "expected probe turns asking for the turn-1 token"
    got = [p["token_in_answer"] for p in probes]
    if live_server.compaction_on:
        assert all(got), (
            "ENABLE_HISTORY_COMPACTION=true should recall the turn-1 token; "
            f"probe turns {[p['turn'] for p in probes]} returned it only in "
            f"{sum(got)}/{len(got)}. Answers: "
            + "; ".join(f"t{p['turn']}={p['answer'][:80]!r}" for p in probes)
        )
    else:
        assert not any(got), (
            "baseline (no compaction) must NOT recall a turn dropped by the "
            f"HISTORY_TURN_CAP={HISTORY_TURN_CAP} hard cap; probe turns "
            f"{[p['turn'] for p in probes]} unexpectedly contained the token: "
            + "; ".join(f"t{p['turn']}={p['answer'][:80]!r}" for p in probes)
        )


def test_latency_and_grounding_recorded(live_server):
    """Record, and sanity-check, per-turn latency/grounding (documented in findings)."""
    timed = [t for t in live_server.turns if t["total_ms"] is not None]
    assert timed, "expected per-turn timing from /ask responses"
    late = [t for t in live_server.turns if t["turn"] >= HISTORY_TURN_CAP + 2]
    for t in late:
        assert t["total_ms"] is not None and t["total_ms"] <= 120_000, (
            f"turn {t['turn']} runaway latency: {t['total_ms']}ms"
        )

    _write_findings(
        {
            f"experiment_{'compaction' if live_server.compaction_on else 'baseline'}": {
                "turns": live_server.turns,
                "seeded_chunks": live_server.seeded,
                "history_turn_cap": HISTORY_TURN_CAP,
                "probe_recall_rate": sum(
                    p["token_in_answer"] for p in _probe_turns(live_server.turns)
                )
                / len(_probe_turns(live_server.turns))
                if _probe_turns(live_server.turns)
                else None,
                "grounded_true_rate": sum(
                    1 for t in live_server.turns if t["grounded"] is True
                )
                / len(live_server.turns),
                "mean_total_ms": round(
                    sum(t["total_ms"] or 0 for t in timed) / len(timed), 1
                ),
                "max_total_ms": max(t["total_ms"] or 0 for t in timed),
            }
        }
    )


def test_summarize_endpoint_live_and_cache(live_server):
    """Live /conversations/summarize: real summary, cache hit on repeat."""
    first = _summarize(live_server.base_url, [
        {"role": "user", "content": t["question"]}
        for t in live_server.turns
    ] + [
        {"role": "assistant", "content": t["answer"]}
        for t in live_server.turns
    ])
    assert first["summary"].strip(), "expected a non-empty summary"
    assert first["cached"] is False

    second = _summarize(live_server.base_url, [
        {"role": "user", "content": t["question"]}
        for t in live_server.turns
    ] + [
        {"role": "assistant", "content": t["answer"]}
        for t in live_server.turns
    ])
    assert second["summary"] == first["summary"]
    assert second["cached"] is True

    # Token savings: full raw history vs. the cached summary (real LLM output).
    full_chars = sum(len(t["question"]) + len(t["answer"]) for t in live_server.turns)
    full_est = max(1, full_chars // 4)
    summary_est = max(1, len(first["summary"]) // 4)
    assert summary_est < full_est, (
        f"summary ({summary_est} est. tokens) should beat full history ({full_est})"
    )
    assert first["turn_count"] == MAX_TURNS


def test_compaction_token_savings_vs_full_history(live_server):
    """Live token estimate: compacted context < shipping the whole history."""
    full = [
        {"role": "user", "content": t["question"]}
        for t in live_server.turns
    ] + [
        {"role": "assistant", "content": t["answer"]}
        for t in live_server.turns
    ]
    # Reproduce what the server would send in the compacted path: a summary of
    # the older turns + the last HISTORY_TURN_CAP turns verbatim.
    recent = full[-(HISTORY_TURN_CAP * 2):]
    early = full[:-(HISTORY_TURN_CAP * 2)]
    summary = _summarize(live_server.base_url, early)
    summary_text = summary["summary"]
    compacted = [{"role": "user", "content": summary_text}] + recent

    def est_tokens(messages: list[dict]) -> int:
        text = " ".join(m.get("content") or "" for m in messages)
        return max(1, len(text) // 4)

    full_cost = est_tokens(full)
    compacted_cost = est_tokens(compacted)
    assert compacted_cost < full_cost, (
        f"compacted {compacted_cost} est. tokens should beat full {full_cost}"
    )