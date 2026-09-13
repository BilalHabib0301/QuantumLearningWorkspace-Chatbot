"""
Compare semantic-only vs hybrid (semantic + BM25, RRF-fused) retrieval on the
shared eval question set (eval/cases.json).

Two levels, so the retrieval comparison is not masked by answer quality:
  1. Retrieval-level (no LLM): hit@k against the case's expected source
     prefixes and expected answer-anchor terms, plus steady-state latency and
     whether the relevance gate would refuse.
  2. Answer-level (full pipeline, same rerank/generation for both methods):
     eval pass/fail, grounded flag, total latency.

Usage (from chatbot/rag-engine):
  python scripts/compare_hybrid_vs_semantic.py
  python scripts/compare_hybrid_vs_semantic.py --retrieval-only
  python scripts/compare_hybrid_vs_semantic.py --rerank-answers true

Writes a markdown report to eval/hybrid_vs_semantic_report.md (override with
--report). Exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb").setLevel(logging.ERROR)

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from eval.eval_suite import _contains_any  # noqa: E402
from hybrid_search import get_bm25_index, hybrid_retrieve, is_hybrid_relevant  # noqa: E402
from rag_service import (  # noqa: E402
    REFUSAL_MESSAGE,
    ask,
    create_engine,
    rewrite_question,
)
from vector_store import is_relevant, retrieve  # noqa: E402

CASES_PATH = RAG_ENGINE_DIR / "eval" / "cases.json"
DEFAULT_REPORT = RAG_ENGINE_DIR / "eval" / "hybrid_vs_semantic_report.md"
RETRIEVAL_POOL = 10
METHODS = ("semantic", "hybrid")


def _norm(text: str) -> str:
    return (text or "").lower().replace(",", "").replace("  ", " ")


def _norm_needles(needles):
    return [_norm(n) for n in (needles or []) if n.strip()]


def _prefix_hit_fraction(ids, prefixes, k):
    if not prefixes:
        return None
    top = set((ids or [])[:k])
    return sum(1 for p in prefixes if any(i.startswith(p) for i in top)) / len(prefixes)


def _anchor_hit_fraction(docs, needles, k):
    norm_needles = _norm_needles(needles)
    if not norm_needles:
        return None
    top_docs = [_norm(d) for d in (docs or [])[:k]]
    found = sum(1 for n in norm_needles if any(n in d for d in top_docs))
    return found / len(norm_needles)


def effective_query(case, client):
    """Standalone query used for retrieval on both methods (rewrite is shared)."""
    history = case.get("history") or []
    if history and client is not None:
        try:
            return rewrite_question(client, history, case["question"])
        except Exception:
            return case["question"]
    return case["question"]


def run_retrieval(engine, method, query):
    cold_build_ms = None
    if method == "hybrid":
        t0 = time.perf_counter()
        get_bm25_index(engine)
        cold_build_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        results = hybrid_retrieve(engine, query, n_results=RETRIEVAL_POOL)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    else:
        t0 = time.perf_counter()
        results = retrieve(engine.collection, engine.embedding_model, query, n_results=RETRIEVAL_POOL)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    return results, latency_ms, cold_build_ms


def retrieval_metrics(results, expect, max_distance):
    prefixes = expect.get("source_id_prefixes_any") or []
    anchors = expect.get("must_contain_any") or []
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    return {
        "ids": ids,
        "prefixes": prefixes,
        "anchors": anchors,
        "k1_prefix": _prefix_hit_fraction(ids, prefixes, 1),
        "k4_prefix": _prefix_hit_fraction(ids, prefixes, 4),
        "k10_prefix": _prefix_hit_fraction(ids, prefixes, 10),
        "k1_anchor": _anchor_hit_fraction(docs, anchors, 1),
        "k4_anchor": _anchor_hit_fraction(docs, anchors, 4),
        "k10_anchor": _anchor_hit_fraction(docs, anchors, 10),
        # Would the first-round relevance gate (same logic as prepare_ask) refuse?
        "gate_refuses": (
            not is_hybrid_relevant(results, max_distance)
            if "rrf_scores" in results
            else not is_relevant(results.get("distances"), max_distance=max_distance)
        ),
    }


MAX_RETRIES = 3
BASE_BACKOFF_S = 8.0

def run_answer_case(engine, case, method, rerank, sleep_s=0.0):
    expect = case.get("expect") or {}
    started = time.perf_counter()
    if sleep_s:
        time.sleep(sleep_s)

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = ask(
                engine,
                case["question"],
                history=case.get("history"),
                top_k=case.get("top_k"),
                include_sources=True,
                rerank=rerank,
                multi_hop=case.get("multi_hop"),
                retrieval_method=method,
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BASE_BACKOFF_S * (2 ** attempt)
                print(f"    (rate-limited, retry {attempt+1}/{MAX_RETRIES} in {wait:.0f}s...)")
                time.sleep(wait)
    if last_exc is not None:
        return {"passed": False, "error": str(last_exc), "latency_ms": None, "grounded": None}
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    answer = result.answer or ""
    reasons = []
    must_refuse = bool(expect.get("must_refuse"))
    if must_refuse:
        if not result.refused and answer.strip() != REFUSAL_MESSAGE:
            reasons.append("expected refusal")
    else:
        if result.refused:
            reasons.append("unexpected refusal")

    if expect.get("must_contain_any") and not must_refuse:
        if not _contains_any(answer, expect["must_contain_any"]):
            reasons.append(f"answer missing any of {expect['must_contain_any']}")

    for bad in expect.get("forbidden_answer") or []:
        if bad.lower() in answer.lower():
            reasons.append(f"forbidden text {bad!r} in answer")

    min_rounds = expect.get("min_rounds")
    if min_rounds is not None and result.retrieval_rounds < int(min_rounds):
        reasons.append(f"retrieval_rounds={result.retrieval_rounds} < {min_rounds}")

    prefixes = expect.get("source_id_prefixes_any")
    if prefixes:
        ids = result.source_ids or []
        missing = [p for p in prefixes if not any(i.startswith(p) for i in ids)]
        if len(missing) == len(prefixes):
            reasons.append(f"no source_ids matching {prefixes}")
        elif len(prefixes) > 1 and missing:
            reasons.append(f"missing source prefixes {missing}")
        if expect.get("must_be_grounded") is True and result.grounded is False:
            reasons.append("expected grounded=true")

    rewritten_needles = expect.get("rewritten_must_contain_any")
    if rewritten_needles:
        if not _contains_any(result.rewritten_question or "", rewritten_needles):
            reasons.append(f"rewritten_question missing any of {rewritten_needles}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "latency_ms": elapsed_ms,
        "grounded": result.grounded,
        "refused": result.refused,
        "needed_clarification": result.needed_clarification,
    }


def _mean(values):
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def build_aggregates(rows, label):
    """rows: list of retrieval metric dicts for one method."""
    return {
        "label": label,
        "k4_prefix": _mean([r["k4_prefix"] for r in rows]),
        "k10_prefix": _mean([r["k10_prefix"] for r in rows]),
        "k4_anchor": _mean([r["k4_anchor"] for r in rows]),
        "k10_anchor": _mean([r["k10_anchor"] for r in rows]),
        "latency_ms": _mean([r["latency_ms"] for r in rows]),
        "gate_refusals": sum(1 for r in rows if r["gate_refuses"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="A/B semantic vs hybrid retrieval")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--collection", default="study_chunks_phase11")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0, help="cap cases (0 = all)")
    parser.add_argument("--with-answers", dest="with_answers", action="store_true")
    parser.add_argument("--no-answers", dest="with_answers", action="store_false")
    parser.set_defaults(with_answers=bool(os.environ.get("GROQ_API_KEY", "").strip()))
    parser.add_argument("--rerank-answers", dest="rerank_answers", action="store_true")
    parser.set_defaults(rerank_answers=False)
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep between answer runs")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    print(f"Loading engine ({args.collection}) for {len(cases)} cases...")
    engine = create_engine(collection_name=args.collection, seed_demo_data=True)
    print(f"Indexed {engine.chunks_indexed} chunks\n")

    client = None
    if args.with_answers:
        from rag_service import _get_groq

        client = _get_groq(engine)

    rows = {m: [] for m in METHODS}
    cold_build_ms = None
    printed_cold = False

    for case in cases:
        case_id = case.get("id", "?")
        query = effective_query(case, client)
        expect = case.get("expect") or {}
        for method in METHODS:
            results, latency_ms, build_ms = run_retrieval(engine, method, query)
            if build_ms is not None and not printed_cold:
                cold_build_ms = build_ms
                printed_cold = True
            entry = retrieval_metrics(results, expect, engine.max_distance)
            entry["case_id"] = case_id
            entry["query"] = query
            entry["latency_ms"] = latency_ms
            rows[method].append(entry)

    sem = build_aggregates(rows["semantic"], "semantic")
    hyb = build_aggregates(rows["hybrid"], "hybrid")

    print("\n=== RETRIEVAL-LEVEL (candidate pool = top-10, no LLM) ===")
    print(
        f"{'':10}{'k4 anch':>8}{'k10 anch':>9}{'k4 pref':>8}{'k10 pref':>9}"
        f"{'lat(ms)':>9}{'refusals':>9}"
    )
    for agg, label in ((sem, "semantic"), (hyb, "hybrid")):
        print(
            f"{label:10}"
            f"{_fmt(agg['k4_anchor']):>8}{_fmt(agg['k10_anchor']):>9}"
            f"{_fmt(agg['k4_prefix']):>8}{_fmt(agg['k10_prefix']):>9}"
            f"{_fmt(agg['latency_ms']):>9}{agg['gate_refusals']:>9}"
        )
    if cold_build_ms is not None:
        print(f"\nHybrid first-call includes BM25 index build: {cold_build_ms:.1f} ms")

    # Per-case winner on top-10 anchor coverage
    wins = {"semantic": 0, "hybrid": 0, "tie": 0}
    detailed = []
    for sem_row, hyb_row in zip(rows["semantic"], rows["hybrid"]):
        a = sem_row["k10_anchor"]
        b = hyb_row["k10_anchor"]
        case_id = sem_row["case_id"]
        if a is None or b is None:
            detailed.append((case_id, sem_row, hyb_row, "N/A"))
            continue
        if b > a:
            wins["hybrid"] += 1
            detailed.append((case_id, sem_row, hyb_row, "hybrid"))
        elif a > b:
            wins["semantic"] += 1
            detailed.append((case_id, sem_row, hyb_row, "semantic"))
        else:
            wins["tie"] += 1
            detailed.append((case_id, sem_row, hyb_row, "tie"))
    print(f"\nTop-10 anchor coverage winner (per case): {wins}")

    print("\nGate refusal agreement (must_refuse / normal cases):")
    for case, sem_row, hyb_row in zip(cases, rows["semantic"], rows["hybrid"]):
        intended_refusal = bool((case.get("expect") or {}).get("must_refuse"))
        a, b = sem_row["gate_refuses"], hyb_row["gate_refuses"]
        verdict = "OK" if a == b else "DIFF"
        print(
            f"  [{verdict}] {case.get('id'):28}"
            f" must_refuse={str(int(intended_refusal)):>1}"
            f" semantic_refuses={str(int(a))}  hybrid_refuses={str(int(b))}"
        )

    # Answer-level
    answer_rows = None
    if args.with_answers:
        print("\n=== ANSWER-LEVEL (full pipeline, rerank_answers=%s) ===" % args.rerank_answers)
        answer_rows = {m: [] for m in METHODS}
        eval_cases = cases if args.with_answers else []
        for method in METHODS:
            passed = 0
            for case in eval_cases:
                outcome = run_answer_case(engine, case, method, args.rerank_answers, args.sleep)
                outcome["case_id"] = case.get("id", "?")
                answer_rows[method].append(outcome)
                if outcome["passed"]:
                    passed += 1
                tag = "PASS" if outcome["passed"] else "FAIL"
                note = "; ".join(outcome.get("reasons") or [])
                err = " (error)" if outcome.get("error") else ""
                print(f"  [{tag}] {method:8} {case.get('id')}: {note}{err}")
            total = len(eval_cases)
            grounded = sum(1 for o in answer_rows[method] if o.get("grounded"))
            avg = _mean([o["latency_ms"] for o in answer_rows[method] if o["latency_ms"] is not None])
            print(f"  {method}: {passed}/{total} passed | avg total latency = {_fmt(avg)} ms | grounded={grounded}")

    write_report(args.report, cases, rows, sem, hyb, wins, cold_build_ms, answer_rows, args)
    print(f"\nReport saved to {args.report}")
    return 0


def _fmt(v):
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return f"{v:>6.2f}"
    return f"{v:>6}"


def write_report(path, cases, rows, sem, hyb, wins, cold_build_ms, answer_rows, args):
    lines = [
        "# Hybrid vs Semantic Retrieval — A/B Report",
        "",
        f"Generated: full pipeline comparison on `{len(cases)}` shared eval cases "
        "(`eval/cases.json`).",
        f"Retrieval method config: `RETRIEVAL_METHOD` + `hybrid_retrieve()` "
        "(semantic + BM25 via reciprocal rank fusion).",
        f"Answer-level re-rank flag: `{args.rerank_answers}`.",
        "",
        "## Retrieval-level metrics (top-10 candidate pool, no LLM)",
        "",
        "| metric | semantic | hybrid |",
        "|---|---|---|",
        f"| mean anchor hit@4 | {_fmt(sem['k4_anchor'])} | {_fmt(hyb['k4_anchor'])} |",
        f"| mean anchor hit@10 | {_fmt(sem['k10_anchor'])} | {_fmt(hyb['k10_anchor'])} |",
        f"| mean prefix hit@4 | {_fmt(sem['k4_prefix'])} | {_fmt(hyb['k4_prefix'])} |",
        f"| mean prefix hit@10 | {_fmt(sem['k10_prefix'])} | {_fmt(hyb['k10_prefix'])} |",
        f"| mean steady-state retrieval latency (ms) | {_fmt(sem['latency_ms'])} | {_fmt(hyb['latency_ms'])} |",
        f"| gate refusals | {sem['gate_refusals']} | {hyb['gate_refusals']} |",
    ]
    if cold_build_ms is not None:
        lines.append(
            f"\nHybrid first-call overhead includes BM25 index build: `{cold_build_ms:.1f} ms` "
            "(cached afterwards)."
        )

    lines += ["", "Per-case top-10 anchor-coverage winner:", ""]
    lines.append("| case | semantic k10 anchor | hybrid k10 anchor | winner |")
    lines.append("|---|---|---|---|")
    for case, sem_row, hyb_row in zip(cases, rows["semantic"], rows["hybrid"]):
        a, b = sem_row["k10_anchor"], hyb_row["k10_anchor"]
        if a is None or b is None:
            w = "n/a (no anchors)"
        elif b > a:
            w = "hybrid"
        elif a > b:
            w = "semantic"
        else:
            w = "tie"
        lines.append(
            f"| {case.get('id')} | {_fmt(a)} | {_fmt(b)} | {w} |"
        )

    lines += ["", "Gate refusal agreement:", ""]
    lines.append("| case | expected | semantic | hybrid |")
    lines.append("|---|---|---|---|")
    for case, sem_row, hyb_row in zip(cases, rows["semantic"], rows["hybrid"]):
        expected = "refuse" if (case.get("expect") or {}).get("must_refuse") else "answer"
        a = "refuse" if sem_row["gate_refuses"] else "answer"
        b = "refuse" if hyb_row["gate_refuses"] else "answer"
        lines.append(f"| {case.get('id')} | {expected} | {a} | {b} |")

    if answer_rows is not None:
        lines += ["", "## Answer-level (full pipeline)", ""]
        lines.append("| method | passed | grounded | avg total (ms) |")
        lines.append("|---|---|---|---|")
        for method in ("semantic", "hybrid"):
            outcomes = answer_rows[method]
            passed = sum(1 for o in outcomes if o["passed"])
            grounded = sum(1 for o in outcomes if o.get("grounded"))
            avg = _mean([o["latency_ms"] for o in outcomes if o["latency_ms"] is not None])
            lines.append(
                f"| {method} | {passed}/{len(outcomes)} | {grounded} | {_fmt(avg)} |"
            )
        lines += ["", "Per-case answer results:", ""]
        lines.append("| case | semantic | hybrid |")
        lines.append("|---|---|---|")
        for case, s_out, h_out in zip(cases, answer_rows["semantic"], answer_rows["hybrid"]):
            s = "PASS" if s_out["passed"] else "FAIL"
            h = "PASS" if h_out["passed"] else "FAIL"
            s_extra = "; ".join(s_out.get("reasons") or []) or (s_out.get("error") or "")
            h_extra = "; ".join(h_out.get("reasons") or []) or (h_out.get("error") or "")
            lines.append(
                f"| {case.get('id')} | {s} {('(' + s_extra + ')') if s_extra else ''} "
                f"| {h} {('(' + h_extra + ')') if h_extra else ''} |"
            )

    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())