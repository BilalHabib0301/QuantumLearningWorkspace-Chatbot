"""
Targeted regression run: 18 cases (15 original Phase 7 + 3 multi-hop).
Calls ask() directly with delays to avoid Groq rate limits.
Reports per-case pass/fail + needed_clarification check.
"""
import json
import sys
import time
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import REFUSAL_MESSAGE, ask, create_engine

CASES_PATH = RAG_ENGINE_DIR / "eval" / "cases_regression.json"
DELAY = 3  # seconds between calls


def contains_any(text, needles):
    if not needles:
        return True
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    print(f"Loading engine for {len(cases)} cases...")
    engine = create_engine(collection_name="study_chunks_eval", seed_demo_data=True)
    print(f"Indexed {engine.chunks_indexed} chunks\n")

    passed = 0
    failed = 0
    results = []

    for i, case in enumerate(cases):
        cid = case["id"]
        expect = case.get("expect") or {}

        # Skip clarification cases (already tested separately)
        if expect.get("must_ask_clarification") is not None:
            print(f"[SKIP] {cid}: clarification case, tested separately")
            results.append((cid, "SKIP", "clarification case"))
            continue

        if i > 0:
            time.sleep(DELAY)

        try:
            result = ask(
                engine,
                case["question"],
                history=case.get("history"),
                top_k=case.get("top_k"),
                include_sources=True,
                rerank=case.get("rerank"),
                multi_hop=case.get("multi_hop"),
            )
        except RuntimeError as e:
            print(f"[FAIL] {cid}: RuntimeError: {e}")
            failed += 1
            results.append((cid, "FAIL", f"RuntimeError: {e}"))
            continue

        reasons = []
        answer = result.answer or ""

        # Check must_refuse
        must_refuse = bool(expect.get("must_refuse"))
        if must_refuse:
            if not result.refused and answer.strip() != REFUSAL_MESSAGE:
                reasons.append("expected refusal")
        else:
            if result.refused:
                reasons.append("unexpected refusal")

        # Check must_contain_any
        if expect.get("must_contain_any") and not must_refuse:
            if not contains_any(answer, expect["must_contain_any"]):
                reasons.append(f"answer missing any of {expect['must_contain_any']}")

        # Check forbidden_answer
        for bad in expect.get("forbidden_answer") or []:
            if bad.lower() in answer.lower():
                reasons.append(f"forbidden text {bad!r}")

        # Check min_rounds
        min_rounds = expect.get("min_rounds")
        if min_rounds is not None and result.retrieval_rounds < int(min_rounds):
            reasons.append(f"rounds={result.retrieval_rounds} < {min_rounds}")

        # Check source prefixes
        prefixes = expect.get("source_id_prefixes_any")
        if prefixes:
            ids = result.source_ids or []
            missing = [p for p in prefixes if not any(i.startswith(p) for i in ids)]
            if len(missing) == len(prefixes):
                reasons.append(f"no source_ids matching {prefixes}")

        # Check grounded
        if expect.get("must_be_grounded") is True and result.grounded is False:
            reasons.append(f"expected grounded=true, got {result.grounded}")

        # Check rewritten_must_contain_any
        rewritten_needles = expect.get("rewritten_must_contain_any")
        if rewritten_needles:
            if not contains_any(result.rewritten_question or "", rewritten_needles):
                reasons.append(f"rewritten missing {rewritten_needles}")

        # Check needed_clarification (should be False for all these)
        if result.needed_clarification:
            reasons.append("unexpected needed_clarification=True")

        status = "PASS" if not reasons else "FAIL"
        detail = "; ".join(reasons) if reasons else "ok"
        print(f"[{status}] {cid}: {detail}")
        if not reasons:
            passed += 1
        else:
            failed += 1
        results.append((cid, status, detail))

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed")
    print(f"{'='*60}")
    for cid, status, detail in results:
        marker = "PASS" if status == "PASS" else ("SKIP" if status == "SKIP" else "FAIL")
        print(f"  [{marker}] {cid}: {detail}")


if __name__ == "__main__":
    main()
