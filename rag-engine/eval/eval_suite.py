"""
Regression eval suite for Team Mu RAG (Phase 6 stretch).

Usage (from chatbot/rag-engine):
  python eval/eval_suite.py
  python eval/eval_suite.py --threshold 7

Exits non-zero if passed cases < threshold (default 7/10).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAG_ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import REFUSAL_MESSAGE, ask, create_engine  # noqa: E402

CASES_PATH = Path(__file__).resolve().parent / "cases.json"
DEFAULT_THRESHOLD = 7


def _contains_any(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def evaluate_case(engine, case: dict) -> tuple[bool, str]:
    expect = case.get("expect") or {}
    result = ask(
        engine,
        case["question"],
        history=case.get("history"),
        top_k=case.get("top_k"),
        include_sources=True,
        rerank=case.get("rerank"),
        multi_hop=case.get("multi_hop"),
    )
    answer = result.answer or ""
    reasons: list[str] = []

    must_refuse = bool(expect.get("must_refuse"))
    if must_refuse:
        if not result.refused and answer.strip() != REFUSAL_MESSAGE:
            reasons.append("expected refusal")
    else:
        if result.refused:
            reasons.append("unexpected refusal")

    if expect.get("must_contain_any") and not must_refuse:
        if not _contains_any(answer, expect["must_contain_any"]):
            reasons.append(
                f"answer missing any of {expect['must_contain_any']}"
            )

    for bad in expect.get("forbidden_answer") or []:
        if bad.lower() in answer.lower():
            reasons.append(f"forbidden text {bad!r} in answer")

    min_rounds = expect.get("min_rounds")
    if min_rounds is not None and result.retrieval_rounds < int(min_rounds):
        reasons.append(
            f"retrieval_rounds={result.retrieval_rounds} < {min_rounds}"
        )

    prefixes = expect.get("source_id_prefixes_any")
    if prefixes:
        ids = result.source_ids or []
        if not any(any(i.startswith(p) for i in ids) for p in prefixes):
            # Prefer seeing at least one of each when both listed — soft check:
            # require that for each prefix, some id matches IF multiple prefixes
            missing = [p for p in prefixes if not any(i.startswith(p) for i in ids)]
            if len(missing) == len(prefixes):
                reasons.append(f"no source_ids matching {prefixes}")
            elif len(prefixes) > 1 and missing:
                reasons.append(f"missing source prefixes {missing}")

        if expect.get("must_be_grounded") is True and result.grounded is False:
            reasons.append(
            f"expected grounded=true (faithfulness check) but got grounded={result.grounded}"
        )

    rewritten_needles = expect.get("rewritten_must_contain_any")
    if rewritten_needles:
        if not _contains_any(result.rewritten_question or "", rewritten_needles):
            reasons.append(
                f"rewritten_question missing any of {rewritten_needles}"
            )

    ok = len(reasons) == 0
    detail = "; ".join(reasons) if reasons else "ok"
    return ok, f"{case.get('id')}: {detail} | rounds={result.retrieval_rounds} grounded={result.grounded}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Team Mu RAG eval suite")
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum passing cases (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="Path to cases.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_report.md",
        help="Where to write the markdown report",
    )
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    print(f"Loading engine for {len(cases)} cases...")
    engine = create_engine(collection_name="study_chunks_eval", seed_demo_data=True)
    print(f"Indexed {engine.chunks_indexed} chunks\n")

    passed = 0
    report_lines = [
        "# Team Mu RAG — Evaluation Report",
        "",
        f"Total cases: {len(cases)}",
        "",
        "| Status | Case ID | Detail |",
        "|--------|---------|--------|",
    ]

    for case in cases:
        ok, detail = evaluate_case(engine, case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {detail}")
        if ok:
            passed += 1
        case_id, _, rest = detail.partition(":")
        report_lines.append(f"| {status} | {case_id.strip()} | {rest.strip()} |")

    total = len(cases)
    print(f"\nScore: {passed}/{total} (threshold {args.threshold})")

    report_lines.append("")
    report_lines.append(f"**Score: {passed}/{total} (threshold {args.threshold})**")

    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report saved to {args.report}")

    if passed < args.threshold:
        print("Below threshold.")
        return 1
    print("Above threshold.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
