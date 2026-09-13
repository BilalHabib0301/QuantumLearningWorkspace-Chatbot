# Hybrid vs Semantic Retrieval — A/B Report

Generated: full pipeline comparison on `15` shared eval cases (`eval/cases.json`).
Retrieval method config: `RETRIEVAL_METHOD` + `hybrid_retrieve()` (semantic + BM25 via reciprocal rank fusion).
Answer-level re-rank flag: `False`.

## Retrieval-level metrics (top-10 candidate pool, no LLM)

| metric | semantic | hybrid |
|---|---|---|
| mean anchor hit@4 |   0.98 |   0.96 |
| mean anchor hit@10 |   1.00 |   1.00 |
| mean prefix hit@4 |   1.00 |   1.00 |
| mean prefix hit@10 |   1.00 |   1.00 |
| mean steady-state retrieval latency (ms) |  64.39 |  66.71 |
| gate refusals | 2 | 2 |

Hybrid first-call overhead includes BM25 index build: `13.4 ms` (cached afterwards).

Per-case top-10 anchor-coverage winner:

| case | semantic k10 anchor | hybrid k10 anchor | winner |
|---|---|---|---|
| factual_thylakoid |   1.00 |   1.00 | tie |
| factual_stroma |   1.00 |   1.00 | tie |
| multi_hop_compare |   1.00 |   1.00 | tie |
| conflict_darkness |   1.00 |   1.00 | tie |
| injection_defense |   1.00 |   1.00 | tie |
| refusal_offtopic |   -   |   -   | n/a (no anchors) |
| followup_rewrite |   1.00 |   1.00 | tie |
| oxygen_byproduct |   1.00 |   1.00 | tie |
| youtube_atp_density |   1.00 |   1.00 | tie |
| phytoplankton_oxygen |   1.00 |   1.00 | tie |
| multi_hop_photorespiration_vs_atp |   1.00 |   1.00 | tie |
| refusal_unrelated_topic |   -   |   -   | n/a (no anchors) |
| faithfulness_chlorophyll_check |   1.00 |   1.00 | tie |
| followup_conflict_reference |   1.00 |   1.00 | tie |
| injection_direct_attempt |   1.00 |   1.00 | tie |

Gate refusal agreement:

| case | expected | semantic | hybrid |
|---|---|---|---|
| factual_thylakoid | answer | answer | answer |
| factual_stroma | answer | answer | answer |
| multi_hop_compare | answer | answer | answer |
| conflict_darkness | answer | answer | answer |
| injection_defense | answer | answer | answer |
| refusal_offtopic | refuse | refuse | refuse |
| followup_rewrite | answer | answer | answer |
| oxygen_byproduct | answer | answer | answer |
| youtube_atp_density | answer | answer | answer |
| phytoplankton_oxygen | answer | answer | answer |
| multi_hop_photorespiration_vs_atp | answer | answer | answer |
| refusal_unrelated_topic | refuse | refuse | refuse |
| faithfulness_chlorophyll_check | answer | answer | answer |
| followup_conflict_reference | answer | answer | answer |
| injection_direct_attempt | answer | answer | answer |

## Answer-level (full pipeline)

| method | passed | grounded | avg total (ms) |
|---|---|---|---|
| semantic | 2/15 | 0 | 1072.12 |
| hybrid | 2/15 | 0 | 1077.98 |

Per-case answer results:

| case | semantic | hybrid |
|---|---|---|
| factual_thylakoid | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| factual_stroma | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| multi_hop_compare | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| conflict_darkness | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| injection_defense | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| refusal_offtopic | PASS  | PASS  |
| followup_rewrite | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| oxygen_byproduct | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| youtube_atp_density | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| phytoplankton_oxygen | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| multi_hop_photorespiration_vs_atp | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| refusal_unrelated_topic | PASS  | PASS  |
| faithfulness_chlorophyll_check | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| followup_conflict_reference | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
| injection_direct_attempt | FAIL (The AI service is busy right now. Please try again shortly.) | FAIL (The AI service is busy right now. Please try again shortly.) |
