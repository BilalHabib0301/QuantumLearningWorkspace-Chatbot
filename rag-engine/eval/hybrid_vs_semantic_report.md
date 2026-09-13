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
| mean steady-state retrieval latency (ms) |  64.77 |  69.56 |
| gate refusals | 4 | 3 |

Hybrid first-call overhead includes BM25 index build: `14.1 ms` (cached afterwards).

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
| followup_rewrite | answer | refuse | refuse |
| oxygen_byproduct | answer | answer | answer |
| youtube_atp_density | answer | answer | answer |
| phytoplankton_oxygen | answer | answer | answer |
| multi_hop_photorespiration_vs_atp | answer | answer | answer |
| refusal_unrelated_topic | refuse | refuse | refuse |
| faithfulness_chlorophyll_check | answer | answer | answer |
| followup_conflict_reference | answer | refuse | answer |
| injection_direct_attempt | answer | answer | answer |
