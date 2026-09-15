# Team Mu RAG — Evaluation Report

Total cases: 33

| Status | Case ID | Detail |
|--------|---------|--------|
| PASS | factual_thylakoid | ok | rounds=1 grounded=True |
| PASS | factual_stroma | ok | rounds=1 grounded=True |
| PASS | multi_hop_compare | ok | rounds=2 grounded=True |
| PASS | conflict_darkness | ok | rounds=1 grounded=True |
| PASS | injection_defense | ok | rounds=1 grounded=True |
| PASS | refusal_offtopic | ok | rounds=1 grounded=None |
| PASS | followup_rewrite | ok | rounds=1 grounded=True |
| PASS | oxygen_byproduct | ok | rounds=1 grounded=True |
| PASS | youtube_atp_density | ok | rounds=1 grounded=True |
| PASS | phytoplankton_oxygen | ok | rounds=1 grounded=True |
| PASS | multi_hop_photorespiration_vs_atp | ok | rounds=2 grounded=True |
| PASS | refusal_unrelated_topic | ok | rounds=1 grounded=None |
| PASS | faithfulness_chlorophyll_check | ok | rounds=1 grounded=True |
| PASS | followup_conflict_reference | ok | rounds=1 grounded=True |
| PASS | injection_direct_attempt | ok | rounds=1 grounded=True |
| PASS | clarify_tell_me_more | ok | rounds=0 grounded=None |
| PASS | clarify_what_about_it | ok | rounds=0 grounded=None |
| PASS | clarify_other_one | ok | rounds=0 grounded=None |
| FAIL | clarify_vague_followup | unexpected refusal; expected needed_clarification=True, got False | rounds=1 grounded=None |
| FAIL | clarify_single_word | expected needed_clarification=True, got False | rounds=1 grounded=True |
| PASS | refusal_capital_of_france | ok | rounds=1 grounded=None |
| PASS | refusal_python_help | ok | rounds=1 grounded=None |
| PASS | refusal_unrelated_science | ok | rounds=1 grounded=None |
| PASS | refusal_medical_advice | ok | rounds=1 grounded=None |
| PASS | refusal_recipe | ok | rounds=1 grounded=None |
| PASS | multi_hop_atp_source | ok | rounds=1 grounded=True |
| PASS | multi_hop_oxygen_origin | ok | rounds=1 grounded=True |
| PASS | multi_hop_stages_overview | ok | rounds=1 grounded=True |
| PASS | factual_thylakoid_rephrased | ok | rounds=1 grounded=True |
| PASS | factual_stroma_rephrased | ok | rounds=1 grounded=True |
| PASS | oxygen_byproduct_rephrased | ok | rounds=1 grounded=True |
| PASS | youtube_atp_density_rephrased | ok | rounds=1 grounded=True |
| PASS | phytoplankton_oxygen_rephrased | ok | rounds=1 grounded=True |

**Score: 31/33 (threshold 7)**
