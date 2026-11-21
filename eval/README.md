# Evaluation harness

The dissertation's evaluation code, published as-is from the private repo. The product side (the two ranking systems, the enrichment pipeline, the Streamlit study app) stays private; the methodology is all here.

| Module | What it does | Standalone? |
|---|---|---|
| `metrics.py` | Retrieval metrics (P@K, MRR, MAP), embedding intrinsics, transition-model and playlist-coherence metrics | yes |
| `schemas.py` | The Pydantic data contract: `DJSetRequest` and the Literal-typed track-enrichment fields. Two-layer validation (typed fields plus a two-retry post-validator) held a 97.2% pass rate across the corpus | yes |
| `ab_comparison.py` | The A/B harness. Each eval query is parsed once by Gemini and the parsed `DJSetRequest` goes to both systems, so the ranking function is the only variable | needs the private scorers |
| `comprehensive_eval_v3.py` | Fixture suite across participant libraries: MAP@10, MRR, P@10, top-K overlap, genre coherence, PR curves | needs the private scorers |
| `baselines.py` | Random, popularity, and BPM-window baselines pushed through the identical metric path | needs the private scorers |
| `query_encoder_generalization_eval.py` | Held-out template generalization test for the query encoder | needs the private training code |
| `user_study_analysis.py` | Study aggregation: paired preferences, confidence weighting, bootstrap CIs for the coverage correlation | yes, given study data |
| `mpd_comprehensive_audit.py` | Audit of the MPD-derived training corpus | yes |
| `visualizer.py` | Generates the evaluation figures | yes |

Two notes:

- The fixture files (eval queries, participant libraries, study records) are excluded because they contain participant data covered by the study's ethics protocol (QMERC DSEECS25.063). The modules marked as needing them won't run outside the private repo; the code is what's being shown here.
- Everything in this directory is MIT licensed (see `LICENSE` in this folder). The documents in the rest of the repo remain all rights reserved.
