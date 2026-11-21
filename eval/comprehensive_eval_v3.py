"""
Comprehensive evaluation suite for the dissertation (Task 6 of overnight run).

Runs A vs B across all 5 participant libraries × (3 anchor queries + 3
library-tailored queries) and emits MAP@10 / MRR / P@10 / top-K overlap /
genre coherence / coverage breakdown / PR-curve points.

"Ground truth" for retrieval metrics: we use the GeminiConsultant's expanded
artist/track recommendations as a SOFT ground-truth set per query (the
consultant runs once per query — its world-knowledge expansion is independent
of the local library). A returned track counts as "relevant" if its primary
artist appears in the consultant's top-30 expansion. This proxy is imperfect
but gives a consistent, reproducible signal across systems and libraries.

If the consultant returns nothing for a query (rare — happens for purely
descriptive queries with no concrete artist anchors), we fall back to
"top-K of System A" as the reference. That degenerate case is reported.

We also report cross-system top-K overlap (System A ∩ System B), which is
the cleanest dissertation-defensible signal.

If query encoder v3 / track projector v1 are not present locally, the run
auto-falls back to v2 + raw Artist2Vec — labelled as "v2-fallback" in the
output filenames so post-Colab-sync runs don't overwrite this baseline.

CLI:
    GEMINI_API_KEY=$(...) poetry run python -m src.evaluation.comprehensive_eval_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.crate_generator import CrateGenerator              # noqa: E402
from src.features.embedding_scorer import (                          # noqa: E402
    EmbeddingScorer, _swap_library_on,
    QUERY_ENCODER_V3_DIR, TRACK_PROJECTOR_V1_DIR,
)
from src.data.constants import GENRE_FAMILIES_LOWER                  # noqa: E402

PARTICIPANTS_DIR = ROOT / "output" / "user_studies" / "participants"
OUTPUT_DIR = ROOT / "output"
ENV_PATH = ROOT / ".env"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_v3")


# ── Anchor + tailored queries ────────────────────────────────────────────────
# Anchors: queries that every audited library (per output/library_audits/SUMMARY.md)
# has at least a *partial* match for. Driving techno is the only universal-strong;
# the other two are "festival EDM" (Dora has *weak* but it's still feasible) and
# "underground minimal tech house" (same — Dora *weak*, others strong).

ANCHOR_QUERIES = [
    "Driving techno from my library. include some techno leaning tech house as well",
    "1 hour underground minimal tech house set, deep cuts, no mainstream",
    "festival EDM big room set 124-130 bpm peak time",
]

# Tailored: 3 queries per participant. Drawn from data/evaluation_queries.json
# (which already has the user-study queries) plus the audit-suggested folder
# queries from output/library_audits/SUMMARY.md.

TAILORED_QUERIES: Dict[str, List[str]] = {
    "Adin Segall": [
        "2hr warm up deep house journey for terrace at club space. artists should not be considered mainstream and should not be beatport chart toppers. deep cuts smaller artists. This is an opening set for chaos in the CBD (do not include chaos in the cbd)",
        "10 house tracks (any subgenre of house) with no vocals",
        "Build a 90-minute set in the spirit of my 'HAUS' folder. Same energy and selection style, but find some adjacent picks too.",
    ],
    "Marcus Phillips": [
        "1 hour peak time underground rave set with minimal / ukg / breakbeat",
        "30 minute morning soulful groovy house mix",
        "30 track set with high energy ukg and breakbeat tracks meant for an underground dj set",
    ],
    "Colin Raab": [
        "1 hr deep house warm up for rooftop terrace",
        "i want a 20 track chill house playlist",
        "start with some house, then techno at a higher bpm, then finish with trance and happy hardcore. have a happy and upbeat mood",
    ],
    "Dora Segall": [
        "Lowkey dark playlist to play at a seated bar",
        "fully instrumental playlist",
        "deep cuts rave playlist",
    ],
    "Elliot Harper": [
        "1 hour-long house set for a late night club set",
        "18 track hard techno set for dark club energy",
        "10 track afro-house set from african only artists",
    ],
}

# Map participant name -> library parquet path
PARTICIPANT_PARQUET = {
    "Adin Segall": PARTICIPANTS_DIR / "adin_segall_587c3c55" / "enriched_library.parquet",
    "Marcus Phillips": PARTICIPANTS_DIR / "marcus_phillips_62664a02" / "enriched_library.parquet",
    "Colin Raab": PARTICIPANTS_DIR / "colin_raab_8ab3627e" / "enriched_library.parquet",
    "Dora Segall": PARTICIPANTS_DIR / "dora_segall_a341b014" / "enriched_library.parquet",
    "Elliot Harper": PARTICIPANTS_DIR / "elliot_harper_14cdec76" / "enriched_library.parquet",
}


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── Metrics ──────────────────────────────────────────────────────────────────

def _genre_relevance(track_genre: str, requested: List[str]) -> float:
    if not requested:
        return 1.0
    tg = (track_genre or "").lower().strip().replace("_", " ")
    if not tg:
        return 0.1
    best = 0.0
    for g in requested:
        gl = g.lower().strip()
        if not gl:
            continue
        if tg == gl:
            return 1.0
        family = set(x.lower() for x in GENRE_FAMILIES_LOWER.get(gl, []))
        if family and tg in family:
            best = max(best, 0.8)
        elif gl in tg or tg in gl:
            best = max(best, 0.5)
    return max(best, 0.1)


def _genre_coherence(tracks: List, requested_genres: List[str]) -> float:
    """Fraction of tracks whose gemini_micro_genre matches the parsed genres
    exactly or in family. Returns 0.0 if no tracks."""
    if not tracks:
        return 0.0
    if not requested_genres:
        return 1.0  # no genre constraint -> trivially coherent
    n_match = 0
    for t in tracks:
        rel = _genre_relevance(getattr(t, "genre", "") or "", requested_genres)
        if rel >= 0.8:
            n_match += 1
    return n_match / len(tracks)


def _bpm_stddev(tracks: List) -> float:
    bpms = [getattr(t, "bpm", None) for t in tracks]
    bpms = [b for b in bpms if b]
    if len(bpms) < 2:
        return 0.0
    return float(np.std(bpms))


def _consultant_relevance_set(consultant, query: str, library_df: pd.DataFrame, limit: int = 30) -> Set[str]:
    """
    Get the consultant's expanded artist set (top-`limit` recs + similar
    artists they pull in). Returns a lowercase set of primary artist tokens.
    """
    try:
        recs = consultant.get_recommendations(query, limit=limit)
    except Exception as e:
        log.warning(f"  consultant recs failed: {e}")
        return set()
    if not recs:
        return set()
    artists: Set[str] = set()
    for r in recs:
        a = r.get("artist") or r.get("artist_name") or ""
        a = str(a).strip().lower()
        if a:
            artists.add(a)
    return artists


def _track_artist_in_set(track, ref_artists: Set[str]) -> bool:
    a = (getattr(track, "artist", "") or "").strip().lower()
    if not a:
        return False
    if a in ref_artists:
        return True
    # primary-artist split
    for sep in [" feat", " ft", " &", " and ", ", ", " vs", " versus ", " b2b "]:
        if sep in a:
            head = a.split(sep)[0].strip()
            if head and head in ref_artists:
                return True
            break
    return False


def _retrieval_metrics(tracks: List, ref_artists: Set[str]) -> Dict[str, float]:
    """
    Compute MAP@10, MRR, P@10 against a reference artist set treated as
    relevant. With no ref set we return zeros — caller should handle that.

    METHODOLOGY NOTE (Stage 4.7, 2026-05-11):
    The AP@10 emitted here is renamed `consultant_AP@10` to distinguish it
    from the training-side `centroid_AP@10` used by the v4 encoder trainer.
    They measure different things:
      - consultant_AP@10: multi-positive (Gemini Consultant artist set,
        ~30 artists) against the POST-HARD-FILTER candidate pool
        (10s-100s of library tracks). Denominator is `n_rel_in_top10`,
        NOT min(k, R). Not recall-sensitive beyond top-10.
      - centroid_AP@10: single-positive (the gold tracklist centroid)
        against the FULL 170K-centroid corpus. Mathematically MRR@10 in
        the single-positive case.
    Both numbers are valid retrieval metrics within their own definition;
    they should never be compared to each other in the dissertation. See
    `docs/handover_2026_05_09_audits_diagnostics.md` §2.1.
    """
    if not tracks:
        return {"P@10": 0.0, "MRR": 0.0, "consultant_AP@10": 0.0, "n_relevant": 0}
    rels = [int(_track_artist_in_set(t, ref_artists)) for t in tracks[:10]]
    n_rel = sum(rels)
    if n_rel == 0:
        return {"P@10": 0.0, "MRR": 0.0, "consultant_AP@10": 0.0, "n_relevant": 0}
    # MRR: 1/rank of first relevant
    mrr = 0.0
    for i, r in enumerate(rels, 1):
        if r:
            mrr = 1.0 / i
            break
    # AP@10: average precision at 10 with single-pass relevance
    cum = 0
    ap = 0.0
    for i, r in enumerate(rels, 1):
        if r:
            cum += 1
            ap += cum / i
    ap = ap / n_rel
    p_at_10 = n_rel / min(10, len(tracks))
    return {"P@10": float(p_at_10), "MRR": float(mrr), "consultant_AP@10": float(ap), "n_relevant": int(n_rel)}


def _pr_curve(tracks: List, ref_artists: Set[str], total_relevant: Optional[int] = None) -> List[Tuple[float, float]]:
    """Return list of (recall, precision) points across rank-cutoffs 1..len(tracks).
    `total_relevant` defaults to ref_artists size (upper bound)."""
    if not tracks or not ref_artists:
        return []
    R = total_relevant if total_relevant is not None else max(1, len(ref_artists))
    rels = [int(_track_artist_in_set(t, ref_artists)) for t in tracks]
    cum = 0
    pts: List[Tuple[float, float]] = []
    for i, r in enumerate(rels, 1):
        cum += r
        pts.append((cum / R, cum / i))
    return pts


# ── Per-query runner ─────────────────────────────────────────────────────────

def run_one_query(
    cg: CrateGenerator,
    scorer: EmbeddingScorer,
    consultant,
    query: str,
    top_k: int,
    scoring_mode: str = "hybrid",
) -> Dict[str, Any]:
    request = cg.parser.parse_request(query)
    request.raw_query = query

    genres = []
    if request.vibe and request.vibe.genre_weights:
        weights = request.vibe.genre_weights
        n = len(weights)
        threshold = 0.05 if n >= 5 else (0.10 if n >= 3 else 0.20)
        genres = [g for g, w in weights.items() if w >= threshold]
        if not genres:
            genres = [max(weights.items(), key=lambda x: x[1])[0]]

    # System A
    a_crate, _ = cg.generate_crate(query, verbose=False, request=request.model_copy(deep=True))
    # Patch 2026-05-13: when System A's hard filter yields < 50% of target_tracks,
    # `generate_crate` returns {type: 'interaction_required', partial_tracks: [...]}
    # (see crate_generator.py:1700-1716). Real Streamlit users click "accept these
    # candidates" and System A delivers — so for the dissertation A/B we count
    # partial_tracks rather than scoring System A as A_ret=0.
    a_tracks_raw = a_crate.get("tracks") or a_crate.get("partial_tracks") or []
    a_tracks = a_tracks_raw[:top_k]
    a_interaction_required = (a_crate.get("type") == "interaction_required")

    # System B
    b_result, _ = scorer.score(
        query, top_k=top_k, verbose=False, request=request.model_copy(deep=True),
        scoring_mode=scoring_mode,
    )
    b_tracks = b_result.get("tracks", [])

    # Reference set from consultant (cached per query — no library dependence)
    ref_artists = _consultant_relevance_set(consultant, query, library_df=cg.tracks_df, limit=30)
    # Patch 2026-05-13: when the consultant fails (exception, NoneType, empty
    # JSON parse, or genuinely empty rec set) ref_artists is empty and the
    # AP/MRR/P metrics are forced to 0. Flag the row so downstream aggregation
    # excludes it — otherwise the unbiased AP numbers get diluted by queries
    # where the ranking quality is unmeasurable.
    consultant_unavailable = (len(ref_artists) == 0)

    a_ids = {str(getattr(t, "track_id", "")) for t in a_tracks}
    b_ids = {str(getattr(t, "track_id", "")) for t in b_tracks}
    overlap = a_ids & b_ids
    overlap_pct = len(overlap) / max(1, min(len(a_tracks), len(b_tracks)))

    a_metrics = _retrieval_metrics(a_tracks, ref_artists)
    b_metrics = _retrieval_metrics(b_tracks, ref_artists)
    a_pr = _pr_curve(a_tracks, ref_artists)
    b_pr = _pr_curve(b_tracks, ref_artists)

    a_genre_coh = _genre_coherence(a_tracks, genres)
    b_genre_coh = _genre_coherence(b_tracks, genres)
    a_bpm_std = _bpm_stddev(a_tracks)
    b_bpm_std = _bpm_stddev(b_tracks)

    return {
        "query": query,
        "parsed_genres": genres,
        "n_consultant_ref_artists": len(ref_artists),
        "consultant_unavailable": consultant_unavailable,
        "a_interaction_required": a_interaction_required,
        "scoring_mode": scoring_mode,
        "a_returned": len(a_tracks),
        "b_returned": len(b_tracks),
        "a_top_artists": [getattr(t, "artist", "") for t in a_tracks[:5]],
        "b_top_artists": [getattr(t, "artist", "") for t in b_tracks[:5]],
        "a_top_titles": [getattr(t, "name", "") for t in a_tracks[:5]],
        "b_top_titles": [getattr(t, "name", "") for t in b_tracks[:5]],
        "overlap_top_k": len(overlap),
        "overlap_pct": overlap_pct,
        "a_genre_coherence": a_genre_coh,
        "b_genre_coherence": b_genre_coh,
        "a_bpm_stddev": a_bpm_std,
        "b_bpm_stddev": b_bpm_std,
        "a_metrics": a_metrics,
        "b_metrics": b_metrics,
        "a_pr_curve": a_pr,
        "b_pr_curve": b_pr,
        "b_candidate_coverage": float(b_result.get("candidate_embedding_coverage", 0.0)),
        "b_coverage_direct": int(b_result.get("candidate_coverage_direct", 0)),
        "b_coverage_similar_artists": int(b_result.get("candidate_coverage_similar_artists", 0)),
        "b_coverage_track_projector": int(b_result.get("candidate_coverage_track_projector", 0)),
        "b_coverage_none": int(b_result.get("candidate_coverage_none", 0)),
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--participants", nargs="*", default=None,
                    help="Subset of participant names; default: all 5")
    ap.add_argument("--include-anchor-only", action="store_true",
                    help="Skip per-library tailored queries (fast smoke test)")
    # Stage 4.7 (2026-05-11): explicit v4/v3 model overrides.
    ap.add_argument("--encoder-dir", type=Path, default=None,
                    help="Override path to query encoder dir (default: auto-resolve to v3_final_infonce).")
    ap.add_argument("--projector-dir", type=Path, default=None,
                    help="Override path to track projector dir (default: auto-resolve to projector_v2).")
    # Patch 2026-05-13: default to 'hybrid' — the dissertation's HEADLINE A/B
    # ("given a heuristic-filtered pool, learned reranking outperforms heuristic
    # scoring"). 'soft_filter' is the legacy v2/v3 default and remains available.
    ap.add_argument("--scoring-mode", choices=["soft_filter", "hybrid"], default="hybrid",
                    help="System B scoring mode. 'hybrid' (default) = heuristic hard "
                         "filter + learned rerank (headline A/B). 'soft_filter' = legacy.")
    args = ap.parse_args()

    _load_env(ENV_PATH)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not in env or .env")
        sys.exit(1)

    use_v3 = (QUERY_ENCODER_V3_DIR / "query_encoder.pt").exists()
    use_proj = (TRACK_PROJECTOR_V1_DIR / "track_projector.pt").exists()
    label = "v3" if (use_v3 and use_proj) else ("v2-fallback" if not (use_v3 or use_proj) else "partial-v3")
    log.info(f"Run label: {label}  (encoder_v3={use_v3}, projector_v1={use_proj})")
    log.info(f"Scoring mode: {args.scoring_mode}  "
             f"(encoder_dir={args.encoder_dir or 'auto-resolve'}, projector_dir={args.projector_dir or 'auto-resolve'})")

    suffix = label
    results_csv = OUTPUT_DIR / f"evaluation_{suffix}_results.csv"
    summary_csv = OUTPUT_DIR / f"evaluation_{suffix}_summary.csv"
    pr_json = OUTPUT_DIR / f"evaluation_{suffix}_pr_curves.json"
    full_json = OUTPUT_DIR / f"evaluation_{suffix}_full.json"

    participants = args.participants or list(PARTICIPANT_PARQUET.keys())
    log.info(f"Participants: {participants}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CrateGenerator + EmbeddingScorer once; we swap the library between
    # participants. The Gemini Consultant lives on cg; we cache its recs
    # per query string so the same query across libraries reuses the call.
    log.info("Initializing CrateGenerator + EmbeddingScorer (default library)...")
    cg = CrateGenerator(gemini_api_key=api_key)
    consultant = cg.consultant
    cg_for_a = cg
    scorer = EmbeddingScorer(
        gemini_api_key=api_key,
        use_consultant=False,
        soft_genre_filter=True,
        crate_generator=cg,
        encoder_dir=args.encoder_dir,
        projector_dir=args.projector_dir,
    )

    all_rows: List[Dict] = []
    pr_curves_payload: Dict[str, Any] = {}
    full_payload: List[Dict] = []
    consultant_cache: Dict[str, Set[str]] = {}

    headline_queries = {  # save PR curves only for these (dissertation figs)
        "anchor:driving_techno": ANCHOR_QUERIES[0],
        "anchor:underground_minimal": ANCHOR_QUERIES[1],
        "anchor:festival_edm": ANCHOR_QUERIES[2],
        "adin:warm_up_terrace": "2hr warm up deep house journey for terrace at club space. artists should not be considered mainstream and should not be beatport chart toppers. deep cuts smaller artists. This is an opening set for chaos in the CBD (do not include chaos in the cbd)",
    }
    headline_inv = {v: k for k, v in headline_queries.items()}

    for p_name in participants:
        parquet = PARTICIPANT_PARQUET.get(p_name)
        if not parquet or not parquet.exists():
            log.warning(f"  no parquet for {p_name}, skipping")
            continue
        log.info(f"\n=== {p_name} ({parquet.name}) ===")
        _swap_library_on(cg_for_a, parquet)
        scorer.refresh_track_embeddings()
        cov = scorer.coverage_stats()
        log.info(
            f"  Library coverage: {cov['coverage_pct']:.1%} "
            f"[direct={cov['direct']}, sim_artists={cov['similar_artists_fallback']}, "
            f"projector={cov.get('track_projector', 0)}, none={cov['no_coverage']}]"
        )

        queries: List[Tuple[str, str]] = []
        for q in ANCHOR_QUERIES:
            queries.append(("anchor", q))
        if not args.include_anchor_only:
            for q in TAILORED_QUERIES.get(p_name, []):
                queries.append(("tailored", q))

        for kind, q in queries:
            t0 = time.time()
            log.info(f"  [{kind}] {q[:90]}")
            # cache consultant recs per query
            if q not in consultant_cache:
                consultant_cache[q] = _consultant_relevance_set(consultant, q, library_df=cg_for_a.tracks_df)

            try:
                row = run_one_query(
                    cg_for_a, scorer, consultant, q, args.top_k,
                    scoring_mode=args.scoring_mode,
                )
            except Exception as e:
                log.warning(f"    ERROR {type(e).__name__}: {e}")
                continue
            row["participant"] = p_name
            row["query_kind"] = kind
            row["library_coverage_pct"] = float(cov["coverage_pct"])
            row["library_size"] = cov["total_tracks"]
            all_rows.append(row)
            full_payload.append(row)
            if q in headline_inv:
                pr_curves_payload.setdefault(headline_inv[q], {})
                pr_curves_payload[headline_inv[q]][p_name] = {
                    "a_pr_curve": row["a_pr_curve"],
                    "b_pr_curve": row["b_pr_curve"],
                    "n_consultant_ref_artists": row["n_consultant_ref_artists"],
                }
            ir_tag = " A=interaction_required" if row.get("a_interaction_required") else ""
            cu_tag = " consultant_UNAVAILABLE" if row.get("consultant_unavailable") else ""
            log.info(
                f"    A_ret={row['a_returned']:>2} B_ret={row['b_returned']:>2} "
                f"overlap={row['overlap_top_k']}/{min(row['a_returned'], row['b_returned']) or 1} "
                f"A_coh={row['a_genre_coherence']:.2f} B_coh={row['b_genre_coherence']:.2f} "
                f"A_AP={row['a_metrics']['consultant_AP@10']:.2f} B_AP={row['b_metrics']['consultant_AP@10']:.2f} "
                f"B_cov={row['b_candidate_coverage']:.0%} ({(time.time()-t0):.1f}s)"
                f"{ir_tag}{cu_tag}"
            )

    # ── Persist ──
    if all_rows:
        # CSV: per-query flat rows
        flat_cols = [
            "participant", "query_kind", "query", "library_size", "library_coverage_pct",
            "n_consultant_ref_artists", "consultant_unavailable",
            "a_interaction_required", "scoring_mode",
            "a_returned", "b_returned",
            "overlap_top_k", "overlap_pct",
            "a_genre_coherence", "b_genre_coherence",
            "a_bpm_stddev", "b_bpm_stddev",
            "a_AP_at_10", "b_AP_at_10", "a_MRR", "b_MRR", "a_P_at_10", "b_P_at_10",
            "a_n_relevant", "b_n_relevant",
            "b_candidate_coverage", "b_coverage_direct", "b_coverage_similar_artists",
            "b_coverage_track_projector", "b_coverage_none",
        ]
        with open(results_csv, "w") as f:
            f.write(",".join(flat_cols) + "\n")
            for r in all_rows:
                row = [
                    r["participant"], r["query_kind"], '"' + r["query"].replace('"', "'") + '"',
                    str(r["library_size"]), f"{r['library_coverage_pct']:.4f}",
                    str(r["n_consultant_ref_artists"]),
                    "1" if r.get("consultant_unavailable") else "0",
                    "1" if r.get("a_interaction_required") else "0",
                    r.get("scoring_mode", ""),
                    str(r["a_returned"]), str(r["b_returned"]),
                    str(r["overlap_top_k"]), f"{r['overlap_pct']:.4f}",
                    f"{r['a_genre_coherence']:.4f}", f"{r['b_genre_coherence']:.4f}",
                    f"{r['a_bpm_stddev']:.2f}", f"{r['b_bpm_stddev']:.2f}",
                    f"{r['a_metrics']['consultant_AP@10']:.4f}", f"{r['b_metrics']['consultant_AP@10']:.4f}",
                    f"{r['a_metrics']['MRR']:.4f}", f"{r['b_metrics']['MRR']:.4f}",
                    f"{r['a_metrics']['P@10']:.4f}", f"{r['b_metrics']['P@10']:.4f}",
                    str(r["a_metrics"]["n_relevant"]), str(r["b_metrics"]["n_relevant"]),
                    f"{r['b_candidate_coverage']:.4f}",
                    str(r["b_coverage_direct"]), str(r["b_coverage_similar_artists"]),
                    str(r["b_coverage_track_projector"]), str(r["b_coverage_none"]),
                ]
                f.write(",".join(row) + "\n")
        log.info(f"\nWrote per-query rows -> {results_csv}")

        # Summary CSV: per-participant aggregates + global aggregates.
        # Patch 2026-05-13:
        #  - AP/MRR/P aggregates exclude rows where consultant_unavailable=True
        #    (no ground truth to measure against → AP forced to 0 confounds the avg).
        #  - n_consultant_unavailable + n_a_interaction_required columns added so
        #    we can tell at a glance how many queries the eval couldn't fairly score.
        with open(summary_csv, "w") as f:
            f.write("scope,n_queries,n_consultable,n_consultant_unavailable,"
                    "n_a_interaction_required,mean_overlap_pct,"
                    "mean_a_AP,mean_b_AP,mean_a_MRR,mean_b_MRR,"
                    "mean_a_genre_coh,mean_b_genre_coh,mean_b_coverage,mean_b_proj_pct\n")

            def aggregate(rows):
                if not rows:
                    return None
                consultable = [r for r in rows if not r.get("consultant_unavailable")]
                n_unav = len(rows) - len(consultable)
                n_a_ir = sum(1 for r in rows if r.get("a_interaction_required"))

                def m_all(key, sub=None):
                    return float(np.mean([r[key] if sub is None else r[key][sub] for r in rows]))

                def m_consultable(key, sub=None):
                    if not consultable:
                        return 0.0
                    return float(np.mean([r[key] if sub is None else r[key][sub] for r in consultable]))

                return (
                    len(rows),
                    len(consultable),
                    n_unav,
                    n_a_ir,
                    m_all("overlap_pct"),
                    m_consultable("a_metrics", "consultant_AP@10"),
                    m_consultable("b_metrics", "consultant_AP@10"),
                    m_consultable("a_metrics", "MRR"),
                    m_consultable("b_metrics", "MRR"),
                    m_all("a_genre_coherence"),
                    m_all("b_genre_coherence"),
                    m_all("b_candidate_coverage"),
                    float(np.mean([
                        r["b_coverage_track_projector"] / max(
                            1,
                            r["a_returned"] + r["b_coverage_none"]
                            + r["b_coverage_direct"] + r["b_coverage_similar_artists"]
                        )
                        for r in rows
                    ])),
                )

            for p_name in sorted({r["participant"] for r in all_rows}):
                sub = [r for r in all_rows if r["participant"] == p_name]
                a = aggregate(sub)
                if a is None:
                    continue
                f.write(f"{p_name}," + ",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in a) + "\n")
            agg = aggregate(all_rows)
            if agg is not None:
                f.write("ALL," + ",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in agg) + "\n")
        log.info(f"Wrote summary -> {summary_csv}")

    # PR curves JSON for the headline figures
    pr_json.write_text(json.dumps(pr_curves_payload, indent=2))
    full_json.write_text(json.dumps([{**r, "a_pr_curve": r["a_pr_curve"], "b_pr_curve": r["b_pr_curve"]} for r in full_payload]))

    # ── Print dissertation summary ─────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"Evaluation summary — {label}")
    print("=" * 90)
    # Patch 2026-05-13: AP columns are averaged over CONSULTABLE rows only
    # (n_cons = consultable; n_unav = consultant_unavailable; n_ir = a_interaction_required).
    print(f"{'Slice':<22}{'N':>4}{'n_cons':>7}{'n_unav':>7}{'n_ir':>5}"
          f"{'overlap':>9}{'A_coh':>7}{'B_coh':>7}{'A_AP*':>7}{'B_AP*':>7}{'B_cov':>7}")
    print("-" * 90)
    if all_rows:
        def _print_row(name, rows):
            if not rows:
                return
            n = len(rows)
            consultable = [r for r in rows if not r.get("consultant_unavailable")]
            n_cons = len(consultable)
            n_unav = n - n_cons
            n_ir = sum(1 for r in rows if r.get("a_interaction_required"))
            ovl = np.mean([r["overlap_pct"] for r in rows])
            ac = np.mean([r["a_genre_coherence"] for r in rows])
            bc = np.mean([r["b_genre_coherence"] for r in rows])
            aap = np.mean([r["a_metrics"]["consultant_AP@10"] for r in consultable]) if consultable else 0.0
            bap = np.mean([r["b_metrics"]["consultant_AP@10"] for r in consultable]) if consultable else 0.0
            cov = np.mean([r["b_candidate_coverage"] for r in rows])
            print(f"{name:<22}{n:>4}{n_cons:>7}{n_unav:>7}{n_ir:>5}"
                  f"{ovl:>9.1%}{ac:>7.2f}{bc:>7.2f}{aap:>7.2f}{bap:>7.2f}{cov:>7.0%}")

        _print_row("ALL", all_rows)
        _print_row("anchor only", [r for r in all_rows if r["query_kind"] == "anchor"])
        _print_row("tailored only", [r for r in all_rows if r["query_kind"] == "tailored"])
        for p_name in sorted({r["participant"] for r in all_rows}):
            sub = [r for r in all_rows if r["participant"] == p_name]
            _print_row(f"  {p_name}", sub)
    print("\n* AP/MRR/P metrics averaged over consultable rows only "
          "(n_unav rows excluded — consultant returned no artist set, metric is undefined).")

    # version comparison table
    print("\n" + "=" * 90)
    print("Dissertation version comparison (best estimates available locally)")
    print("=" * 90)
    print("v1 — synthetic templates, artist-only embedding, hard genre filter")
    print("     (val_loss 4.09 on training; 0/10 top-K overlap on Adin's first 4 queries)")
    print("v2 — synthetic templates + similar_artists fallback + soft genre filter")
    print("     (val_loss 3.34 on training; mean top-K overlap ~54% on Adin's 9 May 2 queries)")
    print(f"{label} — current run, see CSVs above")
    print("=" * 90)


if __name__ == "__main__":
    main()
