"""
A/B comparison: System A (heuristic) vs System B (embedding) retrieval.

Runs every query in data/evaluation_queries.json against the same library
(data/tracks_master.parquet by default) using both scoring methods. Each
query is parsed ONCE by Gemini and the parsed DJSetRequest is fed to both
systems, so the only thing that varies is the scoring function:

    System A: CrateGenerator._score_candidates  (heuristic weights)
    System B: EmbeddingScorer._embedding_score  (cosine on a2v_v2 + query_encoder_v1)

Consultant is OFF for both systems for fair comparison of the scoring step.
The Gemini Consultant injects tracks with inclusion_boost=50 that bypass
hard filters; with consultant on, both systems would return the same
boosted tracks at the top and the scoring contribution would be obscured.

Outputs:
    output/ab_comparison_results.csv         per-track ranks + scores
    output/ab_comparison_summary.csv         per-query metrics
    + printed summary table

CLI:
    python -m src.evaluation.ab_comparison [--participant "Adin Segall"] [--top-k 10]
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.crate_generator import CrateGenerator, CrateTrack  # noqa: E402
from src.features.embedding_scorer import EmbeddingScorer            # noqa: E402
from src.features.schemas import DJSetRequest                        # noqa: E402

QUERIES_PATH = ROOT / "data" / "evaluation_queries.json"
LIBRARY_PATH = ROOT / "data" / "tracks_master.parquet"
OUTPUT_DIR = ROOT / "output"
RESULTS_CSV = OUTPUT_DIR / "ab_comparison_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "ab_comparison_summary.csv"
ENV_PATH = ROOT / ".env"

# Library swap helper for participant parquets
from src.features.embedding_scorer import _swap_library_on  # noqa: E402

logger = logging.getLogger(__name__)


# ── Setup helpers ───────────────────────────────────────────────────────────

def load_env(env_path: Path) -> None:
    """Read .env into os.environ (no python-dotenv dep)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def query_id_for(query_text: str) -> str:
    """Stable 8-char hex ID per unique query."""
    return hashlib.md5(query_text.encode("utf-8")).hexdigest()[:8]


# ── Param extraction (mirrors generate_crate / EmbeddingScorer.score) ──────

def extract_genres(request: DJSetRequest) -> List[str]:
    if not request.vibe or not request.vibe.genre_weights:
        return []
    weights = request.vibe.genre_weights
    n = len(weights)
    threshold = 0.05 if n >= 5 else (0.10 if n >= 3 else 0.20)
    genres = [g for g, w in weights.items() if w >= threshold]
    if not genres:
        genres = [max(weights.items(), key=lambda x: x[1])[0]]
    return genres


def max_per_artist_from_query(query: str) -> int:
    q = query.lower()
    if "album" in q or "discography" in q:
        return 10
    if "b2b" in q or "versus" in q:
        return 5
    return 2


def detect_popularity_prefs(query: str) -> Tuple[bool, bool]:
    q = query.lower()
    prefer_under = any(k in q for k in [
        "underground", "deep cuts", "obscure", "rare", "hidden gem",
        "leftfield", "niche", "under the radar", "smaller artists",
    ])
    prefer_main = any(k in q for k in [
        "hits", "mainstream", "festival", "popular", "crowd pleaser",
        "anthem", "bangers", "chart", "radio",
    ])
    return prefer_under, prefer_main


# ── Shared upstream: hard filter + verification ─────────────────────────────

def shared_pre_filter(
    cg: CrateGenerator,
    query: str,
    request: DJSetRequest,
) -> Dict[str, Any]:
    """
    Run the upstream pipeline that's identical between System A and B.
    Mutates request.raw_query but is otherwise side-effect-free on cg.
    """
    request.raw_query = query

    genres = extract_genres(request)
    primary_genre = genres[0] if genres else None
    bpm_range = request.bpm_range
    if primary_genre:
        bpm_range = cg._validate_genre_tempo(primary_genre, bpm_range)

    excluded = request.specifics.excluded_artists if request.specifics else []
    included = request.specifics.included_artists if request.specifics else []
    min_dur = getattr(request, "min_duration_minutes", None)
    max_dur = getattr(request, "max_duration_minutes", None)
    bpm_strict = getattr(request, "bpm_strict", False)
    year_min = getattr(request, "release_year_min", None)
    year_max = getattr(request, "release_year_max", None)
    sample_mode = max_dur is not None and max_dur <= 0.5

    exclude_mainstream = cg._detect_mainstream_exclusion(query, excluded)
    exclude_classics = cg._detect_classics_exclusion(query, excluded)

    candidates = cg._hard_filter(
        genre=primary_genre,
        bpm_range=bpm_range,
        excluded_artists=excluded,
        included_artists=included,
        min_duration_minutes=min_dur,
        max_duration_minutes=max_dur,
        bpm_strict=bpm_strict,
        exclude_mainstream=exclude_mainstream,
        exclude_classics=exclude_classics,
        raw_query=query,
        consultant_track_ids=set(),  # consultant OFF for fair A/B
        additional_genres=genres[1:] if len(genres) > 1 else [],
        release_year_min=year_min,
        release_year_max=year_max,
        sample_mode=sample_mode,
    )
    n_after_hard = len(candidates)

    if len(candidates) > 0 and not sample_mode:
        high_conf = ["definite", "probable"]
        bypass = candidates.get(
            "inclusion_boost", pd.Series(0.0, index=candidates.index)
        ) > 0
        candidates = candidates[
            (
                (candidates["gemini_is_known_track"] == True)
                & (candidates["gemini_recognition_confidence"].isin(high_conf))
            )
            | bypass
        ]
    n_after_verify = len(candidates)

    return {
        "candidates": candidates,
        "request": request,
        "genres": genres,
        "primary_genre": primary_genre,
        "bpm_range": bpm_range,
        "sample_mode": sample_mode,
        "n_after_hard": n_after_hard,
        "n_after_verify": n_after_verify,
    }


# ── System A: heuristic ────────────────────────────────────────────────────

def run_system_a(
    cg: CrateGenerator,
    query: str,
    request: DJSetRequest,
    top_k: int,
) -> Dict[str, Any]:
    state = shared_pre_filter(cg, query, request)
    candidates = state["candidates"]

    if len(candidates) == 0:
        return {**state, "scored": pd.DataFrame(), "tracks": [], "score_lookup": {}}

    venue_style = state["request"].venue_style
    venue_artists: Set[str] = set()
    if venue_style:
        venue_artists = set(cg._get_venue_artists(venue_style))

    vibe_keywords = request.vibe.keywords if request.vibe else []
    prefer_under, prefer_main = detect_popularity_prefs(query)

    scored = cg._score_candidates(
        candidates,
        venue_style,
        venue_artists,
        vibe_keywords=vibe_keywords,
        prefer_underground=prefer_under,
        prefer_mainstream=prefer_main,
    )

    priority_ids: Set[str] = set()
    if "inclusion_boost" in scored.columns:
        priority_ids = set(
            scored.loc[scored["inclusion_boost"] > 0, "track_id"].astype(str)
        )

    tracks, _ = cg._to_flat_list(
        scored,
        max_tracks=top_k,
        max_per_artist=max_per_artist_from_query(query),
        priority_track_ids=priority_ids,
        sample_mode=state["sample_mode"],
    )

    score_lookup = (
        scored.set_index(scored["track_id"].astype(str))["combined_score"].to_dict()
    )

    return {**state, "scored": scored, "tracks": tracks, "score_lookup": score_lookup}


# ── System B: embedding ─────────────────────────────────────────────────────

def _shared_pre_filter_no_genre(
    cg: CrateGenerator,
    query: str,
    request: DJSetRequest,
) -> Dict[str, Any]:
    """
    Like shared_pre_filter but drops the genre constraint at filter time.
    Used by System B in soft-filter mode. BPM / year / exclusions stay HARD.
    """
    request.raw_query = query

    genres = extract_genres(request)
    primary_genre = genres[0] if genres else None
    bpm_range = request.bpm_range
    if primary_genre:
        bpm_range = cg._validate_genre_tempo(primary_genre, bpm_range)

    excluded = request.specifics.excluded_artists if request.specifics else []
    included = request.specifics.included_artists if request.specifics else []
    min_dur = getattr(request, "min_duration_minutes", None)
    max_dur = getattr(request, "max_duration_minutes", None)
    bpm_strict = getattr(request, "bpm_strict", False)
    year_min = getattr(request, "release_year_min", None)
    year_max = getattr(request, "release_year_max", None)
    sample_mode = max_dur is not None and max_dur <= 0.5

    exclude_mainstream = cg._detect_mainstream_exclusion(query, excluded)
    exclude_classics = cg._detect_classics_exclusion(query, excluded)

    candidates = cg._hard_filter(
        genre=None, additional_genres=[],  # SOFT: drop genre at filter time
        bpm_range=bpm_range,
        excluded_artists=excluded, included_artists=included,
        min_duration_minutes=min_dur, max_duration_minutes=max_dur,
        bpm_strict=bpm_strict,
        exclude_mainstream=exclude_mainstream, exclude_classics=exclude_classics,
        raw_query=query, consultant_track_ids=set(),
        release_year_min=year_min, release_year_max=year_max,
        sample_mode=sample_mode,
    )
    n_after_hard = len(candidates)

    if len(candidates) > 0 and not sample_mode:
        high_conf = ["definite", "probable"]
        bypass = candidates.get("inclusion_boost", pd.Series(0.0, index=candidates.index)) > 0
        candidates = candidates[
            ((candidates["gemini_is_known_track"] == True)
             & (candidates["gemini_recognition_confidence"].isin(high_conf)))
            | bypass
        ]
    n_after_verify = len(candidates)

    return {
        "candidates": candidates, "request": request,
        "genres": genres, "primary_genre": primary_genre,
        "bpm_range": bpm_range, "sample_mode": sample_mode,
        "n_after_hard": n_after_hard, "n_after_verify": n_after_verify,
    }


def run_system_b(
    scorer: EmbeddingScorer,
    query: str,
    request: DJSetRequest,
    top_k: int,
    soft_genre_filter: bool = False,
) -> Dict[str, Any]:
    """
    Run System B retrieval. When soft_genre_filter=True, the genre constraint
    is moved from filter-time to scoring-time (genre_relevance × cosine).
    BPM/year/exclusions stay HARD. This shifts retrieval weight onto the
    embedding — the dissertation's research lever.
    """
    cg = scorer._cg

    if soft_genre_filter:
        state = _shared_pre_filter_no_genre(cg, query, request)
    else:
        state = shared_pre_filter(cg, query, request)
    candidates = state["candidates"]

    if len(candidates) == 0:
        return {
            **state, "scored": pd.DataFrame(), "tracks": [],
            "score_lookup": {}, "coverage": 0.0,
            "candidate_coverage_direct": 0,
            "candidate_coverage_similar_artists": 0,
            "candidate_coverage_none": 0,
            "soft_genre_filter_used": soft_genre_filter,
        }

    cand_idx = candidates.index
    n_with_emb = int(scorer._track_embeddings.loc[cand_idx, "has_embedding"].sum())
    coverage = n_with_emb / max(1, len(candidates))

    cov_src = scorer._track_embeddings.loc[cand_idx, "coverage_source"] \
        if "coverage_source" in scorer._track_embeddings.columns \
        else pd.Series(["direct"] * len(cand_idx), index=cand_idx)

    query_emb = scorer.encode_query(query)
    if soft_genre_filter and state["genres"]:
        gr = scorer._compute_genre_relevance(candidates, state["genres"])
        scored = scorer._embedding_score(candidates, query_emb, genre_relevance=gr)
    else:
        scored = scorer._embedding_score(candidates, query_emb)

    priority_ids: Set[str] = set()
    if "inclusion_boost" in scored.columns:
        priority_ids = set(
            scored.loc[scored["inclusion_boost"] > 0, "track_id"].astype(str)
        )

    tracks, _ = cg._to_flat_list(
        scored,
        max_tracks=top_k,
        max_per_artist=max_per_artist_from_query(query),
        priority_track_ids=priority_ids,
        sample_mode=state["sample_mode"],
    )

    score_lookup = (
        scored.set_index(scored["track_id"].astype(str))["embedding_score"].to_dict()
    )

    return {
        **state,
        "scored": scored,
        "tracks": tracks,
        "score_lookup": score_lookup,
        "coverage": coverage,
        "candidate_coverage_direct": int((cov_src == "direct").sum()),
        "candidate_coverage_similar_artists": int((cov_src == "similar_artists_fallback").sum()),
        "candidate_coverage_none": int((cov_src == "no_coverage").sum()),
        "soft_genre_filter_used": soft_genre_filter,
    }


# ── CSV row builder ─────────────────────────────────────────────────────────

def track_to_row(
    t: CrateTrack,
    score: Optional[float],
    *,
    query_id: str,
    query_text: str,
    participant: str,
    system: str,
    rank: int,
) -> dict:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "participant": participant,
        "system": system,
        "rank": rank,
        "track_artist": t.artist,
        "track_title": t.name,
        "score": f"{score:.6f}" if score is not None else "",
        "genre": t.genre,
        "bpm": f"{t.bpm:.1f}" if t.bpm else "",
        "energy": t.energy,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    arg_parser = argparse.ArgumentParser(
        description="A/B comparison: System A heuristic vs System B embedding retrieval."
    )
    arg_parser.add_argument("--top-k", type=int, default=10)
    arg_parser.add_argument(
        "--participant", type=str, default=None,
        help='Run only this participant (e.g. "Adin Segall"). Default: all 31 queries.',
    )
    arg_parser.add_argument(
        "--queries", type=str, default=str(QUERIES_PATH),
        help="Path to queries JSON. Default: data/evaluation_queries.json",
    )
    arg_parser.add_argument(
        "--library", type=str, default=str(LIBRARY_PATH),
        help="Path to library parquet. Default: data/tracks_master.parquet",
    )
    arg_parser.add_argument(
        "--soft-genre", action="store_true",
        help="System B uses soft genre filter (genre_relevance × cosine).",
    )
    arg_parser.add_argument(
        "--encoder-dir", type=str, default=None,
        help="Override query encoder dir (default: auto-resolve v2 then v1).",
    )
    arg_parser.add_argument(
        "--results-out", type=str, default=str(RESULTS_CSV),
        help="Per-track CSV output path",
    )
    arg_parser.add_argument(
        "--summary-out", type=str, default=str(SUMMARY_CSV),
        help="Per-query summary CSV output path",
    )
    args = arg_parser.parse_args()

    load_env(ENV_PATH)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not in environment or .env", file=sys.stderr)
        sys.exit(1)

    with open(args.queries) as f:
        all_queries = json.load(f)
    if args.participant:
        queries = [q for q in all_queries if q["participant_name"] == args.participant]
        if not queries:
            print(
                f"ERROR: no queries for participant '{args.participant}'. "
                f"Available: {sorted({q['participant_name'] for q in all_queries})}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        queries = all_queries
    print(f"Running {len(queries)} queries × 2 systems × top-{args.top_k}")

    print("\n[init] System A — CrateGenerator (heuristic)")
    cg = CrateGenerator(gemini_api_key=api_key)
    cg.consultant = None  # belt-and-suspenders: never invoked anyway
    print("  Consultant disabled for fair A/B.")

    # Swap library if a non-default parquet is requested
    library_path = Path(args.library)
    default_library = LIBRARY_PATH.resolve()
    if library_path.resolve() != default_library:
        _swap_library_on(cg, library_path)

    print(f"\n[init] System B — EmbeddingScorer (soft_genre={args.soft_genre})")
    encoder_dir = Path(args.encoder_dir) if args.encoder_dir else None
    scorer = EmbeddingScorer(
        gemini_api_key=api_key,
        use_consultant=False,
        soft_genre_filter=args.soft_genre,
        encoder_dir=encoder_dir,
        crate_generator=cg,  # share the (possibly library-swapped) CG
    )
    # If we swapped the library on cg AFTER scorer.__init__ would have built
    # its track-embedding cache from cg.tracks_df, the cache is now stale.
    # When we share a pre-loaded cg, scorer init builds the cache from the
    # current df, so this is a no-op; but call it explicitly for clarity.
    scorer.refresh_track_embeddings()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows: List[dict] = []
    summary_rows: List[dict] = []

    for i, q in enumerate(queries, 1):
        query_text = q["query_text"]
        participant = q["participant_name"]
        qid = query_id_for(query_text)

        print(f"\n[{i}/{len(queries)}] {qid} {participant[:18]:<18} | {query_text[:80]}")

        # Parse ONCE per query — both systems consume the same DJSetRequest
        try:
            request = cg.parser.parse_request(query_text)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"  PARSE FAILED: {err}")
            summary_rows.append({
                "query_id": qid, "participant": participant, "query_text": query_text,
                "a_candidates_hard": 0, "a_candidates_verified": 0, "a_returned": 0,
                "a_avg_score": None, "a_error": f"parse: {err}",
                "b_candidates_hard": 0, "b_candidates_verified": 0, "b_returned": 0,
                "b_avg_score": None, "b_coverage": 0.0, "b_error": f"parse: {err}",
                "overlap": 0, "overlap_pct": 0.0,
            })
            continue

        # ── System A ──
        a_err = None
        a_tracks: List[CrateTrack] = []
        a_lookup: Dict[str, float] = {}
        a_n_hard = a_n_verify = 0
        try:
            a_state = run_system_a(cg, query_text, request.model_copy(deep=True), args.top_k)
            a_tracks = a_state["tracks"]
            a_lookup = a_state["score_lookup"]
            a_n_hard = a_state["n_after_hard"]
            a_n_verify = a_state["n_after_verify"]
            for rank, t in enumerate(a_tracks, 1):
                csv_rows.append(track_to_row(
                    t, a_lookup.get(str(t.track_id)),
                    query_id=qid, query_text=query_text, participant=participant,
                    system="A_heuristic", rank=rank,
                ))
            print(f"  A: hard={a_n_hard:>4}  verify={a_n_verify:>4}  returned={len(a_tracks):>3}")
        except Exception as e:
            a_err = f"{type(e).__name__}: {e}"
            print(f"  A: ERROR {a_err}")

        # ── System B ──
        b_err = None
        b_tracks: List[CrateTrack] = []
        b_lookup: Dict[str, float] = {}
        b_n_hard = b_n_verify = 0
        b_coverage = 0.0
        try:
            b_state = run_system_b(
                scorer, query_text, request.model_copy(deep=True), args.top_k,
                soft_genre_filter=args.soft_genre,
            )
            b_tracks = b_state["tracks"]
            b_lookup = b_state["score_lookup"]
            b_n_hard = b_state["n_after_hard"]
            b_n_verify = b_state["n_after_verify"]
            b_coverage = b_state.get("coverage", 0.0)
            for rank, t in enumerate(b_tracks, 1):
                csv_rows.append(track_to_row(
                    t, b_lookup.get(str(t.track_id)),
                    query_id=qid, query_text=query_text, participant=participant,
                    system="B_embedding", rank=rank,
                ))
            print(
                f"  B: hard={b_n_hard:>4}  verify={b_n_verify:>4}  "
                f"returned={len(b_tracks):>3}  cov={b_coverage:.0%}"
            )
        except Exception as e:
            b_err = f"{type(e).__name__}: {e}"
            print(f"  B: ERROR {b_err}")

        # Per-query metrics
        a_ids = {str(t.track_id) for t in a_tracks}
        b_ids = {str(t.track_id) for t in b_tracks}
        overlap = a_ids & b_ids
        denom = max(1, min(len(a_tracks), len(b_tracks)))  # overlap as % of smaller list

        avg_a = (sum(a_lookup.get(str(t.track_id), 0.0) for t in a_tracks) / len(a_tracks)
                 if a_tracks else None)
        avg_b = (sum(b_lookup.get(str(t.track_id), 0.0) for t in b_tracks) / len(b_tracks)
                 if b_tracks else None)

        if a_tracks and b_tracks:
            print(f"  overlap (top-{args.top_k}): {len(overlap)}/{denom} = {len(overlap)/denom:.0%}")

        summary_rows.append({
            "query_id": qid, "participant": participant, "query_text": query_text,
            "a_candidates_hard": a_n_hard, "a_candidates_verified": a_n_verify,
            "a_returned": len(a_tracks), "a_avg_score": avg_a, "a_error": a_err or "",
            "b_candidates_hard": b_n_hard, "b_candidates_verified": b_n_verify,
            "b_returned": len(b_tracks), "b_avg_score": avg_b,
            "b_coverage": b_coverage, "b_error": b_err or "",
            "b_coverage_direct": b_state.get("candidate_coverage_direct", 0) if not b_err else 0,
            "b_coverage_similar_artists": b_state.get("candidate_coverage_similar_artists", 0) if not b_err else 0,
            "b_coverage_none": b_state.get("candidate_coverage_none", 0) if not b_err else 0,
            "b_soft_genre_filter": int(bool(b_state.get("soft_genre_filter_used", False))) if not b_err else 0,
            "overlap": len(overlap), "overlap_pct": len(overlap) / denom,
        })

    # ── Write CSVs ──
    if csv_rows:
        with open(args.results_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} per-track rows → {args.results_out}")

    if summary_rows:
        with open(args.summary_out, "w", newline="") as f:
            cols = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in summary_rows:
                row = dict(r)
                for k in ("a_avg_score", "b_avg_score"):
                    row[k] = f"{row[k]:.6f}" if row[k] is not None else ""
                row["b_coverage"] = f"{row['b_coverage']:.4f}"
                row["overlap_pct"] = f"{row['overlap_pct']:.4f}"
                writer.writerow(row)
        print(f"Wrote {len(summary_rows)} summary rows → {args.summary_out}")

    # ── Print summary table ──
    print()
    print("=" * 132)
    print("PER-QUERY SUMMARY")
    print("=" * 132)
    header = (
        f"{'qid':<9} {'participant':<17} "
        f"{'A_hard':>6} {'A_ret':>5} {'A_avg':>7} "
        f"{'B_hard':>6} {'B_ret':>5} {'B_avg':>7} {'B_cov':>5} "
        f"{'overlap':>7}  query"
    )
    print(header)
    print("-" * 132)
    for r in summary_rows:
        avg_a = f"{r['a_avg_score']:.3f}" if r["a_avg_score"] is not None else "—"
        avg_b = f"{r['b_avg_score']:.3f}" if r["b_avg_score"] is not None else "—"
        cov = f"{r['b_coverage']:.0%}" if r["b_returned"] else "—"
        ovl = f"{r['overlap']}/{min(r['a_returned'], r['b_returned'])}" if r["a_returned"] and r["b_returned"] else "—"
        line = (
            f"{r['query_id']:<9} {r['participant'][:16]:<17} "
            f"{r['a_candidates_hard']:>6} {r['a_returned']:>5} {avg_a:>7} "
            f"{r['b_candidates_hard']:>6} {r['b_returned']:>5} {avg_b:>7} {cov:>5} "
            f"{ovl:>7}  {r['query_text'][:55]}"
        )
        print(line)

    # Aggregates
    print("-" * 132)
    a_avgs = [r["a_avg_score"] for r in summary_rows if r["a_avg_score"] is not None]
    b_avgs = [r["b_avg_score"] for r in summary_rows if r["b_avg_score"] is not None]
    overlaps = [
        r["overlap_pct"] for r in summary_rows
        if r["a_returned"] and r["b_returned"]
    ]
    coverages = [r["b_coverage"] for r in summary_rows if r["b_returned"]]
    a_errs = sum(1 for r in summary_rows if r["a_error"])
    b_errs = sum(1 for r in summary_rows if r["b_error"])

    print(f"AGGREGATE  N={len(summary_rows)} queries")
    if a_avgs:
        print(f"  System A — mean of per-query avg scores (heuristic, ~[0, 1.5]): {sum(a_avgs)/len(a_avgs):.3f}")
    if b_avgs:
        print(f"  System B — mean of per-query avg cosines (~[-1, 1]):            {sum(b_avgs)/len(b_avgs):.3f}")
    if coverages:
        print(f"  System B — mean candidate embedding coverage:                   {sum(coverages)/len(coverages):.1%}")
    if overlaps:
        print(f"  Mean top-{args.top_k} overlap (smaller-list denominator):              {sum(overlaps)/len(overlaps):.1%}")
    print(f"  Errors: A={a_errs}, B={b_errs}")
    print("=" * 132)


if __name__ == "__main__":
    main()
