"""
Stage 5.1 — Baseline retrieval methods for v4 eval.

Three baselines, all reading the same library + queries that
`comprehensive_eval_v3` does. Each produces an in-memory `tracks` list
ranked by its own scoring function so downstream metric computation
(consultant_AP@10, P@10, MRR, top-K overlap, BPM stddev, genre coherence)
is identical across methods.

  - random        — shuffle library, return top-K. Sanity floor.
  - bm25          — BM25 over per-track Gemini text (concatenated micro_genre,
                    subgenre_tags, mood_tags, sound_characteristics,
                    mixing_notes, cultural_context, description). No learning.
  - system_a      — heuristic CrateGenerator (same as the main A/B System A).

CLI:
  poetry run python -m src.evaluation.baselines \\
      --baseline {random, bm25, system_a} \\
      [--participant "Adin Segall"] \\
      [--queries data/evaluation_queries.json] \\
      [--library output/user_studies/participants/.../enriched_library.parquet] \\
      [--top-k 10] \\
      [--seed 20260512]

Output: prints a JSON record per (participant, query) to stdout. The
v4 eval orchestrator captures these for cross-method comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _flatten_str(x) -> str:
    """Coerce list-or-str to a single space-joined string."""
    if isinstance(x, list):
        return " ".join(str(v) for v in x if v)
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def _doc_for_track(row: pd.Series) -> str:
    """Build a BM25 document for a track from Gemini-enriched text fields."""
    parts = [
        _flatten_str(row.get("gemini_micro_genre")),
        _flatten_str(row.get("gemini_subgenre_tags")),
        _flatten_str(row.get("gemini_mood_tags")),
        _flatten_str(row.get("gemini_sound_characteristics")),
        _flatten_str(row.get("gemini_mixing_notes")),
        _flatten_str(row.get("gemini_cultural_context")),
        _flatten_str(row.get("gemini_description")),
        _flatten_str(row.get("gemini_dj_use_case")),
        _flatten_str(row.get("gemini_crowd_context")),
        _flatten_str(row.get("gemini_venue_type")),
    ]
    return " ".join(p for p in parts if p).lower()


def _track_record(row: pd.Series) -> dict:
    """Same shape that CrateGenerator emits — enough for metric computation."""
    return {
        "track_id": str(row.get("track_id", "")),
        "name": row.get("name", ""),
        "artist": row.get("artist", ""),
        "bpm": float(row.get("bpm") or 0.0),
        "gemini_micro_genre": row.get("gemini_micro_genre", ""),
    }


def baseline_random(df: pd.DataFrame, query: str, top_k: int, seed: int) -> list[dict]:
    rng = random.Random(seed + hash(query) % (2**31))
    indices = list(df.index)
    rng.shuffle(indices)
    return [_track_record(df.loc[i]) for i in indices[:top_k]]


def baseline_bm25(df: pd.DataFrame, query: str, top_k: int) -> list[dict]:
    if not HAS_BM25:
        raise RuntimeError("rank_bm25 not installed — `poetry add rank-bm25`")
    docs = [_doc_for_track(df.loc[i]).split() for i in df.index]
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(-scores)[:top_k]
    rows = df.iloc[top_idx]
    return [_track_record(r) for _, r in rows.iterrows()]


def baseline_system_a(query: str, library_path: Optional[Path], top_k: int) -> list[dict]:
    """Run System A (heuristic CrateGenerator) and return its top-K tracks."""
    from src.features.crate_generator import CrateGenerator
    import os

    cg = CrateGenerator(gemini_api_key=os.environ.get("GEMINI_API_KEY"))
    if library_path:
        # `_swap_library_on` lives in embedding_scorer; mirror here cheaply
        from src.features.embedding_scorer import _swap_library_on
        _swap_library_on(cg, library_path)

    result, _ = cg.generate_crate(
        query, verbose=False, include_unknown=False,
    )
    if isinstance(result, dict):
        tracks = result.get("tracks") or []
    else:
        tracks = []
    return [
        {"track_id": str(getattr(t, "track_id", "")),
         "name": getattr(t, "name", ""),
         "artist": getattr(t, "artist", ""),
         "bpm": float(getattr(t, "bpm", 0.0) or 0.0),
         "gemini_micro_genre": getattr(t, "gemini_micro_genre", "")}
        for t in tracks[:top_k]
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["random", "bm25", "system_a"], required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--library", type=Path, required=True,
                   help="Path to enriched_library.parquet")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260512)
    args = p.parse_args()

    df = pd.read_parquet(args.library)
    if args.baseline == "random":
        tracks = baseline_random(df, args.query, args.top_k, args.seed)
    elif args.baseline == "bm25":
        tracks = baseline_bm25(df, args.query, args.top_k)
    else:
        tracks = baseline_system_a(args.query, args.library, args.top_k)

    print(json.dumps({
        "baseline": args.baseline,
        "query": args.query,
        "top_k": args.top_k,
        "tracks": tracks,
    }, indent=2))


if __name__ == "__main__":
    main()
