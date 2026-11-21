"""
Query-encoder generalization eval (Task 4).

Trains a query encoder on 80% of synthetic-query template families (3
families held out: T4_radio_mix, T5_genre_from_year, T9_time_genre).
Evaluates centroid retrieval on:

  In-template queries  (templates the model has seen)
  Out-template queries (templates the model has NEVER seen)

For each test query the ground-truth target is its tracklist's IDF-weighted
centroid. We rank ALL 170k centroids by cosine similarity and report:

  P@10  : fraction of test queries whose true centroid is in top-10
  R@10  : same as P@10 in single-target retrieval (kept for clarity)
  MRR   : mean reciprocal rank of the true centroid

The dissertation use case: pre-empt the examiner question
  "does the encoder generalise to unseen template patterns?"

Run:
  poetry run python -m src.evaluation.query_encoder_generalization_eval
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModel, AutoTokenizer

from src.models.train_query_encoder_v2 import _build_model, MODEL_NAME, MAX_LEN
from src.models.train_query_encoder_heldout import HELDOUT_TEMPLATES

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

CENTROID_PATH = DATA_DIR / "tracklist_centroids_v2.npz"
TPL_PAIRS_PATH = DATA_DIR / "query_encoder_training_pairs_with_template.json"
HELDOUT_ENCODER_DIR = MODELS_DIR / "query_encoder_heldout"
RESULT_PATH = OUTPUT_DIR / "query_encoder_generalization_eval.json"

SEED = 42
SAMPLE_PER_BUCKET = 5000  # eval at most this many queries per bucket


def _load_encoder(encoder_dir: Path):
    ckpt = torch.load(encoder_dir / "query_encoder.pt", map_location="cpu")
    tok = AutoTokenizer.from_pretrained(str(encoder_dir))
    transformer = AutoModel.from_pretrained(ckpt["model_name"])
    model = _build_model(transformer, ckpt["base_dim"], ckpt["target_dim"])
    model.transformer.load_state_dict(ckpt["transformer_state_dict"])
    try:
        model.projection.load_state_dict(ckpt["projection_state_dict"])
    except RuntimeError:
        # v1 used ReLU-only projection — fall back
        import torch.nn as nn
        model.projection = nn.Sequential(
            nn.Linear(ckpt["base_dim"], ckpt.get("hidden_dim", 256)),
            nn.ReLU(),
            nn.Linear(ckpt.get("hidden_dim", 256), ckpt["target_dim"]),
        )
        model.projection.load_state_dict(ckpt["projection_state_dict"])
    return model.eval(), tok, ckpt


def _retrieval_metrics(queries: list[str], centroid_idx_truth: list[int],
                       model, tok, centroids: np.ndarray, batch_size: int = 256,
                       device: str = "cpu", k: int = 10) -> dict:
    """For each query, compute rank of the true centroid among ALL centroids."""
    model = model.to(device)
    centroids_t = torch.tensor(centroids, dtype=torch.float32, device=device)

    ranks = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        truth = centroid_idx_truth[start:start + batch_size]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.no_grad():
            q = model(enc["input_ids"], enc["attention_mask"])
            sims = q @ centroids_t.T  # (B, N_centroids)
            # Rank of true centroid: count how many centroids score >= the true one
            true_scores = sims[range(len(chunk)), truth]
            # rank-1 if no centroid scores higher; >1 means there are k centroids above
            higher = (sims > true_scores.unsqueeze(1)).sum(dim=1)
            equal_or_higher = higher  # Don't count ties as worse — defensible
            ranks.extend((equal_or_higher + 1).cpu().tolist())

    ranks_arr = np.array(ranks, dtype=np.int64)
    p_at_k = float((ranks_arr <= k).mean())
    mrr = float((1.0 / ranks_arr).mean())
    median_rank = float(np.median(ranks_arr))
    return {
        "n": len(queries),
        f"p_at_{k}": p_at_k,
        f"r_at_{k}": p_at_k,  # single-target retrieval — same as P@k
        "mrr": mrr,
        "median_rank": median_rank,
    }


def main():
    if not HELDOUT_ENCODER_DIR.exists() or not (HELDOUT_ENCODER_DIR / "query_encoder.pt").exists():
        print(f"ERROR: held-out encoder not found at {HELDOUT_ENCODER_DIR}")
        print("Run `poetry run python -m src.models.train_query_encoder_heldout` first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────
    print(f"Loading {TPL_PAIRS_PATH.name}...")
    with TPL_PAIRS_PATH.open() as f:
        pairs = json.load(f)
    print(f"  {len(pairs):,} pairs")

    cd = np.load(CENTROID_PATH)
    centroids = cd["centroids"]
    print(f"  Centroids: {centroids.shape}")

    # Same tracklist split as training (80/20 by tracklist_id, SEED=42)
    rng = random.Random(SEED)
    unique_tl_ids = list(set(p["tracklist_id"] for p in pairs))
    rng.shuffle(unique_tl_ids)
    split_idx = int(len(unique_tl_ids) * 0.8)
    val_tl_ids = set(unique_tl_ids[split_idx:])
    print(f"  Val tracklists: {len(val_tl_ids):,} / {len(unique_tl_ids):,}")

    # Eval pools — only val tracklists, split by template family
    in_tpl_pool = [p for p in pairs
                   if p["tracklist_id"] in val_tl_ids
                   and p["template"] not in HELDOUT_TEMPLATES]
    out_tpl_pool = [p for p in pairs
                    if p["tracklist_id"] in val_tl_ids
                    and p["template"] in HELDOUT_TEMPLATES]
    print(f"  In-template eval pool:  {len(in_tpl_pool):,}")
    print(f"  Out-template eval pool: {len(out_tpl_pool):,}")
    if not out_tpl_pool:
        print("ERROR: no out-of-template pairs in val tracklists.", file=sys.stderr)
        sys.exit(1)

    # Sample to keep eval cost bounded
    rng.seed(SEED)
    in_sample = rng.sample(in_tpl_pool, min(SAMPLE_PER_BUCKET, len(in_tpl_pool)))
    out_sample = rng.sample(out_tpl_pool, min(SAMPLE_PER_BUCKET, len(out_tpl_pool)))
    print(f"  Sampled: in={len(in_sample):,}, out={len(out_sample):,}")

    # Per-template breakdown for the held-out group
    out_by_tpl: dict[str, list] = {}
    for p in out_tpl_pool:
        out_by_tpl.setdefault(p["template"], []).append(p)
    out_by_tpl_sample = {
        tpl: rng.sample(plist, min(SAMPLE_PER_BUCKET // len(out_by_tpl), len(plist)))
        for tpl, plist in out_by_tpl.items()
    }

    # ── Load held-out encoder ──────────────────────────────────
    print(f"\nLoading encoder from {HELDOUT_ENCODER_DIR.name}/...")
    model, tok, ckpt = _load_encoder(HELDOUT_ENCODER_DIR)
    print(f"  trained_epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")
    print(f"  held_out_templates={ckpt.get('held_out_templates')}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Run metrics ────────────────────────────────────────────
    def _run(label, sample):
        qs = [p["query_text"] for p in sample]
        truth = [p["centroid_index"] for p in sample]
        m = _retrieval_metrics(qs, truth, model, tok, centroids, device=device)
        print(f"  {label:<35} n={m['n']:>5}  P@10={m['p_at_10']:.4f}  "
              f"MRR={m['mrr']:.4f}  median_rank={m['median_rank']:.0f}")
        return m

    print("\nEvaluating retrieval against full centroid pool (170k)...")
    print()
    results = {
        "encoder": str(HELDOUT_ENCODER_DIR.name),
        "checkpoint": {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "held_out_templates": ckpt.get("held_out_templates"),
        },
        "n_centroids": int(centroids.shape[0]),
        "buckets": {},
        "per_held_out_template": {},
    }
    results["buckets"]["in_template_val_tls"] = _run("IN-TEMPLATE  (seen)",  in_sample)
    results["buckets"]["out_template_val_tls"] = _run("OUT-TEMPLATE (held-out)", out_sample)

    print("\nPer held-out template family:")
    for tpl, plist in sorted(out_by_tpl_sample.items()):
        m = _run(f"  {tpl}", plist)
        results["per_held_out_template"][tpl] = m

    # ── Generalization gap ─────────────────────────────────────
    in_metrics = results["buckets"]["in_template_val_tls"]
    out_metrics = results["buckets"]["out_template_val_tls"]
    results["gap"] = {
        "p_at_10_drop": in_metrics["p_at_10"] - out_metrics["p_at_10"],
        "mrr_drop":     in_metrics["mrr"]     - out_metrics["mrr"],
        "interpretation": (
            "Negative or near-zero drop = encoder generalizes across templates. "
            "Large positive drop = encoder over-fits to the templates it saw."
        ),
    }

    print(f"\nGeneralization gap:  ΔP@10 = {results['gap']['p_at_10_drop']:+.4f}, "
          f"ΔMRR = {results['gap']['mrr_drop']:+.4f}")

    with RESULT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {RESULT_PATH}")


if __name__ == "__main__":
    main()
