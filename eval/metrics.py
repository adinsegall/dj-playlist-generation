"""
ML Evaluation Metrics for DJ Playlist Generation.

Provides publication-quality metrics for:
- Embedding quality (intrinsic evaluation, analogy, clustering)
- Retrieval metrics (Precision@K, Recall@K, MRR)
- Transition model evaluation (perplexity, coverage, source attribution)
- Playlist coherence metrics
- Statistical significance tests (Wilcoxon, bootstrap CI, Cohen's d)

Author: Adin Segall
"""

import json
import pickle
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class MetricResult:
    """Container for a metric result with confidence interval."""
    name: str
    value: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    std: float = 0.0
    n_samples: int = 0

    def __repr__(self) -> str:
        if self.ci_lower > 0 and self.ci_upper > 0:
            return f"{self.name}: {self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return f"{self.name}: {self.value:.4f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std": self.std,
            "n_samples": self.n_samples,
        }


class EmbeddingMetrics:
    """Intrinsic evaluation metrics for artist embeddings."""

    def __init__(self, model=None, model_path: Optional[Path] = None):
        """
        Initialize with a gensim Word2Vec model.

        Args:
            model: Loaded Word2Vec model
            model_path: Path to model file (if model not provided)
        """
        if model is None and model_path:
            from gensim.models import Word2Vec
            self.model = Word2Vec.load(str(model_path))
        else:
            self.model = model

        self.wv = self.model.wv if self.model else None

    def cosine_similarity(self, artist_a: str, artist_b: str) -> float:
        """Calculate cosine similarity between two artists."""
        if not self.wv:
            return 0.0

        a_lower = artist_a.lower().strip()
        b_lower = artist_b.lower().strip()

        if a_lower not in self.wv or b_lower not in self.wv:
            return 0.0

        vec_a = self.wv[a_lower]
        vec_b = self.wv[b_lower]

        return float(1 - cosine(vec_a, vec_b))

    def collaborator_vs_random_similarity(
        self,
        collaborator_pairs: List[Tuple[str, str]],
        n_random: int = 1000,
        seed: int = 42
    ) -> Dict[str, MetricResult]:
        """
        Compare mean cosine similarity for collaborator pairs vs random pairs.

        Collaborators should have higher similarity if embeddings capture
        musical relationships.

        Args:
            collaborator_pairs: List of (artist_a, artist_b) known collaborators
            n_random: Number of random pairs to sample
            seed: Random seed for reproducibility

        Returns:
            Dict with collaborator_sim, random_sim, and effect_size
        """
        if not self.wv:
            return {}

        np.random.seed(seed)

        # Calculate collaborator similarities
        collab_sims = []
        for a, b in collaborator_pairs:
            sim = self.cosine_similarity(a, b)
            if sim > 0:  # Both artists found
                collab_sims.append(sim)

        # Sample random pairs
        vocab = list(self.wv.key_to_index.keys())
        # Filter out SOURCE_ tags
        vocab = [v for v in vocab if not v.startswith("SOURCE_")]

        random_sims = []
        attempts = 0
        while len(random_sims) < n_random and attempts < n_random * 10:
            a, b = np.random.choice(vocab, size=2, replace=False)
            sim = self.cosine_similarity(a, b)
            if sim > 0:
                random_sims.append(sim)
            attempts += 1

        collab_mean = np.mean(collab_sims) if collab_sims else 0.0
        collab_std = np.std(collab_sims) if collab_sims else 0.0
        random_mean = np.mean(random_sims) if random_sims else 0.0
        random_std = np.std(random_sims) if random_sims else 0.0

        # Cohen's d effect size
        if collab_std > 0 and random_std > 0:
            pooled_std = np.sqrt((collab_std**2 + random_std**2) / 2)
            effect_size = (collab_mean - random_mean) / pooled_std if pooled_std > 0 else 0.0
        else:
            effect_size = 0.0

        return {
            "collaborator_similarity": MetricResult(
                name="Collaborator Similarity",
                value=collab_mean,
                std=collab_std,
                n_samples=len(collab_sims)
            ),
            "random_similarity": MetricResult(
                name="Random Pair Similarity",
                value=random_mean,
                std=random_std,
                n_samples=len(random_sims)
            ),
            "effect_size_cohens_d": MetricResult(
                name="Effect Size (Cohen's d)",
                value=effect_size,
                n_samples=len(collab_sims) + len(random_sims)
            ),
        }

    def analogy_accuracy(
        self,
        analogies: List[Tuple[str, str, str, str]],
        topn: int = 10
    ) -> MetricResult:
        """
        Test embedding analogies: a:b :: c:d

        Example: fisher:chris_lake :: carl_cox:adam_beyer
        Tests if embeddings capture genre/style relationships.

        Args:
            analogies: List of (a, b, c, expected_d) tuples
            topn: Number of candidates to check for expected answer

        Returns:
            MetricResult with accuracy
        """
        if not self.wv:
            return MetricResult(name="Analogy Accuracy", value=0.0)

        correct = 0
        total = 0

        for a, b, c, expected_d in analogies:
            a_low, b_low, c_low, d_low = a.lower(), b.lower(), c.lower(), expected_d.lower()

            # Check all artists exist
            if not all(x in self.wv for x in [a_low, b_low, c_low, d_low]):
                continue

            total += 1

            try:
                # a:b :: c:? -> solve for d
                # d = b - a + c
                result = self.wv.most_similar(
                    positive=[b_low, c_low],
                    negative=[a_low],
                    topn=topn
                )

                candidates = [name for name, score in result]
                if d_low in candidates:
                    correct += 1
            except KeyError:
                continue

        accuracy = correct / total if total > 0 else 0.0

        return MetricResult(
            name="Analogy Accuracy",
            value=accuracy,
            n_samples=total
        )

    def genre_clustering_silhouette(
        self,
        artist_genres: Dict[str, str],
        n_clusters: int = 10,
        seed: int = 42
    ) -> MetricResult:
        """
        Calculate silhouette score for genre clustering in embedding space.

        Higher silhouette score means embeddings separate genres well.

        Args:
            artist_genres: Dict mapping artist name to genre label
            n_clusters: Number of clusters for KMeans
            seed: Random seed

        Returns:
            MetricResult with silhouette score
        """
        if not self.wv:
            return MetricResult(name="Genre Clustering Silhouette", value=0.0)

        # Collect embeddings and labels for artists with known genres
        embeddings = []
        labels = []

        for artist, genre in artist_genres.items():
            artist_lower = artist.lower().strip()
            if artist_lower in self.wv:
                embeddings.append(self.wv[artist_lower])
                labels.append(genre)

        if len(embeddings) < n_clusters * 2:
            return MetricResult(
                name="Genre Clustering Silhouette",
                value=0.0,
                n_samples=len(embeddings)
            )

        X = np.array(embeddings)
        y = np.array(labels)

        # Get unique genres and encode
        unique_genres = list(set(labels))
        genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        y_encoded = np.array([genre_to_idx[g] for g in y])

        # Calculate silhouette score using true labels
        try:
            score = silhouette_score(X, y_encoded)
        except ValueError:
            score = 0.0

        return MetricResult(
            name="Genre Clustering Silhouette",
            value=score,
            n_samples=len(embeddings)
        )

    def source_tag_analysis(self) -> Dict[str, Any]:
        """
        Analyze how SOURCE_ tags cluster with different artists.

        Returns similarity of source tags to each other and top associated artists.
        """
        if not self.wv:
            return {}

        source_tags = ["SOURCE_1001TL", "SOURCE_DJMIX", "SOURCE_MPD"]
        results = {}

        for tag in source_tags:
            if tag not in self.wv:
                continue

            # Get top artists near this source
            try:
                neighbors = self.wv.most_similar(tag, topn=20)
                # Filter out other source tags
                artists = [(n, s) for n, s in neighbors if not n.startswith("SOURCE_")]
                results[tag] = {
                    "top_artists": artists[:10],
                    "vector_norm": float(np.linalg.norm(self.wv[tag]))
                }
            except KeyError:
                continue

        # Cross-source similarities
        for i, tag_a in enumerate(source_tags):
            for tag_b in source_tags[i+1:]:
                if tag_a in self.wv and tag_b in self.wv:
                    sim = self.cosine_similarity(tag_a, tag_b)
                    results[f"{tag_a}_vs_{tag_b}"] = sim

        return results


class RetrievalMetrics:
    """Information retrieval metrics for artist similarity."""

    def __init__(self, model=None, artist_genres: Optional[Dict[str, str]] = None):
        """
        Initialize with model and genre ground truth.

        Args:
            model: Word2Vec model
            artist_genres: Dict mapping artist to genre for relevance judgments
        """
        self.model = model
        self.wv = model.wv if model else None
        self.artist_genres = artist_genres or {}

    def precision_at_k(
        self,
        query_artist: str,
        k: int = 10
    ) -> float:
        """
        Of top K similar artists, how many are same genre?

        Args:
            query_artist: Artist to query
            k: Number of results to consider

        Returns:
            Precision score (0-1)
        """
        if not self.wv or not self.artist_genres:
            return 0.0

        query_lower = query_artist.lower().strip()
        if query_lower not in self.wv:
            return 0.0

        query_genre = self.artist_genres.get(query_artist) or self.artist_genres.get(query_lower)
        if not query_genre:
            return 0.0

        try:
            neighbors = self.wv.most_similar(query_lower, topn=k + 5)
            neighbors = [(n, s) for n, s in neighbors if not n.startswith("SOURCE_")][:k]
        except KeyError:
            return 0.0

        relevant = 0
        for neighbor, score in neighbors:
            neighbor_genre = self.artist_genres.get(neighbor) or self.artist_genres.get(neighbor.lower())
            if neighbor_genre == query_genre:
                relevant += 1

        return relevant / k if k > 0 else 0.0

    def recall_at_k(
        self,
        query_artist: str,
        k: int = 10
    ) -> float:
        """
        Of all same-genre artists, how many in top K?

        Args:
            query_artist: Artist to query
            k: Number of results to consider

        Returns:
            Recall score (0-1)
        """
        if not self.wv or not self.artist_genres:
            return 0.0

        query_lower = query_artist.lower().strip()
        if query_lower not in self.wv:
            return 0.0

        query_genre = self.artist_genres.get(query_artist) or self.artist_genres.get(query_lower)
        if not query_genre:
            return 0.0

        # Count total same-genre artists in vocab
        total_same_genre = sum(
            1 for a, g in self.artist_genres.items()
            if g == query_genre and a.lower() in self.wv and a.lower() != query_lower
        )

        if total_same_genre == 0:
            return 0.0

        try:
            neighbors = self.wv.most_similar(query_lower, topn=k + 5)
            neighbors = [(n, s) for n, s in neighbors if not n.startswith("SOURCE_")][:k]
        except KeyError:
            return 0.0

        retrieved_same_genre = 0
        for neighbor, score in neighbors:
            neighbor_genre = self.artist_genres.get(neighbor) or self.artist_genres.get(neighbor.lower())
            if neighbor_genre == query_genre:
                retrieved_same_genre += 1

        return retrieved_same_genre / total_same_genre

    def mean_reciprocal_rank(
        self,
        query_artists: List[str],
        max_k: int = 100
    ) -> MetricResult:
        """
        Calculate Mean Reciprocal Rank (MRR) for genre retrieval.

        MRR = 1/|Q| * sum(1/rank_i) where rank_i is position of first relevant result.

        Args:
            query_artists: List of artists to query
            max_k: Maximum rank to consider

        Returns:
            MetricResult with MRR
        """
        if not self.wv or not self.artist_genres:
            return MetricResult(name="MRR", value=0.0)

        reciprocal_ranks = []

        for query in query_artists:
            query_lower = query.lower().strip()
            if query_lower not in self.wv:
                continue

            query_genre = self.artist_genres.get(query) or self.artist_genres.get(query_lower)
            if not query_genre:
                continue

            try:
                neighbors = self.wv.most_similar(query_lower, topn=max_k + 5)
                neighbors = [(n, s) for n, s in neighbors if not n.startswith("SOURCE_")]
            except KeyError:
                continue

            # Find first relevant result
            for rank, (neighbor, score) in enumerate(neighbors[:max_k], 1):
                neighbor_genre = self.artist_genres.get(neighbor) or self.artist_genres.get(neighbor.lower())
                if neighbor_genre == query_genre:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)  # No relevant result found

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        std = np.std(reciprocal_ranks) if reciprocal_ranks else 0.0

        return MetricResult(
            name="MRR",
            value=mrr,
            std=std,
            n_samples=len(reciprocal_ranks)
        )

    def aggregate_retrieval_metrics(
        self,
        query_artists: List[str],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, MetricResult]:
        """
        Calculate all retrieval metrics for a set of query artists.

        Args:
            query_artists: List of artists to evaluate
            k_values: K values for P@K and R@K

        Returns:
            Dict of metric name to MetricResult
        """
        results = {}

        for k in k_values:
            p_scores = []
            r_scores = []

            for artist in query_artists:
                p = self.precision_at_k(artist, k)
                r = self.recall_at_k(artist, k)
                if p > 0 or r > 0:  # Artist was found
                    p_scores.append(p)
                    r_scores.append(r)

            if p_scores:
                results[f"P@{k}"] = MetricResult(
                    name=f"Precision@{k}",
                    value=np.mean(p_scores),
                    std=np.std(p_scores),
                    n_samples=len(p_scores)
                )

            if r_scores:
                results[f"R@{k}"] = MetricResult(
                    name=f"Recall@{k}",
                    value=np.mean(r_scores),
                    std=np.std(r_scores),
                    n_samples=len(r_scores)
                )

        # Add MRR
        results["MRR"] = self.mean_reciprocal_rank(query_artists)

        return results


class TransitionMetrics:
    """Metrics for evaluating the transition matrix model."""

    def __init__(
        self,
        transition_matrix: Optional[sparse.csr_matrix] = None,
        artist_to_idx: Optional[Dict[str, int]] = None,
        idx_to_artist: Optional[Dict[int, str]] = None,
        model_path: Optional[Path] = None
    ):
        """
        Initialize with transition matrix.

        Args:
            transition_matrix: Sparse transition probability matrix
            artist_to_idx: Artist name to index mapping
            idx_to_artist: Index to artist name mapping
            model_path: Path to load models from
        """
        if transition_matrix is None and model_path:
            trans_path = model_path / "artist_transitions.npz"
            index_path = model_path / "artist_transitions.index.pkl"

            self.transition_matrix = sparse.load_npz(trans_path)
            with open(index_path, 'rb') as f:
                indices = pickle.load(f)
                self.artist_to_idx = indices['artist_to_idx']
                self.idx_to_artist = indices['idx_to_artist']
        else:
            self.transition_matrix = transition_matrix
            self.artist_to_idx = artist_to_idx or {}
            self.idx_to_artist = idx_to_artist or {}

    def transition_probability(self, from_artist: str, to_artist: str) -> float:
        """Get P(to_artist | from_artist) from transition matrix."""
        if self.transition_matrix is None:
            return 0.0

        from_idx = self.artist_to_idx.get(from_artist.lower())
        to_idx = self.artist_to_idx.get(to_artist.lower())

        if from_idx is None or to_idx is None:
            return 0.0

        return float(self.transition_matrix[from_idx, to_idx])

    def perplexity(
        self,
        test_sequences: List[List[str]],
        smoothing: float = 1e-10
    ) -> MetricResult:
        """
        Calculate perplexity of transition model on test sequences.

        Lower perplexity = better prediction of next artist.

        Perplexity = exp(-1/N * sum(log P(a_i | a_{i-1})))

        Args:
            test_sequences: List of artist sequences (from DJ sets)
            smoothing: Smoothing factor for zero probabilities

        Returns:
            MetricResult with perplexity
        """
        if self.transition_matrix is None:
            return MetricResult(name="Perplexity", value=float('inf'))

        total_log_prob = 0.0
        n_transitions = 0

        for sequence in test_sequences:
            for i in range(1, len(sequence)):
                from_artist = sequence[i-1].lower()
                to_artist = sequence[i].lower()

                prob = self.transition_probability(from_artist, to_artist)
                prob = max(prob, smoothing)  # Avoid log(0)

                total_log_prob += np.log(prob)
                n_transitions += 1

        if n_transitions == 0:
            return MetricResult(name="Perplexity", value=float('inf'))

        avg_log_prob = total_log_prob / n_transitions
        perplexity = np.exp(-avg_log_prob)

        return MetricResult(
            name="Perplexity",
            value=perplexity,
            n_samples=n_transitions
        )

    def coverage(
        self,
        test_sequences: List[List[str]],
        embedding_model=None
    ) -> Dict[str, MetricResult]:
        """
        Calculate coverage: % of transitions found in matrix vs embedding fallback.

        Args:
            test_sequences: List of artist sequences
            embedding_model: Word2Vec model for fallback

        Returns:
            Dict with matrix_coverage and embedding_coverage
        """
        matrix_hits = 0
        embedding_hits = 0
        total = 0

        wv = embedding_model.wv if embedding_model else None

        for sequence in test_sequences:
            for i in range(1, len(sequence)):
                from_artist = sequence[i-1].lower()
                to_artist = sequence[i].lower()
                total += 1

                # Check matrix
                prob = self.transition_probability(from_artist, to_artist)
                if prob > 0:
                    matrix_hits += 1

                # Check embedding
                if wv and from_artist in wv and to_artist in wv:
                    embedding_hits += 1

        return {
            "matrix_coverage": MetricResult(
                name="Transition Matrix Coverage",
                value=matrix_hits / total if total > 0 else 0.0,
                n_samples=total
            ),
            "embedding_coverage": MetricResult(
                name="Embedding Coverage",
                value=embedding_hits / total if total > 0 else 0.0,
                n_samples=total
            ),
            "total_transitions": total,
            "matrix_hits": matrix_hits,
            "embedding_hits": embedding_hits,
        }

    def source_attribution(
        self,
        playlist: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Track which selection method provided each transition in a playlist.

        Args:
            playlist: List of track dicts with 'selection_method' field

        Returns:
            Dict counting each method
        """
        method_counts = Counter()

        for track in playlist:
            method = track.get('selection_method', 'unknown')
            method_counts[method] += 1

        return dict(method_counts)


class PlaylistMetrics:
    """Metrics for evaluating generated playlists."""

    def __init__(
        self,
        embedding_model=None,
        transition_matrix: Optional[sparse.csr_matrix] = None,
        artist_to_idx: Optional[Dict[str, int]] = None
    ):
        """Initialize with models for evaluation."""
        self.embedding_model = embedding_model
        self.wv = embedding_model.wv if embedding_model else None
        self.transition_matrix = transition_matrix
        self.artist_to_idx = artist_to_idx or {}

    def coherence_score(
        self,
        playlist: List[Dict[str, Any]],
        window: int = 3
    ) -> float:
        """
        Calculate coherence as average similarity within sliding window.

        Args:
            playlist: List of track dicts with 'artist' field
            window: Window size for local coherence

        Returns:
            Coherence score (0-1)
        """
        if not self.wv or len(playlist) < 2:
            return 0.0

        similarities = []

        for i in range(len(playlist) - 1):
            window_end = min(i + window, len(playlist))
            for j in range(i + 1, window_end):
                artist_a = playlist[i].get('artist', '').lower()
                artist_b = playlist[j].get('artist', '').lower()

                if artist_a in self.wv and artist_b in self.wv:
                    sim = float(1 - cosine(self.wv[artist_a], self.wv[artist_b]))
                    similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def genre_entropy(
        self,
        playlist: List[Dict[str, Any]],
        genre_field: str = 'genre'
    ) -> float:
        """
        Calculate genre entropy (diversity measure).

        Lower entropy = more focused on single genre.

        Args:
            playlist: List of track dicts
            genre_field: Field name for genre

        Returns:
            Entropy value
        """
        genres = [t.get(genre_field) for t in playlist if t.get(genre_field)]

        if not genres:
            return 0.0

        genre_counts = Counter(genres)
        total = len(genres)
        probs = [count / total for count in genre_counts.values()]

        return float(-sum(p * np.log2(p) for p in probs if p > 0))

    def transition_quality(
        self,
        playlist: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate average transition probability for adjacent tracks.

        Args:
            playlist: List of track dicts with 'artist' field

        Returns:
            Average transition probability
        """
        if self.transition_matrix is None or len(playlist) < 2:
            return 0.0

        probs = []

        for i in range(len(playlist) - 1):
            artist_a = playlist[i].get('artist', '').lower()
            artist_b = playlist[i + 1].get('artist', '').lower()

            idx_a = self.artist_to_idx.get(artist_a)
            idx_b = self.artist_to_idx.get(artist_b)

            if idx_a is not None and idx_b is not None:
                prob = float(self.transition_matrix[idx_a, idx_b])
                probs.append(prob)

        return np.mean(probs) if probs else 0.0

    def energy_curve_adherence(
        self,
        playlist: List[Dict[str, Any]],
        target_profile: str = 'steady'
    ) -> float:
        """
        Measure how well playlist follows target energy curve.

        Args:
            playlist: List of track dicts with 'estimated_energy' and 'target_energy'
            target_profile: 'steady', 'ramp_up', 'ramp_down', 'peak'

        Returns:
            Adherence score (0-1), higher is better
        """
        if not playlist:
            return 0.0

        errors = []
        for track in playlist:
            estimated = track.get('estimated_energy', 0.5)
            target = track.get('target_energy', 0.5)
            errors.append(abs(estimated - target))

        # Adherence = 1 - mean_absolute_error
        mae = np.mean(errors) if errors else 0.0
        return max(0.0, 1.0 - mae)

    def bpm_smoothness(
        self,
        playlist: List[Dict[str, Any]],
        max_acceptable_jump: float = 8.0
    ) -> float:
        """
        Measure BPM smoothness (avoid jarring tempo changes).

        Args:
            playlist: List of track dicts with 'bpm' field
            max_acceptable_jump: BPM difference considered smooth

        Returns:
            Smoothness score (0-1)
        """
        if len(playlist) < 2:
            return 1.0

        smooth_transitions = 0
        total_transitions = 0

        for i in range(len(playlist) - 1):
            bpm_a = playlist[i].get('bpm', 0)
            bpm_b = playlist[i + 1].get('bpm', 0)

            if bpm_a > 0 and bpm_b > 0:
                diff = abs(bpm_a - bpm_b)
                if diff <= max_acceptable_jump:
                    smooth_transitions += 1
                total_transitions += 1

        return smooth_transitions / total_transitions if total_transitions > 0 else 1.0

    def aggregate_playlist_metrics(
        self,
        playlist: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate all playlist metrics."""
        return {
            "coherence": self.coherence_score(playlist),
            "genre_entropy": self.genre_entropy(playlist),
            "transition_quality": self.transition_quality(playlist),
            "energy_adherence": self.energy_curve_adherence(playlist),
            "bpm_smoothness": self.bpm_smoothness(playlist),
        }


class CrateMetrics:
    """
    Metrics for evaluating crate-based (unordered) track selection.

    These metrics focus on track selection quality rather than sequencing,
    suitable for evaluating DJ crates/pools where order doesn't matter.
    """

    def __init__(self, embedding_model=None):
        """Initialize with optional embedding model for coherence."""
        self.embedding_model = embedding_model
        self.wv = embedding_model.wv if embedding_model else None

    def genre_adherence(
        self,
        tracks: List[Dict[str, Any]],
        target_genres: List[str],
        genre_field: str = 'gemini_micro_genre'
    ) -> float:
        """
        Calculate what percentage of tracks match target genres.

        Args:
            tracks: List of track dicts
            target_genres: List of target genre strings
            genre_field: Field name for genre

        Returns:
            Adherence score (0-1)
        """
        if not tracks or not target_genres:
            return 0.0

        target_lower = [g.lower() for g in target_genres]
        matching = 0

        for track in tracks:
            track_genre = str(track.get(genre_field, '')).lower()
            if any(target in track_genre for target in target_lower):
                matching += 1

        return matching / len(tracks)

    def bpm_compliance(
        self,
        tracks: List[Dict[str, Any]],
        bpm_min: float,
        bpm_max: float,
        tolerance: float = 2.0
    ) -> float:
        """
        Calculate what percentage of tracks are within BPM range.

        Args:
            tracks: List of track dicts with 'bpm' field
            bpm_min: Minimum BPM
            bpm_max: Maximum BPM
            tolerance: BPM tolerance buffer

        Returns:
            Compliance score (0-1)
        """
        if not tracks:
            return 0.0

        compliant = 0
        valid_tracks = 0

        for track in tracks:
            bpm = track.get('bpm', 0)
            if bpm > 0:
                valid_tracks += 1
                if bpm_min - tolerance <= bpm <= bpm_max + tolerance:
                    compliant += 1

        return compliant / valid_tracks if valid_tracks > 0 else 0.0

    def artist_diversity(
        self,
        tracks: List[Dict[str, Any]],
        max_per_artist: int = 3
    ) -> float:
        """
        Calculate artist diversity (penalize too many tracks from same artist).

        Args:
            tracks: List of track dicts with 'artist' field
            max_per_artist: Maximum acceptable tracks per artist

        Returns:
            Diversity score (0-1)
        """
        if not tracks:
            return 0.0

        artist_counts = Counter(
            track.get('artist', 'unknown').lower() for track in tracks
        )

        # Calculate penalty for artists exceeding max
        excess = sum(
            max(0, count - max_per_artist)
            for count in artist_counts.values()
        )

        # Score = 1 - (excess / total)
        return max(0.0, 1.0 - excess / len(tracks))

    def selection_coherence(
        self,
        tracks: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate average pairwise coherence between all tracks.

        For crates, we want tracks that fit together even if order doesn't matter.

        Args:
            tracks: List of track dicts with 'artist' field

        Returns:
            Coherence score (0-1)
        """
        if not self.wv or len(tracks) < 2:
            return 0.0

        from scipy.spatial.distance import cosine

        similarities = []
        artists = [t.get('artist', '').lower() for t in tracks]

        # Sample pairs if too many (avoid O(n^2) for large crates)
        n = len(artists)
        if n > 50:
            # Sample 500 random pairs
            import random
            pairs = random.sample([(i, j) for i in range(n) for j in range(i+1, n)], min(500, n*(n-1)//2))
        else:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

        for i, j in pairs:
            a, b = artists[i], artists[j]
            if a in self.wv and b in self.wv:
                sim = float(1 - cosine(self.wv[a], self.wv[b]))
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def track_relevance(
        self,
        tracks: List[Dict[str, Any]],
        query_keywords: List[str]
    ) -> float:
        """
        Estimate track relevance to query based on metadata matching.

        Args:
            tracks: List of track dicts
            query_keywords: Keywords extracted from user query

        Returns:
            Relevance score (0-1)
        """
        if not tracks or not query_keywords:
            return 0.0

        keywords_lower = [k.lower() for k in query_keywords]
        relevance_scores = []

        for track in tracks:
            # Check various fields for keyword matches
            searchable = ' '.join([
                str(track.get('gemini_micro_genre', '')),
                str(track.get('gemini_mood_tags', '')),
                str(track.get('gemini_dj_use_case', '')),
                str(track.get('gemini_energy_profile', '')),
                str(track.get('artist', '')),
                str(track.get('name', ''))
            ]).lower()

            matches = sum(1 for kw in keywords_lower if kw in searchable)
            relevance_scores.append(matches / len(keywords_lower))

        return np.mean(relevance_scores)

    def aggregate_crate_metrics(
        self,
        tracks: List[Dict[str, Any]],
        target_genres: Optional[List[str]] = None,
        bpm_range: Optional[Tuple[float, float]] = None,
        query_keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all crate metrics.

        Args:
            tracks: List of track dicts
            target_genres: Target genres for adherence check
            bpm_range: (min, max) BPM range
            query_keywords: Keywords from query for relevance

        Returns:
            Dict of metric name to value
        """
        metrics = {
            "artist_diversity": self.artist_diversity(tracks),
            "selection_coherence": self.selection_coherence(tracks),
        }

        if target_genres:
            metrics["genre_adherence"] = self.genre_adherence(tracks, target_genres)

        if bpm_range:
            metrics["bpm_compliance"] = self.bpm_compliance(tracks, bpm_range[0], bpm_range[1])

        if query_keywords:
            metrics["track_relevance"] = self.track_relevance(tracks, query_keywords)

        return metrics


class StatisticalTests:
    """Statistical significance tests for comparing approaches."""

    @staticmethod
    def wilcoxon_test(
        scores_a: List[float],
        scores_b: List[float],
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired samples.

        Args:
            scores_a: Scores from approach A
            scores_b: Scores from approach B
            alternative: 'two-sided', 'greater', or 'less'

        Returns:
            Dict with statistic and p-value
        """
        if len(scores_a) != len(scores_b) or len(scores_a) < 5:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            statistic, p_value = stats.wilcoxon(
                scores_a, scores_b,
                alternative=alternative,
                zero_method='wilcox'
            )
        except ValueError:
            return {"statistic": 0.0, "p_value": 1.0}

        return {
            "statistic": float(statistic),
            "p_value": float(p_value)
        }

    @staticmethod
    def bootstrap_ci(
        scores: List[float],
        n_iterations: int = 1000,
        ci_level: float = 0.95,
        seed: int = 42
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            scores: List of scores
            n_iterations: Number of bootstrap iterations
            ci_level: Confidence level (e.g., 0.95 for 95% CI)
            seed: Random seed

        Returns:
            Tuple of (mean, ci_lower, ci_upper)
        """
        if not scores:
            return (0.0, 0.0, 0.0)

        np.random.seed(seed)
        scores_array = np.array(scores)
        n = len(scores_array)

        bootstrap_means = []
        for _ in range(n_iterations):
            sample = np.random.choice(scores_array, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - ci_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        mean = np.mean(scores_array)

        return (float(mean), float(ci_lower), float(ci_upper))

    @staticmethod
    def cohens_d(
        scores_a: List[float],
        scores_b: List[float]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

        Args:
            scores_a: Scores from approach A
            scores_b: Scores from approach B

        Returns:
            Cohen's d value
        """
        if not scores_a or not scores_b:
            return 0.0

        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        std_a = np.std(scores_a, ddof=1)
        std_b = np.std(scores_b, ddof=1)

        # Pooled standard deviation
        n_a, n_b = len(scores_a), len(scores_b)
        pooled_std = np.sqrt(
            ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
        )

        if pooled_std == 0:
            return 0.0

        return float((mean_a - mean_b) / pooled_std)

    @staticmethod
    def paired_t_test(
        scores_a: List[float],
        scores_b: List[float]
    ) -> Dict[str, float]:
        """
        Paired t-test for comparing two approaches.

        Args:
            scores_a: Scores from approach A
            scores_b: Scores from approach B

        Returns:
            Dict with statistic and p-value
        """
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            statistic, p_value = stats.ttest_rel(scores_a, scores_b)
        except ValueError:
            return {"statistic": 0.0, "p_value": 1.0}

        return {
            "statistic": float(statistic),
            "p_value": float(p_value)
        }

    @staticmethod
    def compare_approaches(
        approach_a_scores: Dict[str, List[float]],
        approach_b_scores: Dict[str, List[float]],
        n_bootstrap: int = 1000
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive comparison of two approaches across all metrics.

        Args:
            approach_a_scores: Dict of metric_name -> list of scores for approach A
            approach_b_scores: Dict of metric_name -> list of scores for approach B
            n_bootstrap: Number of bootstrap iterations

        Returns:
            Dict with comparison results for each metric
        """
        results = {}

        for metric_name in approach_a_scores.keys():
            if metric_name not in approach_b_scores:
                continue

            scores_a = approach_a_scores[metric_name]
            scores_b = approach_b_scores[metric_name]

            # Bootstrap CIs
            mean_a, ci_lower_a, ci_upper_a = StatisticalTests.bootstrap_ci(
                scores_a, n_iterations=n_bootstrap
            )
            mean_b, ci_lower_b, ci_upper_b = StatisticalTests.bootstrap_ci(
                scores_b, n_iterations=n_bootstrap
            )

            # Statistical tests
            wilcoxon = StatisticalTests.wilcoxon_test(scores_a, scores_b)
            effect_size = StatisticalTests.cohens_d(scores_a, scores_b)

            results[metric_name] = {
                "approach_a": {
                    "mean": mean_a,
                    "ci_lower": ci_lower_a,
                    "ci_upper": ci_upper_a,
                    "std": np.std(scores_a) if scores_a else 0.0,
                    "n": len(scores_a)
                },
                "approach_b": {
                    "mean": mean_b,
                    "ci_lower": ci_lower_b,
                    "ci_upper": ci_upper_b,
                    "std": np.std(scores_b) if scores_b else 0.0,
                    "n": len(scores_b)
                },
                "wilcoxon_p": wilcoxon["p_value"],
                "effect_size": effect_size,
                "significant": wilcoxon["p_value"] < 0.05,
            }

        return results


def load_ground_truth_genres(
    db_path: Path = DATA_DIR / "1001tracklists.db",
    tracks_path: Path = DATA_DIR / "tracks_master.parquet"
) -> Dict[str, str]:
    """
    Load genre ground truth from 1001TL database and local tracks.

    Returns:
        Dict mapping artist name (lowercase) to genre
    """
    artist_genres = {}

    # Load from 1001TL database
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get genre tags from tracklists
            cursor.execute("""
                SELECT DISTINCT t.artist, tl.genre_tags
                FROM tracks t
                JOIN tracklists tl ON t.tracklist_id = tl.id
                WHERE tl.genre_tags IS NOT NULL AND tl.genre_tags != ''
            """)

            for artist, genre_tags in cursor.fetchall():
                if artist and genre_tags:
                    artist_lower = artist.lower().strip()
                    # Take first genre tag
                    genres = [g.strip() for g in genre_tags.split(',')]
                    if genres:
                        artist_genres[artist_lower] = genres[0]

            conn.close()
        except Exception as e:
            print(f"Warning: Could not load genres from 1001TL: {e}")

    # Load from local tracks
    if tracks_path.exists():
        try:
            tracks = pd.read_parquet(tracks_path)

            for _, row in tracks.iterrows():
                artist = str(row.get('artist', '')).lower().strip()

                # Try various genre fields
                genre = (
                    row.get('gemini_micro_genre') or
                    row.get('beatport_genre') or
                    row.get('genre')
                )

                if artist and genre and artist not in artist_genres:
                    artist_genres[artist] = str(genre)

        except Exception as e:
            print(f"Warning: Could not load genres from tracks: {e}")

    return artist_genres


def get_known_collaborator_pairs() -> List[Tuple[str, str]]:
    """Return list of known artist collaborator pairs for evaluation."""
    return [
        # Tech House collaborators
        ("fisher", "chris lake"),
        ("camelphat", "elderbrook"),
        ("solardo", "eli brown"),
        ("dom dolla", "nelly furtado"),

        # Techno collaborators
        ("adam beyer", "cirez d"),
        ("charlotte de witte", "enrico sangiuliano"),
        ("amelie lens", "farrago"),
        ("tale of us", "mind against"),

        # House collaborators
        ("disclosure", "sam smith"),
        ("duke dumont", "jax jones"),
        ("calvin harris", "dua lipa"),
        ("martin garrix", "dua lipa"),

        # Trance collaborators
        ("above & beyond", "armin van buuren"),
        ("gareth emery", "ashley wallbridge"),

        # Drum & Bass
        ("sub focus", "wilkinson"),
        ("netsky", "digital farm animals"),

        # Cross-genre
        ("skrillex", "diplo"),
        ("flume", "toro y moi"),
        ("kaytranada", "anderson .paak"),
    ]


def get_genre_analogies() -> List[Tuple[str, str, str, str]]:
    """Return genre analogies for testing: a:b :: c:d"""
    return [
        # Tech house : Techno
        ("fisher", "chris lake", "adam beyer", "cirez d"),
        ("camelphat", "elderbrook", "charlotte de witte", "amelie lens"),

        # UK House : US House
        ("disclosure", "sam smith", "chris lake", "fisher"),

        # Progressive : Melodic Techno
        ("eric prydz", "deadmau5", "tale of us", "mind against"),

        # Mainstream EDM : Underground
        ("martin garrix", "dua lipa", "nina kraviz", "helena hauff"),
    ]


if __name__ == "__main__":
    """Quick test of metrics."""
    from gensim.models import Word2Vec

    print("=" * 60)
    print("EVALUATION METRICS TEST")
    print("=" * 60)

    # Load models
    hybrid_path = MODELS_DIR / "hybrid_artist2vec.model"
    if not hybrid_path.exists():
        print(f"Model not found: {hybrid_path}")
        exit(1)

    model = Word2Vec.load(str(hybrid_path))
    print(f"Loaded model with {len(model.wv):,} artists")

    # Test embedding metrics
    print("\n--- Embedding Metrics ---")
    emb_metrics = EmbeddingMetrics(model=model)

    # Collaborator vs random
    collab_pairs = get_known_collaborator_pairs()
    collab_results = emb_metrics.collaborator_vs_random_similarity(collab_pairs)
    for name, result in collab_results.items():
        print(f"  {result}")

    # Analogies
    analogies = get_genre_analogies()
    analogy_result = emb_metrics.analogy_accuracy(analogies)
    print(f"  {analogy_result}")

    # Source tag analysis
    print("\n--- Source Tag Analysis ---")
    source_analysis = emb_metrics.source_tag_analysis()
    for tag in ["SOURCE_1001TL", "SOURCE_DJMIX", "SOURCE_MPD"]:
        if tag in source_analysis:
            print(f"  {tag}:")
            for artist, score in source_analysis[tag]["top_artists"][:5]:
                print(f"    {artist}: {score:.3f}")

    # Load ground truth genres
    print("\n--- Loading Ground Truth ---")
    artist_genres = load_ground_truth_genres()
    print(f"  Loaded {len(artist_genres):,} artist-genre mappings")

    # Genre clustering
    if artist_genres:
        cluster_result = emb_metrics.genre_clustering_silhouette(artist_genres)
        print(f"  {cluster_result}")

    # Test retrieval metrics
    print("\n--- Retrieval Metrics ---")
    ret_metrics = RetrievalMetrics(model=model, artist_genres=artist_genres)

    test_artists = ["fisher", "adam beyer", "disclosure", "eric prydz", "charlotte de witte"]
    test_artists = [a for a in test_artists if a in model.wv]

    ret_results = ret_metrics.aggregate_retrieval_metrics(test_artists)
    for name, result in ret_results.items():
        print(f"  {result}")

    # Test statistical tests
    print("\n--- Statistical Tests ---")
    scores_a = [0.8, 0.75, 0.82, 0.79, 0.85, 0.78, 0.81, 0.77, 0.83, 0.80]
    scores_b = [0.6, 0.58, 0.65, 0.62, 0.59, 0.63, 0.61, 0.57, 0.64, 0.60]

    wilcoxon = StatisticalTests.wilcoxon_test(scores_a, scores_b)
    print(f"  Wilcoxon p-value: {wilcoxon['p_value']:.4f}")

    effect = StatisticalTests.cohens_d(scores_a, scores_b)
    print(f"  Cohen's d: {effect:.4f}")

    mean, ci_low, ci_high = StatisticalTests.bootstrap_ci(scores_a)
    print(f"  Bootstrap CI: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    print("\n" + "=" * 60)
    print("METRICS TEST COMPLETE")
    print("=" * 60)
